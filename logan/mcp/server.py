import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context

from logan.mcp.stdout_guard import suppress_stdout

logger = logging.getLogger("logan.mcp")

DEFAULT_LOGAN_HOME = os.path.join(Path.home(), ".logan")


def _default_output_dir() -> str:
    """Return a timestamped output directory under LOGAN_OUTPUT_DIR or ~/.logan/runs/."""
    base = os.environ.get("LOGAN_OUTPUT_DIR", os.path.join(DEFAULT_LOGAN_HOME, "runs"))
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, run_name)


# ---------------------------------------------------------------------------
# Server state: holds the cached ML model across tool calls
# ---------------------------------------------------------------------------

class _ServerState:
    def __init__(self):
        self._model_manager = None
        self._model_type = None
        self._model_name = None

    def get_or_create_model_manager(self, model_type, model):
        """Return cached ModelManager, creating one on first call."""
        from logan.log_diagnosis.models import ModelManager

        type_val = model_type.value if hasattr(model_type, "value") else str(model_type)
        model_val = model.value if hasattr(model, "value") else str(model)

        if (
            self._model_manager is not None
            and self._model_type == type_val
            and self._model_name == model_val
        ):
            return self._model_manager

        with suppress_stdout():
            mgr = ModelManager(model_type, model)
        self._model_manager = mgr
        self._model_type = type_val
        self._model_name = model_val
        return mgr


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(server: FastMCP):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LOGAN_DISABLE_PANDARALLEL"] = "1"
    state = _ServerState()
    try:
        yield state
    finally:
        state._model_manager = None


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "logan",
    instructions=(
        "LogAn is an intelligent log analysis tool. Use analyze_logs to run "
        "the full pipeline on log files. Use get_run_summary to retrieve "
        "structured results from a completed run. Use read_log_sample to "
        "peek at raw log file contents."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_type(model_type_str: str):
    from logan.log_diagnosis.models import ModelType
    try:
        return ModelType(model_type_str)
    except ValueError:
        raise ValueError(
            f"Invalid model_type: '{model_type_str}'. "
            f"Choose from: {', '.join(m.value for m in ModelType)}"
        )


def _resolve_model(model_str: str):
    from logan.log_diagnosis.models.model_zero_shot_classifer import ZeroShotModels
    for m in ZeroShotModels:
        if m.name.lower() == model_str.lower() or m.value == model_str:
            return m
    return model_str


def _read_json_file(path: str):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _collect_results(output_dir: str) -> dict:
    """Read metrics and structured data written by the pipeline."""
    results = {}

    for name in ("preprocessing", "drain", "anomaly"):
        data = _read_json_file(os.path.join(output_dir, "metrics", f"{name}.json"))
        if data:
            results[f"{name}_metrics"] = data

    timeline = _read_json_file(
        os.path.join(output_dir, "developer_debug_files", "golden_signal_timeline.json")
    )
    if timeline:
        results["golden_signal_timeline"] = timeline

    signal_map = _read_json_file(
        os.path.join(output_dir, "developer_debug_files", "temp_id_to_signal_map.json")
    )
    if signal_map:
        results["template_signal_map"] = signal_map

    anomaly_html = os.path.join(output_dir, "log_diagnosis", "anomalies.html")
    summary_html = os.path.join(output_dir, "log_diagnosis", "summary.html")
    results["report_paths"] = {
        "anomalies_html": anomaly_html if os.path.isfile(anomaly_html) else None,
        "summary_html": summary_html if os.path.isfile(summary_html) else None,
    }

    return results


# ---------------------------------------------------------------------------
# Tool: analyze_logs
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_logs(
    files: list[str],
    output_dir: str = "",
    time_range: str = "all-data",
    model_type: str = "zero_shot",
    model: str = "crossencoder",
    debug_mode: bool = True,
    process_all_files: bool = False,
    process_log_files: bool = True,
    process_txt_files: bool = False,
    clean_up: bool = False,
    ctx: Context = None,
) -> dict:
    """Run the full LogAn analysis pipeline on log files.

    Performs preprocessing, Drain3 templatization, and ML-based anomaly
    detection (golden signals + fault categories). Generates HTML reports
    and returns structured results.

    Model loading on the first call may take 30-60 seconds; subsequent
    calls reuse the cached model.

    Args:
        files: Paths to log files or directories to analyze.
        output_dir: Directory where reports and artifacts are written. Defaults to ~/.logan/runs/<timestamp>. Override the base path with the LOGAN_OUTPUT_DIR environment variable.
        time_range: Time range filter. Options: all-data, 1-day, 2-day, ..., 1-week, 2-week, 1-month.
        model_type: Model type for classification. Options: zero_shot, similarity, custom.
        model: Model name. Built-in: crossencoder, bart. Or a HuggingFace model name.
        debug_mode: Save additional debug artifacts.
        process_all_files: Process all text-based files regardless of extension.
        process_log_files: Process .log files found in directories.
        process_txt_files: Process .txt files found in directories.
        clean_up: Remove existing output_dir before running.
    """
    loop = asyncio.get_event_loop()
    state: _ServerState = ctx.request_context.lifespan_context

    if not output_dir:
        output_dir = _default_output_dir()

    # Validate inputs
    for f in files:
        if not os.path.exists(f):
            return {"status": "error", "message": f"File not found: {f}"}

    resolved_model_type = _resolve_model_type(model_type)
    resolved_model = _resolve_model(model)
    debug_str = "true" if debug_mode else "false"

    # Step 0: prepare output directory
    from logan.log_diagnosis.utils import prepare_output_dir
    prepare_output_dir(output_dir, clean_up)

    # Step 1: preprocessing
    await ctx.report_progress(0, 3)
    await ctx.info("Preprocessing log files...")

    def _run_preprocessing():
        from logan.preprocessing.preprocessing import Preprocessing
        with suppress_stdout():
            pp = Preprocessing(debug_str)
            pp.preprocess(
                files, time_range, output_dir,
                process_all_files, process_log_files, process_txt_files,
            )
        return pp.df

    df = await loop.run_in_executor(None, _run_preprocessing)

    if df is None or len(df) == 0:
        return {
            "status": "error",
            "message": "No log lines could be extracted from the input files.",
        }

    log_lines = len(df)

    # Step 2: Drain3 templatization
    await ctx.report_progress(1, 3)
    await ctx.info("Generating log templates (Drain3)...")

    def _run_drain():
        from logan.drain.run_drain import Templatizer
        with suppress_stdout():
            drain_config = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "drain", "drain3.ini"
            )
            templatizer = Templatizer(debug_mode=debug_str, config_path=drain_config)
            templatizer.miner(
                df, output_dir,
                os.path.join(output_dir, "test_templates", "tm-test.templates.json"),
            )
        return templatizer.df

    templatized_df = await loop.run_in_executor(None, _run_drain)
    template_count = templatized_df["test_ids"].nunique() if "test_ids" in templatized_df.columns else 0

    # Step 3: anomaly detection
    await ctx.report_progress(2, 3)
    await ctx.info("Detecting anomalies (model loading may take 30-60s on first call)...")

    def _run_anomaly():
        from logan.log_diagnosis.anomaly import Anomaly

        model_mgr = state.get_or_create_model_manager(resolved_model_type, resolved_model)

        anomaly = object.__new__(Anomaly)
        anomaly.debug_mode = debug_str
        anomaly.model_manager = model_mgr

        with suppress_stdout():
            anomaly.get_anomaly_report(templatized_df, output_dir)

    await loop.run_in_executor(None, _run_anomaly)

    await ctx.report_progress(3, 3)

    results = _collect_results(output_dir)

    # Summarize golden signal distribution from timeline data
    gs_distribution = {}
    timeline = results.get("golden_signal_timeline", {})
    for bin_entry in timeline.get("bins", []):
        for signal in timeline.get("signals", []):
            gs_distribution[signal] = gs_distribution.get(signal, 0) + bin_entry.get(signal, 0)

    return {
        "status": "success",
        "log_lines_processed": log_lines,
        "templates_found": template_count,
        "golden_signal_distribution": gs_distribution,
        **results,
    }


# ---------------------------------------------------------------------------
# Tool: get_run_summary
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_run_summary(output_dir: str) -> dict:
    """Get structured results from a completed LogAn analysis run.

    Returns metrics, golden signal timeline, template-to-signal mapping,
    and report paths as structured JSON that agents can reason about.

    Args:
        output_dir: The output directory from a previous analyze_logs call.
    """
    if not os.path.isdir(output_dir):
        return {"status": "error", "message": f"Directory not found: {output_dir}"}

    log_diagnosis_dir = os.path.join(output_dir, "log_diagnosis")
    if not os.path.isdir(log_diagnosis_dir):
        return {
            "status": "error",
            "message": f"No analysis results found in {output_dir}. Run analyze_logs first.",
        }

    results = _collect_results(output_dir)

    gs_distribution = {}
    timeline = results.get("golden_signal_timeline", {})
    for bin_entry in timeline.get("bins", []):
        for signal in timeline.get("signals", []):
            gs_distribution[signal] = gs_distribution.get(signal, 0) + bin_entry.get(signal, 0)

    return {"status": "success", "golden_signal_distribution": gs_distribution, **results}


# ---------------------------------------------------------------------------
# Tool: read_log_sample
# ---------------------------------------------------------------------------

@mcp.tool()
async def read_log_sample(
    file_path: str,
    num_lines: int = 50,
    offset: int = 0,
) -> dict:
    """Read a sample of lines from a log file.

    Useful for inspecting log format and content before running analysis.

    Args:
        file_path: Path to the log file.
        num_lines: Number of lines to read (default 50).
        offset: Line number to start reading from (0-based, default 0).
    """
    if not os.path.isfile(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
    except OSError as e:
        return {"status": "error", "message": f"Cannot read file: {e}"}

    total_lines = len(all_lines)
    end = min(offset + num_lines, total_lines)
    selected = [line.rstrip("\n\r") for line in all_lines[offset:end]]

    return {
        "status": "success",
        "file_path": file_path,
        "total_lines": total_lines,
        "offset": offset,
        "num_lines": len(selected),
        "lines": selected,
    }
