import os
import sys
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


LOG_DIR = os.path.join(Path.home(), ".logan", "logs")
LOG_FILE = os.path.join(LOG_DIR, "mcp.log")


@contextmanager
def suppress_stdout():
    """Redirect stdout/stderr to ~/.logan/logs/mcp.log.

    MCP stdio transport uses stdout for JSON-RPC. Any print() or
    logging-to-stdout from Logan internals would corrupt the protocol.
    Wrap all Logan class calls in this context manager.

    Uses a real file instead of StringIO so that multiprocessing-based
    libraries (pandarallel) can safely inherit the file descriptors in
    forked workers.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(LOG_FILE, "a")
    log_file.write(f"\n--- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
    log_file.flush()
    sys.stdout = log_file
    sys.stderr = log_file

    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers = [
        h for h in root.handlers
        if not (isinstance(h, logging.StreamHandler) and h.stream in (old_stdout, old_stderr))
    ]

    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        root.handlers = old_handlers
        log_file.close()
