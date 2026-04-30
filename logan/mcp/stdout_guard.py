import sys
import io
import logging
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """Redirect stdout/stderr to StringIO buffers.

    MCP stdio transport uses stdout for JSON-RPC. Any print() or
    logging-to-stdout from Logan internals would corrupt the protocol.
    Wrap all Logan class calls in this context manager.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

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
