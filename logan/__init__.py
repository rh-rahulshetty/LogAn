"""
Logan - Log Analysis Tool

A powerful tool for log preprocessing, templatization, and anomaly detection.

Usage:
    from logan.cli import main
    main()

Or from command line:
    logan analyze --input-files logs/ --output-dir ./output
"""

from logan._version import __version__

__author__ = "Logan Team"

# Expose CLI entry point
from logan.cli import cli, main

__all__ = ["cli", "main", "__version__"]

