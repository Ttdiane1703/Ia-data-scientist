"""Console configuration shared by the command-line components."""

import sys


def configure_console() -> None:
    """Use UTF-8 output when the active Windows terminal supports reconfigure."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
