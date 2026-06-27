"""General helper functions used by multiple packages."""

from __future__ import annotations

import re
from pathlib import Path
import sys


def safe_filename(name: str) -> str:
    """Convert arbitrary text into a filesystem-safe filename stem."""

    return re.sub(r"[^\w\-]+", "_", name.strip())


def read_if_exists(*paths: Path) -> Path | None:
    """Return the first path that exists from a list of candidates."""

    for path in paths:
        if path.exists():
            return path
    return None


def configure_utf8_console() -> None:
    """Force UTF-8 for stdout and stderr when running on Windows terminals."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
