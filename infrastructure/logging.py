"""Application logging configuration.

Logs are written to ``logs/application.log`` with size-based rotation instead
of truncating the file on every Streamlit rerun.  Rotation keeps recent
history available for debugging without allowing the file to grow without
bound.

The API key is never logged; callers log event descriptions and error
messages only.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_FILE = _PROJECT_ROOT / "logs" / "application.log"

_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3
_HANDLER_MARKER = "_oncolink_file_handler"
_configured = False


def setup_logging(
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
) -> Path:
    """Configure application logging once per process.

    Streamlit reruns the script on every interaction; the ``_configured``
    guard prevents duplicate handlers and duplicated log lines.  Returns the
    active log file path so callers/tests can report it.
    """
    global _configured

    path = Path(log_file) if log_file else DEFAULT_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    if _configured and any(getattr(h, _HANDLER_MARKER, False) for h in root.handlers):
        return path

    # Remove only handlers installed by this module (not pytest/caplog).
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    _configured = True
    logging.getLogger(__name__).info("Logging initialised at %s", path)
    return path
