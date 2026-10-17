"""Logging setup shared by the command line, the viewer and the examples.

The library never prints to report progress: diagnostics go through
:mod:`logging`, so an application embedding the decoder decides where they go.
Tables, arrays and reports are written to files; only the command line echoes
a short summary to the terminal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Attribute that marks the handlers installed by :func:`configure_logging`.
_OWN_HANDLER = "_radar_stream_decoder_handler"


def configure_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Attach a console handler, and optionally a file handler, to the root logger.

    Calling it again replaces the handlers of the previous call instead of
    stacking duplicates; handlers installed by anything else are left alone.

    Parameters
    ----------
    level : str or int, optional
        ``"DEBUG"``, ``"INFO"``, ``"WARNING"`` ... or a numeric level.
    log_file : str or pathlib.Path, optional
        Also append every record to this file.

    Raises
    ------
    ValueError
        For an unknown level name.

    Examples
    --------
    >>> configure_logging("WARNING")
    >>> logging.getLogger("radar_stream_decoder").getEffectiveLevel() == logging.WARNING
    True
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(
                f"unknown logging level {level!r}; use DEBUG, INFO, WARNING, ERROR or CRITICAL"
            )
    else:
        resolved = int(level)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _OWN_HANDLER, False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _OWN_HANDLER, True)
        root.addHandler(handler)
    root.setLevel(resolved)
