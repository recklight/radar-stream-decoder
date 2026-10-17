"""Collection rules for the whole project (tests and doctests).

The viewer module imports PySide6 and matplotlib at import time. Without the
``gui`` extra its doctest collection would fail, so it is skipped instead;
``tests/test_viz.py`` skips itself in the same situation.
"""

from __future__ import annotations

import importlib.util

collect_ignore: list[str] = []

if importlib.util.find_spec("PySide6") is None or importlib.util.find_spec("matplotlib") is None:
    collect_ignore.append("src/radar_stream_decoder/viz.py")
