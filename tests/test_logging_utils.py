"""Logging setup: replaces only its own handlers, never an application's."""

from __future__ import annotations

import logging

import pytest

from radar_stream_decoder.logging_utils import _OWN_HANDLER, configure_logging


@pytest.fixture
def root_logger():
    root = logging.getLogger()
    before_handlers, before_level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if handler not in before_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(before_level)


def own(root: logging.Logger) -> list[logging.Handler]:
    return [h for h in root.handlers if getattr(h, _OWN_HANDLER, False)]


def test_repeated_calls_replace_their_own_handlers(root_logger, tmp_path):
    foreign = logging.NullHandler()
    root_logger.addHandler(foreign)
    configure_logging("INFO")
    configure_logging("DEBUG", log_file=tmp_path / "logs" / "run.log")
    assert len(own(root_logger)) == 2  # console + file, not stacked
    assert foreign in root_logger.handlers  # the application's handler survives
    assert root_logger.level == logging.DEBUG
    logging.getLogger("radar_stream_decoder.test").debug("written to the file")
    for handler in own(root_logger):
        handler.flush()
    assert "written to the file" in (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    configure_logging(logging.WARNING)
    assert len(own(root_logger)) == 1 and foreign in root_logger.handlers


def test_unknown_level_is_rejected(root_logger):
    with pytest.raises(ValueError, match="unknown logging level"):
        configure_logging("chatty")
