import logging
import pytest
from src.utils.logger import get_logger


@pytest.fixture(autouse=True)
def reset_loggers():
    yield
    logging.Logger.manager.loggerDict.clear()


def test_returns_logger_with_correct_name():
    logger = get_logger("test_module", log_to_file=False)
    assert logger.name == "test_module"


def test_default_level_is_info():
    logger = get_logger("test_level", log_to_file=False)
    assert logger.level == logging.INFO


def test_custom_level_is_set():
    logger = get_logger("test_debug", log_to_file=False, level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_console_handler_always_added():
    logger = get_logger("test_console", log_to_file=False)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_file_handler_added_when_enabled(tmp_path):
    logger = get_logger("test_file", log_dir=str(tmp_path), log_to_file=True)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_file_handler_not_added_when_disabled():
    logger = get_logger("test_nofile", log_to_file=False)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_log_file_created_on_disk(tmp_path):
    get_logger("test_disk", log_dir=str(tmp_path), log_to_file=True)
    log_files = list(tmp_path.glob("test_disk_*.log"))
    assert len(log_files) == 1


def test_log_file_named_with_logger_name(tmp_path):
    get_logger("myapp", log_dir=str(tmp_path), log_to_file=True)
    log_files = list(tmp_path.glob("myapp_*.log"))
    assert len(log_files) == 1


def test_duplicate_call_returns_same_logger():
    logger1 = get_logger("test_dup", log_to_file=False)
    logger2 = get_logger("test_dup", log_to_file=False)
    assert logger1 is logger2


def test_no_duplicate_handlers_on_repeated_call():
    logger = get_logger("test_handlers", log_to_file=False)
    get_logger("test_handlers", log_to_file=False)
    assert len(logger.handlers) == 1


def test_log_dir_created_if_not_exists(tmp_path):
    nested_dir = str(tmp_path / "a" / "b" / "c")
    get_logger("test_nested", log_dir=nested_dir, log_to_file=True)
    from pathlib import Path
    assert Path(nested_dir).exists()
