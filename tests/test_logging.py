"""Package isolation, repeatable configuration and structured exception output."""

import io
import json
import logging
import subprocess
import sys

import pytest

from embodiedforge.logging import (
    JsonFormatter,
    add_logger_handler,
    auto_configure_debug_logging,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def restore_package_logger():
    logger = logging.getLogger("embodiedforge")
    previous = (list(logger.handlers), logger.level, logger.propagate)
    yield
    setup_logging(handlers=[])
    logger.handlers[:] = previous[0]
    logger.setLevel(previous[1])
    logger.propagate = previous[2]


def test_configuration_does_not_mutate_root_or_caller_handlers():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handlers = [handler]
    root = logging.getLogger()
    original = (list(root.handlers), root.level)
    setup_logging("INFO", handlers, with_json_format=True)
    setup_logging("INFO", handlers, with_json_format=True)
    add_logger_handler(handler)
    get_logger("worker", run_id="run-1").info("一条日志")
    rows = stream.getvalue().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["context"] == {"run_id": "run-1"}
    assert row["logger"] == "embodiedforge.worker"
    assert (
        row["source"]["function"]
        == "test_configuration_does_not_mutate_root_or_caller_handlers"
    )
    assert handlers == [handler]
    assert (root.handlers, root.level) == original


def test_json_exception_remains_one_line_with_traceback():
    stream = io.StringIO()
    setup_logging("DEBUG", [logging.StreamHandler(stream)], with_json_format=True)
    try:
        raise ValueError("错误\n原因")
    except ValueError:
        get_logger("data", episode_id=4).exception("写入失败")
    rows = stream.getvalue().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert "ValueError" in payload["exception"]
    assert payload["message"] == "写入失败"


def test_borrowed_handler_is_detached_without_closing():
    class Handler(logging.Handler):
        closed_by_owner = False

        def emit(self, record):
            pass

        def close(self):
            self.closed_by_owner = True
            super().close()

    handler = Handler()
    setup_logging(handlers=[handler])
    setup_logging(handlers=[])
    assert not handler.closed_by_owner


def test_environment_config_is_explicit_and_invalid_flag_is_harmless(monkeypatch):
    monkeypatch.setenv("EMBODIEDFORGE_DEBUG", "nonsense")
    assert not auto_configure_debug_logging()
    monkeypatch.setenv("EMBODIEDFORGE_DEBUG", "1")
    monkeypatch.setenv("EMBODIEDFORGE_JSON_FORMAT_LOG", "true")
    assert auto_configure_debug_logging()
    assert isinstance(
        logging.getLogger("embodiedforge").handlers[0].formatter, JsonFormatter
    )


def test_cli_logs_do_not_corrupt_stdout_json():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodiedforge",
            "plan",
            "--log-json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["physics"] == "numpy"
    row = json.loads(result.stderr)
    assert row["level"] == "INFO"
    assert "Starting plan" in row["message"]


def test_invalid_configuration_preserves_existing_handlers():
    handler = logging.StreamHandler(io.StringIO())
    setup_logging(handlers=[handler])
    with pytest.raises(ValueError):
        setup_logging("not-a-level")
    with pytest.raises(TypeError):
        setup_logging(handlers=[object()])
    assert logging.getLogger("embodiedforge").handlers == [handler]
