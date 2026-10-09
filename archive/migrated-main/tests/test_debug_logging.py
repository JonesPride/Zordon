import json
from pathlib import Path

from zordon.debug_logging import DebugLogger


def test_debug_logger_records_only_allowlisted_metadata(tmp_path: Path) -> None:
    path = tmp_path / "debug.jsonl"
    logger = DebugLogger(path)

    logger.event(
        "turn_failed",
        duration_ms=12,
        message_count=4,
        api_key="sk-secret",
        user_text="private prompt",
        exception="private traceback",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "turn_failed"
    assert record["duration_ms"] == 12
    assert record["message_count"] == 4
    rendered = json.dumps(record)
    assert "sk-secret" not in rendered
    assert "private prompt" not in rendered
    assert "private traceback" not in rendered


def test_disabled_debug_logger_writes_nothing(tmp_path: Path) -> None:
    logger = DebugLogger(None)
    logger.event("turn_started", message_count=1)
    assert list(tmp_path.iterdir()) == []
