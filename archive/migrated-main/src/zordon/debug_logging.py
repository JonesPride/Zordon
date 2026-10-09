from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

_SAFE_FIELDS = frozenset({"duration_ms", "message_count", "output_chars", "provider", "status"})


class DebugLogger:
    """Append opt-in diagnostic metadata without conversation or exception content."""

    def __init__(self, path: Path | None) -> None:
        self._path = path

    def event(self, event: str, **fields: object) -> None:
        if self._path is None:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **{key: value for key, value in fields.items() if key in _SAFE_FIELDS},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        if os.name != "nt":
            self._path.chmod(0o600)
