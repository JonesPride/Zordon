from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ROOT, state_root


@dataclass
class AuditLog:
    path: Path | None = None
    events: list[dict[str, Any]] | None = None
    write_errors: list[str] | None = None

    def record(self, event: str, **fields: Any) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        if self.events is not None:
            self.events.append(entry)
        if self.path is None:
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except OSError as exc:
            self._remember_write_error(exc)

    def _remember_write_error(self, exc: OSError) -> None:
        if self.write_errors is not None:
            self.write_errors.append(str(exc))


def default_audit_log(root: Path) -> AuditLog:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    base = state_root() if root == ROOT else root
    return AuditLog(path=base / "logs" / f"zordon-{today}.jsonl")
