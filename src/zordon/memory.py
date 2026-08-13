from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ROOT, state_root


@dataclass(frozen=True)
class Memory:
    key: str
    value: str
    category: str
    created_at: str

    def as_model_text(self) -> str:
        return f"- {self.key}: {self.value} ({self.category})"


class MemoryStore:
    def __init__(self, path: Path | None = None, entries: list[Memory] | None = None):
        self.path = path
        self._entries = entries
        if self._entries is None:
            self._entries = self._load()

    def remember(self, key: str, value: str, category: str = "preference") -> Memory:
        key = _required_text(key, "key")
        value = _required_text(value, "value")
        category = _required_text(category, "category")

        memory = Memory(
            key=key,
            value=value,
            category=category,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._entries = [
            existing
            for existing in self._entries
            if existing.key.lower() != key.lower()
        ]
        self._entries.append(memory)
        if self.path is None:
            return memory

        self._append(memory)
        return memory

    def all(self) -> list[Memory]:
        return list(self._entries)

    def forget(self, key: str) -> bool:
        key = _required_text(key, "key")
        original_count = len(self._entries)
        self._entries = [
            existing
            for existing in self._entries
            if existing.key.lower() != key.lower()
        ]
        removed = len(self._entries) != original_count
        if removed:
            self._rewrite()
        return removed

    def summary(self, limit: int = 12) -> str:
        entries = self._entries[-limit:]
        if not entries:
            return ""
        return "\n".join(memory.as_model_text() for memory in entries)

    def raw_lines(self) -> list[str]:
        return [
            json.dumps(_memory_to_dict(memory), ensure_ascii=True)
            for memory in self._entries
        ]

    def _load(self) -> list[Memory]:
        if self.path is None or not self.path.exists():
            return []

        entries_by_key: dict[str, Memory] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                memory = _memory_from_dict(data)
                entries_by_key[memory.key.lower()] = memory
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return list(entries_by_key.values())

    def _append(self, memory: Memory) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_memory_to_dict(memory), ensure_ascii=True) + "\n"
        _append_text(self.path, line)

    def _rewrite(self) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(_memory_to_dict(memory), ensure_ascii=True) + "\n"
            for memory in self._entries
        ]
        self.path.write_text("".join(lines), encoding="utf-8")


def default_memory_store(root: Path) -> MemoryStore:
    base = state_root() if root == ROOT else root
    store = MemoryStore(path=base / "data" / "memory.jsonl")
    if base != root:
        _migrate_legacy_memories(root / "data" / "memory.jsonl", store)
    return store


def _migrate_legacy_memories(legacy_path: Path, store: MemoryStore) -> None:
    if not legacy_path.exists():
        return
    existing_keys = {memory.key.lower() for memory in store.all()}
    for memory in MemoryStore(path=legacy_path).all():
        if memory.key.lower() in existing_keys:
            continue
        store.remember(memory.key, memory.value, memory.category)
        existing_keys.add(memory.key.lower())


def _memory_from_dict(data: dict[str, Any]) -> Memory:
    return Memory(
        key=_required_text(data.get("key"), "key"),
        value=_required_text(data.get("value"), "value"),
        category=_required_text(data.get("category", "preference"), "category"),
        created_at=_required_text(data.get("created_at"), "created_at"),
    )


def _memory_to_dict(memory: Memory) -> dict[str, str]:
    return {
        "key": memory.key,
        "value": memory.value,
        "category": memory.category,
        "created_at": memory.created_at,
    }


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _append_text(path: Path, text: str) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(str(path), flags)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)
