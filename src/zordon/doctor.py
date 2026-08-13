from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT, Config, is_temp_state_root, state_root
from .drafts import drafts_dir
from .memory import MemoryStore
from .tools import ToolRegistry


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    def as_text(self) -> str:
        status = "OK" if self.ok else "WARN"
        return f"- {status} {self.name}: {self.detail}"


def run_doctor(config: Config, memory: MemoryStore, tools: ToolRegistry, root: Path) -> list[Check]:
    return [
        _check_api_key(),
        _check_state_root(root),
        _check_config(config),
        _check_memory(memory),
        _check_drafts(root),
        _check_tools(tools),
        _check_optional_dependency("sounddevice"),
        _check_optional_dependency("keyboard"),
    ]


def doctor_text(config: Config, memory: MemoryStore, tools: ToolRegistry, root: Path) -> str:
    checks = run_doctor(config, memory, tools, root)
    summary = "healthy" if all(check.ok for check in checks) else "needs attention"
    lines = [f"Zordon doctor: {summary}"]
    lines.extend(check.as_text() for check in checks)
    return "\n".join(lines)


def _check_api_key() -> Check:
    if os.getenv("OPENAI_API_KEY"):
        return Check("OPENAI_API_KEY", True, "set")
    return Check("OPENAI_API_KEY", False, "not set; model, transcription, and speech calls will fail")


def _check_state_root(root: Path) -> Check:
    actual_root = state_root() if root == ROOT else root
    if is_temp_state_root(actual_root):
        return Check("state", False, f"using temporary storage at {actual_root}; set ZORDON_STATE_DIR or use AppData")
    return Check("state", True, f"durable storage at {actual_root}")


def _check_config(config: Config) -> Check:
    missing = [
        name
        for name, value in [
            ("assistant_name", config.assistant_name),
            ("model", config.model),
            ("transcription_model", config.transcription_model),
            ("tts_model", config.tts_model),
            ("tts_voice", config.tts_voice),
            ("push_to_talk_key", config.push_to_talk_key),
        ]
        if not str(value).strip()
    ]
    if missing:
        return Check("config", False, f"missing values: {', '.join(missing)}")
    if config.audio_sample_rate <= 0 or config.request_timeout_seconds <= 0:
        return Check("config", False, "audio_sample_rate and request_timeout_seconds must be positive")
    return Check("config", True, f"{config.assistant_name} using {config.model}")


def _check_memory(memory: MemoryStore) -> Check:
    if memory.path is None:
        return Check("memory", True, f"{len(memory.all())} in-memory entries")
    parent = memory.path.parent
    if parent.exists() or parent.parent.exists():
        return Check("memory", True, f"{len(memory.all())} entries at {memory.path}")
    return Check("memory", False, f"parent path is not reachable: {parent}")


def _check_drafts(root: Path) -> Check:
    directory = drafts_dir(root)
    if directory.exists():
        return Check("drafts", True, f"folder exists at {directory}")
    return Check("drafts", True, f"folder will be created at {directory}")


def _check_tools(tools: ToolRegistry) -> Check:
    names = tools.names()
    if not names:
        return Check("tools", False, "no tools registered")
    return Check("tools", True, f"{len(names)} registered: {', '.join(names)}")


def _check_optional_dependency(name: str) -> Check:
    if importlib.util.find_spec(name) is not None:
        return Check(name, True, "installed")
    return Check(
        name,
        False,
        "not installed; run python -m pip install -r requirements-voice.txt for push-to-talk voice",
    )
