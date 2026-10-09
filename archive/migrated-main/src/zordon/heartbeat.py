from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .memory import MemoryStore
from .tools import ToolRegistry


@dataclass(frozen=True)
class Pulse:
    assistant_name: str
    model: str
    transcription_model: str
    tts_model: str
    tts_voice: str
    memory_count: int
    memory_path: Path | None
    audit_path: Path | None
    tool_names: list[str]

    def as_text(self) -> str:
        lines = [
            f"{self.assistant_name} pulse",
            f"- model: {self.model}",
            f"- voice: {self.tts_voice} via {self.tts_model}",
            f"- transcription: {self.transcription_model}",
            f"- memories: {self.memory_count}",
            f"- tools: {', '.join(self.tool_names)}",
        ]
        if self.memory_path is not None:
            lines.append(f"- memory file: {self.memory_path}")
        if self.audit_path is not None:
            lines.append(f"- audit log: {self.audit_path}")
        return "\n".join(lines)


def build_pulse(
    config: Config,
    memory: MemoryStore,
    tools: ToolRegistry,
    audit_path: Path | None,
) -> Pulse:
    return Pulse(
        assistant_name=config.assistant_name,
        model=config.model,
        transcription_model=config.transcription_model,
        tts_model=config.tts_model,
        tts_voice=config.tts_voice,
        memory_count=len(memory.all()),
        memory_path=memory.path,
        audit_path=audit_path,
        tool_names=tools.names(),
    )
