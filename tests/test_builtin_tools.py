from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from zordon.config import Tier2Limits
from zordon.tools import InvalidArgumentsError, ToolContext, ToolResult
from zordon.tools.builtin import build_builtin_registry, build_system_prompt


@dataclass
class RecordingDocuments:
    calls: list[tuple[object, ...]] = field(default_factory=list)

    def search(self, query, folder_ids, max_results):
        self.calls.append(("search", query, folder_ids, max_results))
        return ToolResult.success("searched")

    def read(self, folder_id, relative_path, start_char, max_chars):
        self.calls.append(("read", folder_id, relative_path, start_char, max_chars))
        return ToolResult.success("read")


@dataclass
class RecordingAudio:
    calls: list[tuple[object, ...]] = field(default_factory=list)

    def search(self, query, folder_ids, max_results, *, turn_number):
        self.calls.append((query, folder_ids, max_results, turn_number))
        return ToolResult.success("searched")


@dataclass
class RecordingPlayback:
    calls: list[tuple[object, ...]] = field(default_factory=list)

    def play(self, audio_id, turn_number):
        self.calls.append(("play", audio_id, turn_number))
        return ToolResult.success("played")

    def pause(self):
        self.calls.append(("pause",))
        return ToolResult.success("paused")

    def resume(self):
        self.calls.append(("resume",))
        return ToolResult.success("resumed")

    def stop(self):
        self.calls.append(("stop",))
        return ToolResult.success("stopped")

    def now_playing(self):
        self.calls.append(("now_playing",))
        return ToolResult.success("status")


def make_registry(limits: Tier2Limits | None = None):
    documents = RecordingDocuments()
    audio = RecordingAudio()
    playback = RecordingPlayback()
    registry = build_builtin_registry(documents, audio, playback, limits or Tier2Limits())
    return registry, documents, audio, playback


def test_registry_exposes_exact_builtins_with_strict_schemas() -> None:
    registry, _, _, _ = make_registry()
    tools = registry.list_tools()

    assert tuple(tool.name for tool in tools) == (
        "get_current_datetime",
        "calculate",
        "search_documents",
        "read_document",
        "search_audio",
        "play_audio",
        "pause_audio",
        "resume_audio",
        "stop_audio",
        "now_playing",
    )
    assert all(tool.input_schema["additionalProperties"] is False for tool in tools)
    assert [tool.safety_class for tool in tools] == [
        "read_only",
        "compute",
        "read_only",
        "read_only",
        "read_only",
        "playback",
        "playback",
        "playback",
        "playback",
        "read_only",
    ]


def test_configured_schema_maxima_are_enforced() -> None:
    limits = Tier2Limits(search_results=7, document_read_chars=4321)
    registry, _, _, _ = make_registry(limits)

    calculate = registry.get("calculate").input_schema
    document_search = registry.get("search_documents").input_schema
    document_read = registry.get("read_document").input_schema

    assert calculate["properties"]["expression"]["maxLength"] == 256
    assert document_search["properties"]["max_results"]["maximum"] == 7
    assert document_read["properties"]["start_char"]["minimum"] == 0
    assert document_read["properties"]["max_chars"]["maximum"] == 4321


def test_adapters_delegate_and_propagate_turn_number() -> None:
    registry, documents, audio, playback = make_registry()
    context = ToolContext(turn_number=9)

    registry.execute("search_documents", {"query": "notes"}, context)
    registry.execute("read_document", {"folder_id": "docs", "relative_path": "a.txt"}, context)
    registry.execute("search_audio", {"query": "song"}, context)
    registry.execute("play_audio", {"audio_id": "opaque"}, context)
    for name in ("pause_audio", "resume_audio", "stop_audio", "now_playing"):
        registry.execute(name, {}, context)

    assert documents.calls == [
        ("search", "notes", None, None),
        ("read", "docs", "a.txt", 0, None),
    ]
    assert audio.calls == [("song", None, None, 9)]
    assert playback.calls == [
        ("play", "opaque", 9),
        ("pause",),
        ("resume",),
        ("stop",),
        ("now_playing",),
    ]


def test_folder_filter_array_rejects_non_string_items_before_delegation() -> None:
    registry, documents, _, _ = make_registry()

    with pytest.raises(InvalidArgumentsError):
        registry.execute(
            "search_documents",
            {"query": "notes", "folder_ids": ["docs", 7]},
            ToolContext(turn_number=1),
        )

    assert documents.calls == []


def test_capability_prompt_states_security_and_scope_boundaries() -> None:
    registry, _, _, _ = make_registry()
    prompt = build_system_prompt(registry).casefold()

    for phrase in (
        "tools can fail",
        "do not invent",
        "untrusted data",
        "folder id",
        "relative path",
        "later user turn",
        "cannot mutate files",
        "cannot browse",
        "cannot transcribe",
        "cannot create playlists",
        "cannot work in the background",
    ):
        assert phrase in prompt
