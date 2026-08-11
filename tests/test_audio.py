from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from zordon.audio import AudioCatalog, AudioEntry
from zordon.config import ApprovedRoot, Tier2Limits
from zordon.paths import ApprovedFolders
from zordon.tools.base import ToolResult


class MetadataLoader:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> Any:
        self.calls.append(path)
        value = self.metadata[path.name]
        if isinstance(value, BaseException):
            raise value
        return value


class IdFactory:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"opaque-{self.count}"


def audio(tags: Any = None, duration: Any = 123.5) -> Any:
    return SimpleNamespace(
        tags={} if tags is None else tags,
        info=SimpleNamespace(length=duration),
    )


@pytest.fixture
def make_catalog(tmp_path: Path) -> Callable[..., tuple[AudioCatalog, Path, MetadataLoader]]:
    def make(
        files: dict[str, Any],
        *,
        audio_scan: int = 20_000,
        search_results: int = 10,
        id_factory: Callable[[], str] | None = None,
    ) -> tuple[AudioCatalog, Path, MetadataLoader]:
        root = tmp_path / "music"
        root.mkdir(exist_ok=True)
        for relative_path in files:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"audio")
        loader = MetadataLoader({Path(name).name: value for name, value in files.items()})
        folders = ApprovedFolders((ApprovedRoot("music", root.resolve()),))
        limits = Tier2Limits(audio_scan=audio_scan, search_results=search_results)
        catalog = AudioCatalog(
            folders,
            limits,
            metadata_loader=loader,
            id_factory=id_factory or IdFactory(),
        )
        return catalog, root, loader

    return make


@pytest.mark.parametrize("suffix", [".mp3", ".m4a", ".wav", ".flac", ".ogg"])
def test_search_supports_each_audio_format(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]], suffix: str
) -> None:
    catalog, _, _ = make_catalog(
        {f"Artist/Track{suffix.upper()}": audio({"title": ["The Song"]})}
    )

    result = catalog.search("the song", turn_number=1)

    assert result.ok is True
    assert result.data["results"][0]["format"] == suffix.removeprefix(".")
    assert result.data["results"][0]["relative_path"] == f"Artist/Track{suffix.upper()}"


@pytest.mark.parametrize(
    ("query", "field"),
    [
        ("anthem", "title"),
        ("quinton", "artist"),
        ("night drive", "album"),
        ("hip-hop", "genre"),
        ("07", "track"),
        ("2026", "year"),
        ("hidden gem", "filename"),
        ("mixtapes", "relative_path"),
    ],
)
def test_search_matches_every_supported_metadata_field(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
    query: str,
    field: str,
) -> None:
    catalog, _, _ = make_catalog(
        {
            "Mixtapes/Hidden Gem.mp3": audio(
                {
                    "title": ["Anthem"],
                    "artist": "Quinton",
                    "album": ["Night", "Drive"],
                    "genre": ["Hip-Hop"],
                    "tracknumber": ["07/12"],
                    "date": ["2026"],
                },
                62.25,
            )
        }
    )

    result = catalog.search(query, turn_number=2)

    item = result.data["results"][0]
    assert item[field]
    assert item["duration_seconds"] == 62.25
    assert item["folder_id"] == "music"
    assert item["untrusted_data"] is True


def test_search_normalizes_scalar_list_and_missing_tags(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "Fallback Song.mp3": audio(
                {
                    "title": [],
                    "artist": "Solo Artist",
                    "album": ["First", "Second"],
                    "genre": None,
                },
                None,
            )
        }
    )

    result = catalog.search("fallback", turn_number=1)

    item = result.data["results"][0]
    assert item["title"] == "Fallback Song"
    assert item["artist"] == "Solo Artist"
    assert item["album"] == "First; Second"
    assert item["genre"] == ""
    assert item["duration_seconds"] is None


def test_search_caps_metadata_fields_at_two_thousand_characters(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog({"Long.mp3": audio({"title": "x" * 2_100})})

    result = catalog.search("xxx", turn_number=1)

    assert len(result.data["results"][0]["title"]) == 2_000


def test_search_skips_corrupt_or_unsupported_metadata_without_leaking_errors(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "Good.mp3": audio({"title": "Good"}),
            "Bad.flac": RuntimeError("private parser failure"),
        }
    )

    result = catalog.search("good", turn_number=1)

    assert [item["title"] for item in result.data["results"]] == ["Good"]
    assert result.data["skipped_files"] == 1
    assert "private" not in result.summary
    assert "private" not in str(result.data)


def test_search_returns_deterministic_ranking_and_respects_result_limit(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "z.mp3": audio({"title": "Target"}),
            "a-target.mp3": audio({"title": "Other"}),
            "b.mp3": audio({"artist": "Target"}),
        },
        search_results=2,
    )

    result = catalog.search("target", max_results=2, turn_number=1)

    assert [item["relative_path"] for item in result.data["results"]] == ["z.mp3", "b.mp3"]


def test_search_marks_scan_limit_as_partial(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    files = {f"{index:03}.mp3": audio({"title": "match"}) for index in range(101)}
    catalog, _, loader = make_catalog(files, audio_scan=100)

    result = catalog.search("match", turn_number=1)

    assert result.data["partial"] is True
    assert len(loader.calls) == 100


def test_search_validates_request_and_approved_folders(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
    tmp_path: Path,
) -> None:
    catalog, _, _ = make_catalog({"Track.mp3": audio()})

    assert catalog.search(" ", turn_number=1).code == "audio_search_invalid"
    assert catalog.search("x", max_results=0, turn_number=1).code == "audio_search_invalid"
    assert catalog.search("x", folder_ids=["other"], turn_number=1).code == "folder_not_approved"
    empty = AudioCatalog(ApprovedFolders(()), Tier2Limits())
    assert empty.search("x", turn_number=1).code == "approved_folders_missing"


def test_search_reuses_cache_and_invalidates_it_after_file_change(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, root, loader = make_catalog({"Track.mp3": audio({"title": "Track"})})

    catalog.search("track", turn_number=1)
    catalog.search("track", turn_number=2)
    assert len(loader.calls) == 1

    track = root / "Track.mp3"
    track.write_bytes(b"new audio bytes")
    os.utime(track, ns=(track.stat().st_atime_ns, track.stat().st_mtime_ns + 1))
    catalog.search("track", turn_number=3)
    assert len(loader.calls) == 2


def test_audio_ids_are_opaque_and_not_paths(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, root, _ = make_catalog({"private/Track.mp3": audio({"title": "Track"})})

    result = catalog.search("track", turn_number=1)

    audio_id = result.data["results"][0]["audio_id"]
    assert audio_id == "opaque-1"
    assert str(root) not in audio_id
    assert str(root) not in str(result.data)


def test_single_result_is_immediately_playable(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, root, _ = make_catalog({"Track.mp3": audio({"title": "Track"})})
    result = catalog.search("track", turn_number=4)

    resolved = catalog.resolve_for_playback(result.data["results"][0]["audio_id"], 4)

    assert isinstance(resolved, AudioEntry)
    assert resolved.relative_path == "Track.mp3"
    assert str(root) not in repr(resolved)


def test_ambiguous_results_require_a_later_turn(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "Lose Yourself.mp3": audio({"title": "Lose Yourself"}),
            "Lose Yourself Live.mp3": audio({"title": "Lose Yourself Live"}),
        }
    )
    result = catalog.search("lose yourself", turn_number=4)
    first_id = result.data["results"][0]["audio_id"]

    denied = catalog.resolve_for_playback(first_id, turn_number=4)

    assert isinstance(denied, ToolResult)
    assert denied.code == "audio_selection_required"
    assert isinstance(catalog.resolve_for_playback(first_id, turn_number=5), AudioEntry)


def test_result_limit_cannot_bypass_ambiguous_selection_delay(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "Track One.mp3": audio({"title": "Track One"}),
            "Track Two.mp3": audio({"title": "Track Two"}),
        }
    )
    result = catalog.search("track", max_results=1, turn_number=4)
    audio_id = result.data["results"][0]["audio_id"]

    denied = catalog.resolve_for_playback(audio_id, turn_number=4)

    assert isinstance(denied, ToolResult)
    assert denied.code == "audio_selection_required"


def test_new_ambiguous_search_replaces_previous_pending_ids(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "One Mix.mp3": audio({"title": "One Mix"}),
            "One Live.mp3": audio({"title": "One Live"}),
            "Two Mix.mp3": audio({"title": "Two Mix"}),
            "Two Live.mp3": audio({"title": "Two Live"}),
        }
    )
    old = catalog.search("one", turn_number=1).data["results"][0]["audio_id"]

    catalog.search("two", turn_number=2)

    invalid = catalog.resolve_for_playback(old, turn_number=3)
    assert isinstance(invalid, ToolResult)
    assert invalid.code == "audio_id_invalid"


def test_no_match_search_replaces_previous_pending_ids(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]],
) -> None:
    catalog, _, _ = make_catalog(
        {
            "One Mix.mp3": audio({"title": "One Mix"}),
            "One Live.mp3": audio({"title": "One Live"}),
        }
    )
    old = catalog.search("one", turn_number=1).data["results"][0]["audio_id"]

    assert catalog.search("missing", turn_number=2).data["results"] == []

    invalid = catalog.resolve_for_playback(old, turn_number=3)
    assert isinstance(invalid, ToolResult)
    assert invalid.code == "audio_id_invalid"


@pytest.mark.parametrize("change", ["modified", "deleted"])
def test_changed_or_deleted_files_invalidate_audio_ids(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]], change: str
) -> None:
    catalog, root, _ = make_catalog({"Track.mp3": audio({"title": "Track"})})
    audio_id = catalog.search("track", turn_number=1).data["results"][0]["audio_id"]
    track = root / "Track.mp3"
    if change == "deleted":
        track.unlink()
    else:
        track.write_bytes(b"changed")

    invalid = catalog.resolve_for_playback(audio_id, turn_number=2)

    assert isinstance(invalid, ToolResult)
    assert invalid.code == "audio_id_invalid"
    assert str(root) not in invalid.summary


def test_file_moved_outside_boundary_invalidates_audio_id(
    make_catalog: Callable[..., tuple[AudioCatalog, Path, MetadataLoader]], tmp_path: Path
) -> None:
    catalog, root, _ = make_catalog({"Track.mp3": audio({"title": "Track"})})
    audio_id = catalog.search("track", turn_number=1).data["results"][0]["audio_id"]
    outside = tmp_path / "outside.mp3"
    (root / "Track.mp3").replace(outside)
    try:
        (root / "Track.mp3").symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")

    invalid = catalog.resolve_for_playback(audio_id, turn_number=2)

    assert isinstance(invalid, ToolResult)
    assert invalid.code == "audio_id_invalid"
    assert str(outside) not in invalid.summary
