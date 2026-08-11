from __future__ import annotations

import math
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mutagen

from zordon.config import Tier2Limits
from zordon.paths import ApprovedFolderError, ApprovedFolders, ResolvedFile
from zordon.tools.base import ToolResult

_SUPPORTED_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
_METADATA_LIMIT = 2_000

MetadataLoader = Callable[[Path], Any]
IdFactory = Callable[[], str]


@dataclass(frozen=True, slots=True)
class _AudioMetadata:
    title: str
    artist: str
    album: str
    genre: str
    track: str
    year: str
    filename: str
    relative_folder: str
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class AudioEntry:
    """A process-local catalog entry whose canonical path remains private."""

    audio_id: str
    folder_id: str
    relative_path: str
    path: Path = field(repr=False)
    size: int
    mtime_ns: int
    metadata: Mapping[str, str | float | None]
    playable_after_turn: int


class AudioCatalog:
    """Search approved audio metadata and enforce ask-first selection."""

    def __init__(
        self,
        folders: ApprovedFolders,
        limits: Tier2Limits,
        *,
        metadata_loader: MetadataLoader | None = None,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._folders = folders
        self._limits = limits
        self._metadata_loader = metadata_loader or _load_metadata
        self._id_factory = id_factory or (lambda: secrets.token_urlsafe(18))
        self._cache: dict[tuple[Path, int, int], _AudioMetadata] = {}
        self._entries: dict[str, AudioEntry] = {}
        self._pending_ids: set[str] = set()

    def search(
        self,
        query: str,
        folder_ids: Iterable[str] | None = None,
        max_results: int | None = None,
        *,
        turn_number: int,
    ) -> ToolResult:
        normalized_query = query.strip().casefold() if isinstance(query, str) else ""
        result_limit = self._limits.search_results if max_results is None else max_results
        if (
            not normalized_query
            or isinstance(result_limit, bool)
            or not isinstance(result_limit, int)
            or not 1 <= result_limit <= min(self._limits.search_results, 25)
            or isinstance(turn_number, bool)
            or not isinstance(turn_number, int)
            or turn_number < 1
        ):
            return ToolResult.failure(
                "audio_search_invalid",
                "The audio search request is invalid.",
            )

        selected = tuple(folder_ids) if folder_ids is not None else ()
        if not selected:
            selected = self._folders.folder_ids
        if not selected:
            return ToolResult.failure(
                "approved_folders_missing",
                "No approved folders are configured for audio search.",
            )

        try:
            scanned = self._folders.iter_files(
                selected,
                _SUPPORTED_SUFFIXES,
                self._limits.audio_scan + 1,
            )
        except ApprovedFolderError as exc:
            return ToolResult.failure(exc.code, str(exc))

        partial = len(scanned) > self._limits.audio_scan
        ranked: list[tuple[float, str, str, ResolvedFile, _AudioMetadata]] = []
        skipped_files = 0
        tokens = tuple(dict.fromkeys(normalized_query.split()))
        for file in scanned[: self._limits.audio_scan]:
            try:
                metadata = self._metadata(file)
            except Exception:  # noqa: BLE001 - parser failures are per-file warnings
                skipped_files += 1
                continue
            if metadata is None:
                skipped_files += 1
                continue
            score = _score(metadata, normalized_query, tokens)
            if score <= 0:
                continue
            ranked.append(
                (
                    -score,
                    file.folder_id.casefold(),
                    file.relative_path.casefold(),
                    file,
                    metadata,
                )
            )
        ranked.sort(key=lambda item: item[:3])
        ambiguous = len(ranked) > 1
        matches = ranked[:result_limit]

        for old_id in self._pending_ids:
            self._entries.pop(old_id, None)
        self._pending_ids.clear()

        playable_after_turn = turn_number + 1 if ambiguous else turn_number
        results: list[dict[str, Any]] = []
        for _, _, _, file, metadata in matches:
            audio_id = self._new_id()
            safe_metadata = _metadata_dict(metadata)
            entry = AudioEntry(
                audio_id=audio_id,
                folder_id=file.folder_id,
                relative_path=file.relative_path,
                path=file.path,
                size=file.size,
                mtime_ns=file.mtime_ns,
                metadata=safe_metadata,
                playable_after_turn=playable_after_turn,
            )
            self._entries[audio_id] = entry
            if ambiguous:
                self._pending_ids.add(audio_id)
            results.append(
                {
                    "audio_id": audio_id,
                    "folder_id": file.folder_id,
                    "relative_path": file.relative_path,
                    "format": file.path.suffix.casefold().removeprefix("."),
                    **safe_metadata,
                    "untrusted_data": True,
                }
            )

        return ToolResult.success(
            "Audio search complete.",
            {
                "query": query.strip(),
                "results": results,
                "partial": partial,
                "skipped_files": skipped_files,
                "untrusted_data": True,
            },
        )

    def resolve_for_playback(
        self,
        audio_id: str,
        turn_number: int,
    ) -> AudioEntry | ToolResult:
        entry = self._entries.get(audio_id) if isinstance(audio_id, str) else None
        if entry is None:
            return _invalid_audio_id()
        if turn_number < entry.playable_after_turn:
            return ToolResult.failure(
                "audio_selection_required",
                "Choose one audio result on a later turn before playing it.",
            )
        try:
            current = self._folders.resolve_file(entry.folder_id, entry.relative_path)
        except ApprovedFolderError:
            self._invalidate(entry.audio_id)
            return _invalid_audio_id()
        if (
            current.path != entry.path
            or current.size != entry.size
            or current.mtime_ns != entry.mtime_ns
        ):
            self._invalidate(entry.audio_id)
            return _invalid_audio_id()
        return entry

    def _metadata(self, file: ResolvedFile) -> _AudioMetadata | None:
        key = (file.path, file.size, file.mtime_ns)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        loaded = self._metadata_loader(file.path)
        if loaded is None:
            return None
        metadata = _normalize_metadata(loaded, file)
        stale = [cache_key for cache_key in self._cache if cache_key[0] == file.path]
        for cache_key in stale:
            del self._cache[cache_key]
        self._cache[key] = metadata
        return metadata

    def _new_id(self) -> str:
        for _ in range(100):
            candidate = self._id_factory()
            if isinstance(candidate, str) and candidate and candidate not in self._entries:
                return candidate
        raise RuntimeError("Unable to allocate an audio ID.")

    def _invalidate(self, audio_id: str) -> None:
        self._entries.pop(audio_id, None)
        self._pending_ids.discard(audio_id)


def _load_metadata(path: Path) -> Any:
    return mutagen.File(path, easy=True)


def _normalize_metadata(audio: Any, file: ResolvedFile) -> _AudioMetadata:
    tags = audio.tags if isinstance(getattr(audio, "tags", None), Mapping) else {}
    filename = _cap(file.path.stem)
    relative_folder = _cap(Path(file.relative_path).parent.as_posix())
    if relative_folder == ".":
        relative_folder = ""
    title = _tag(tags, "title") or filename
    return _AudioMetadata(
        title=title,
        artist=_tag(tags, "artist"),
        album=_tag(tags, "album"),
        genre=_tag(tags, "genre"),
        track=_tag(tags, "tracknumber", "track"),
        year=_tag(tags, "date", "year"),
        filename=filename,
        relative_folder=relative_folder,
        duration_seconds=_duration(audio),
    )


def _tag(tags: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name not in tags:
            continue
        value = tags[name]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            normalized = "; ".join(str(item).strip() for item in value if str(item).strip())
        elif value is None:
            normalized = ""
        else:
            normalized = str(value).strip()
        if normalized:
            return _cap(normalized)
    return ""


def _duration(audio: Any) -> float | None:
    value = getattr(getattr(audio, "info", None), "length", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    duration = float(value)
    return duration if math.isfinite(duration) and duration >= 0 else None


def _cap(value: str) -> str:
    return value[:_METADATA_LIMIT]


def _metadata_dict(metadata: _AudioMetadata) -> dict[str, str | float | None]:
    return {
        "title": metadata.title,
        "artist": metadata.artist,
        "album": metadata.album,
        "genre": metadata.genre,
        "track": metadata.track,
        "year": metadata.year,
        "filename": metadata.filename,
        "relative_folder": metadata.relative_folder,
        "duration_seconds": metadata.duration_seconds,
    }


def _score(metadata: _AudioMetadata, query: str, tokens: tuple[str, ...]) -> float:
    fields = {
        "title": metadata.title.casefold(),
        "artist": metadata.artist.casefold(),
        "filename": metadata.filename.casefold(),
        "album": metadata.album.casefold(),
        "genre": metadata.genre.casefold(),
        "track": metadata.track.casefold(),
        "year": metadata.year.casefold(),
        "relative_folder": metadata.relative_folder.casefold(),
    }
    searchable = " ".join(fields.values())
    if not all(token in searchable for token in tokens):
        return 0
    score = 100 if query in searchable else 0
    for name, weight in (
        ("title", 80),
        ("artist", 70),
        ("filename", 60),
        ("album", 50),
        ("genre", 40),
    ):
        if query in fields[name]:
            score += weight
    covered = sum(any(token in value for value in fields.values()) for token in tokens)
    score += 30 * covered / len(tokens)
    return score


def _invalid_audio_id() -> ToolResult:
    return ToolResult.failure(
        "audio_id_invalid",
        "The audio selection is no longer valid. Search again.",
    )
