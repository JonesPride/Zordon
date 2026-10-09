from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath

from zordon.config import ApprovedRoot

_SKIPPED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}


class ApprovedFolderError(RuntimeError):
    """A safe local-path failure with a stable agent-facing code."""

    def __init__(self, code: str, summary: str) -> None:
        super().__init__(summary)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResolvedFile:
    folder_id: str
    relative_path: str
    path: Path = field(repr=False)
    size: int
    mtime_ns: int


class ApprovedFolders:
    """Resolve and scan files without exposing or escaping configured roots."""

    def __init__(self, roots: Iterable[ApprovedRoot]) -> None:
        self._roots: dict[str, Path] = {}
        seen_paths: set[str] = set()
        for root in roots:
            folder_id = root.folder_id.casefold()
            canonical_path = root.path.resolve(strict=True)
            normalized_path = os.path.normcase(str(canonical_path))
            if folder_id in self._roots:
                raise ValueError(f"Duplicate approved folder ID '{folder_id}'.")
            if normalized_path in seen_paths:
                raise ValueError("Approved folder roots must be unique.")
            self._roots[folder_id] = canonical_path
            seen_paths.add(normalized_path)

    @property
    def folder_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._roots, key=str.casefold))

    def resolve_file(self, folder_id: str, relative_path: str) -> ResolvedFile:
        normalized_id = folder_id.strip().casefold()
        try:
            root = self._roots[normalized_id]
        except KeyError as exc:
            raise ApprovedFolderError(
                "folder_not_approved", f"Folder '{normalized_id}' is not approved."
            ) from exc

        normalized_relative = _validate_relative_path(relative_path)
        try:
            candidate = (root / normalized_relative).resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ApprovedFolderError(
                "file_not_found",
                f"File '{normalized_relative}' was not found in folder '{normalized_id}'.",
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise ApprovedFolderError(
                "path_outside_approved_folder", "The requested path is not safe."
            ) from exc

        if not _is_within(root, candidate):
            raise ApprovedFolderError(
                "path_outside_approved_folder",
                f"File '{normalized_relative}' is outside folder '{normalized_id}'.",
            )
        if not candidate.is_file():
            raise ApprovedFolderError(
                "file_not_found",
                f"File '{normalized_relative}' was not found in folder '{normalized_id}'.",
            )
        stat = candidate.stat()
        return ResolvedFile(
            folder_id=normalized_id,
            relative_path=normalized_relative,
            path=candidate,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

    def iter_files(
        self,
        folder_ids: Iterable[str],
        suffixes: Iterable[str],
        limit: int,
    ) -> tuple[ResolvedFile, ...]:
        if limit <= 0:
            raise ValueError("File scan limit must be positive.")
        selected: list[tuple[str, Path]] = []
        for requested_id in folder_ids:
            folder_id = requested_id.strip().casefold()
            try:
                selected.append((folder_id, self._roots[folder_id]))
            except KeyError as exc:
                raise ApprovedFolderError(
                    "folder_not_approved", f"Folder '{folder_id}' is not approved."
                ) from exc
        selected.sort(key=lambda item: item[0].casefold())
        allowed_suffixes = {suffix.casefold() for suffix in suffixes}
        found: list[ResolvedFile] = []

        for folder_id, root in selected:
            for current, directory_names, file_names in os.walk(root, followlinks=False):
                current_path = Path(current)
                directory_names[:] = sorted(
                    (
                        name
                        for name in directory_names
                        if not _skip_directory(current_path / name, name)
                    ),
                    key=str.casefold,
                )
                for file_name in sorted(file_names, key=str.casefold):
                    if Path(file_name).suffix.casefold() not in allowed_suffixes:
                        continue
                    relative = (current_path / file_name).relative_to(root).as_posix()
                    try:
                        resolved = self.resolve_file(folder_id, relative)
                    except ApprovedFolderError as exc:
                        if exc.code in {"file_not_found", "path_outside_approved_folder"}:
                            continue
                        raise
                    found.append(resolved)
                    if len(found) >= limit:
                        return tuple(found)
        return tuple(found)


def _validate_relative_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ApprovedFolderError(
            "path_outside_approved_folder", "The requested path must be relative."
        )
    value = relative_path.strip()
    native = PurePath(value)
    windows = PureWindowsPath(value)
    if (
        native.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part == ".." for part in native.parts)
        or any(part == ".." for part in windows.parts)
    ):
        raise ApprovedFolderError(
            "path_outside_approved_folder", "The requested path must be relative."
        )
    return value


def _is_within(root: Path, candidate: Path) -> bool:
    normalized_root = os.path.normcase(str(root))
    normalized_candidate = os.path.normcase(str(candidate))
    try:
        return os.path.commonpath((normalized_root, normalized_candidate)) == normalized_root
    except ValueError:
        return False


def _skip_directory(path: Path, name: str) -> bool:
    if name.startswith(".") or name.casefold() in _SKIPPED_DIRECTORIES:
        return True
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return True
