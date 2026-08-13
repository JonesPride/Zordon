from __future__ import annotations

import re
from pathlib import Path

from .config import ROOT, state_root


def drafts_dir(root: Path) -> Path:
    base = state_root() if root == ROOT else root
    return base / "drafts"


def save_draft(root: Path, filename: str, content: str) -> Path:
    filename = safe_draft_filename(filename)
    if not content.strip():
        raise ValueError("content must be a non-empty string")

    directory = drafts_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def list_drafts(root: Path) -> list[str]:
    directory = drafts_dir(root)
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".txt", ".md"}
    )


def latest_draft(root: Path) -> Path:
    directory = drafts_dir(root)
    if not directory.exists():
        raise FileNotFoundError("No drafts saved yet.")

    drafts = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".txt", ".md"}
    ]
    if not drafts:
        raise FileNotFoundError("No drafts saved yet.")
    return max(drafts, key=lambda path: path.stat().st_mtime)


def read_draft(root: Path, filename: str) -> str:
    filename = safe_draft_filename(filename)
    directory = drafts_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No draft found for {filename}.")
    return path.read_text(encoding="utf-8")


def read_latest_draft(root: Path) -> tuple[Path, str]:
    path = latest_draft(root)
    return path, path.read_text(encoding="utf-8")


def delete_draft(root: Path, filename: str) -> Path:
    filename = safe_draft_filename(filename)
    directory = drafts_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No draft found for {filename}.")
    path.unlink()
    return path


def safe_draft_filename(filename: str) -> str:
    filename = _required_text(filename, "filename")
    if "/" in filename or "\\" in filename:
        raise ValueError("filename cannot include path separators")
    if filename in {".", ".."}:
        raise ValueError("filename is not valid")
    if not filename.endswith((".txt", ".md")):
        raise ValueError("filename must end in .txt or .md")
    if not re.match(r"^[A-Za-z0-9._ -]+$", filename):
        raise ValueError("filename can only contain letters, numbers, spaces, dots, underscores, and hyphens")
    return filename


def _ensure_inside(directory: Path, path: Path) -> None:
    resolved_directory = directory.resolve()
    resolved_path = path.resolve()
    if resolved_directory not in resolved_path.parents:
        raise ValueError("filename must stay inside the drafts folder")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
