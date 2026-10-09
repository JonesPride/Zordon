from __future__ import annotations

import filecmp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT, is_temp_state_root, state_root
from .drafts import drafts_dir
from .images import images_dir
from .logs import logs_dir

STATE_DIRS = ("data", "drafts", "images", "logs")


@dataclass(frozen=True)
class MigrationResult:
    copied: list[Path]
    skipped: list[Path]
    conflicts: list[Path]

    def as_text(self) -> str:
        return "\n".join(
            [
                f"copied: {len(self.copied)}",
                f"skipped: {len(self.skipped)}",
                f"conflicts: {len(self.conflicts)}",
            ]
        )


def state_text(app_root: Path = ROOT) -> str:
    root = state_root() if app_root == ROOT else app_root
    location = "temporary" if is_temp_state_root(root) else "durable"
    return "\n".join(
        [
            "Zordon state",
            f"- state root: {root}",
            f"- location: {location}",
            f"- memory file: {root / 'data' / 'memory.jsonl'}",
            f"- drafts folder: {drafts_dir(app_root)}",
            f"- images folder: {images_dir(app_root)}",
            f"- logs folder: {logs_dir(app_root)}",
        ]
    )


def open_state_folder(app_root: Path = ROOT) -> Path:
    root = state_root() if app_root == ROOT else app_root
    return open_path(root, create_directory=True)


def open_path(path: Path, create_directory: bool = False) -> Path:
    if create_directory:
        path.mkdir(parents=True, exist_ok=True)
    os.startfile(str(path))
    return path


def migrate_state(source: Path, target: Path) -> MigrationResult:
    copied: list[Path] = []
    skipped: list[Path] = []
    conflicts: list[Path] = []
    if not source.exists():
        return MigrationResult(copied=copied, skipped=skipped, conflicts=conflicts)

    target.mkdir(parents=True, exist_ok=True)
    for dirname in STATE_DIRS:
        source_dir = source / dirname
        if not source_dir.exists():
            continue
        for source_path in source_dir.rglob("*"):
            relative = source_path.relative_to(source)
            target_path = target / relative
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
                copied.append(relative)
                continue
            if filecmp.cmp(source_path, target_path, shallow=False):
                skipped.append(relative)
                continue
            conflicts.append(relative)
    return MigrationResult(copied=copied, skipped=skipped, conflicts=conflicts)
