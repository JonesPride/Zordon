from __future__ import annotations

from pathlib import Path

import pytest

from zordon.config import ApprovedRoot
from zordon.paths import ApprovedFolderError, ApprovedFolders


@pytest.fixture
def approved(tmp_path: Path) -> tuple[ApprovedFolders, Path]:
    root = tmp_path / "root"
    root.mkdir()
    return ApprovedFolders((ApprovedRoot("notes", root.resolve()),)), root


def test_resolve_file_returns_safe_identity(
    approved: tuple[ApprovedFolders, Path],
) -> None:
    folders, root = approved
    file = root / "sub" / "Note.MD"
    file.parent.mkdir()
    file.write_text("hello")

    resolved = folders.resolve_file("NOTES", "sub/Note.MD")

    assert resolved.folder_id == "notes"
    assert resolved.relative_path == "sub/Note.MD"
    assert resolved.path == file.resolve()
    assert resolved.size == 5
    assert resolved.mtime_ns == file.stat().st_mtime_ns
    assert str(root) not in repr(resolved)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "   ",
        "../secret.txt",
        "sub/../secret.txt",
        "/etc/passwd",
        "C:\\secret.txt",
        "C:secret.txt",
        "\\\\server\\share\\file.txt",
    ],
)
def test_resolve_file_rejects_unsafe_path(
    approved: tuple[ApprovedFolders, Path], relative_path: str
) -> None:
    folders, _ = approved

    with pytest.raises(ApprovedFolderError) as error:
        folders.resolve_file("notes", relative_path)

    assert error.value.code == "path_outside_approved_folder"


def test_resolve_file_rejects_unknown_folder_and_missing_file(
    approved: tuple[ApprovedFolders, Path],
) -> None:
    folders, root = approved

    with pytest.raises(ApprovedFolderError) as unknown:
        folders.resolve_file("other", "x.txt")
    assert unknown.value.code == "folder_not_approved"

    with pytest.raises(ApprovedFolderError) as missing:
        folders.resolve_file("notes", "missing.txt")
    assert missing.value.code == "file_not_found"
    assert str(root) not in str(missing.value)


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")


def test_resolve_file_allows_in_root_symlink(
    approved: tuple[ApprovedFolders, Path],
) -> None:
    folders, root = approved
    target = root / "target.txt"
    target.write_text("safe")
    link = root / "link.txt"
    _symlink_or_skip(link, target)

    assert folders.resolve_file("notes", "link.txt").path == target.resolve()


def test_resolve_file_blocks_symlink_escape(
    approved: tuple[ApprovedFolders, Path], tmp_path: Path
) -> None:
    folders, root = approved
    target = tmp_path / "outside.txt"
    target.write_text("private")
    link = root / "link.txt"
    _symlink_or_skip(link, target)

    with pytest.raises(ApprovedFolderError) as error:
        folders.resolve_file("notes", "link.txt")
    assert error.value.code == "path_outside_approved_folder"


def test_iter_files_is_sorted_filtered_and_prunes_directories(
    approved: tuple[ApprovedFolders, Path],
) -> None:
    folders, root = approved
    (root / "b.TXT").write_text("b")
    (root / "A.md").write_text("a")
    (root / "ignore.exe").write_text("x")
    for name in (".hidden", ".git", ".venv", "venv", "node_modules", "__pycache__"):
        directory = root / name
        directory.mkdir()
        (directory / "hidden.txt").write_text("hidden")

    found = folders.iter_files(["NOTES"], {".txt", ".md"}, limit=10)

    assert [item.relative_path for item in found] == ["A.md", "b.TXT"]


def test_iter_files_does_not_traverse_directory_links(
    approved: tuple[ApprovedFolders, Path], tmp_path: Path
) -> None:
    folders, root = approved
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    _symlink_or_skip(root / "linked", outside, directory=True)

    assert folders.iter_files(["notes"], {".txt"}, limit=10) == ()


def test_iter_files_enforces_limit_and_validates_folders(
    approved: tuple[ApprovedFolders, Path],
) -> None:
    folders, root = approved
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")

    assert len(folders.iter_files(["notes"], {".txt"}, limit=1)) == 1
    with pytest.raises(ApprovedFolderError) as unknown:
        folders.iter_files(["other"], {".txt"}, limit=1)
    assert unknown.value.code == "folder_not_approved"
    with pytest.raises(ValueError, match="positive"):
        folders.iter_files(["notes"], {".txt"}, limit=0)
