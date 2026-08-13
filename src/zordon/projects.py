from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil


@dataclass(frozen=True)
class Project:
    slug: str
    name: str
    path: Path
    created_at: str


def projects_dir(root: Path) -> Path:
    return root / "projects"


def project_drafts_dir(project: Project) -> Path:
    return project.path / "drafts"


def project_images_dir(project: Project) -> Path:
    return project.path / "images"


def project_notes_dir(project: Project) -> Path:
    return project.path / "notes"


def project_brief_path(project: Project) -> Path:
    return project.path / "brief.md"


def content_plan_path(project: Project) -> Path:
    return project_notes_dir(project) / "content-plan.md"


def active_project_file(root: Path) -> Path:
    return root / "data" / "active_project.txt"


def create_project(root: Path, name: str) -> Project:
    name = _required_text(name, "project name")
    slug = project_slug(name)
    directory = projects_dir(root) / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "drafts").mkdir(exist_ok=True)
    (directory / "images").mkdir(exist_ok=True)
    (directory / "notes").mkdir(exist_ok=True)
    metadata_path = directory / "project.json"
    if metadata_path.exists():
        project = read_project(root, slug)
    else:
        project = Project(
            slug=slug,
            name=name,
            path=directory,
            created_at=datetime.now(UTC).isoformat(),
        )
        metadata_path.write_text(json.dumps(_project_to_dict(project), indent=2), encoding="utf-8")
    set_active_project(root, slug)
    return project


def read_project(root: Path, slug: str) -> Project:
    slug = project_slug(slug)
    directory = projects_dir(root) / slug
    if not directory.exists():
        directory = _find_project_directory(root, slug)
        slug = directory.name
    metadata_path = directory / "project.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No project found for {slug}.")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    name = _required_text(data.get("name"), "project name")
    created_at = _required_text(data.get("created_at"), "created_at")
    return Project(slug=slug, name=name, path=directory, created_at=created_at)


def list_projects(root: Path) -> list[Project]:
    directory = projects_dir(root)
    if not directory.exists():
        return []
    projects: list[Project] = []
    for path in directory.iterdir():
        if not path.is_dir():
            continue
        try:
            projects.append(read_project(root, path.name))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
    return sorted(projects, key=lambda project: project.name.lower())


def set_active_project(root: Path, slug: str) -> Project:
    project = read_project(root, slug)
    path = active_project_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(project.slug, encoding="utf-8")
    return project


def active_project(root: Path) -> Project | None:
    path = active_project_file(root)
    if not path.exists():
        return None
    slug = path.read_text(encoding="utf-8").strip()
    if not slug:
        return None
    try:
        return read_project(root, slug)
    except FileNotFoundError:
        return None


def project_slug(name: str) -> str:
    name = _required_text(name, "project name").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    if not slug:
        raise ValueError("project name must include letters or numbers")
    return slug[:64]


def _find_project_directory(root: Path, slug: str) -> Path:
    directory = projects_dir(root)
    candidates = [slug]
    if slug.startswith("the-"):
        candidates.append(slug.removeprefix("the-"))
    else:
        candidates.append(f"the-{slug}")
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path
    return directory / slug


def format_projects(root: Path) -> str:
    projects = list_projects(root)
    if not projects:
        return "No projects saved yet."
    active = active_project(root)
    active_slug = active.slug if active else ""
    lines = ["Zordon projects"]
    for project in projects:
        marker = " *" if project.slug == active_slug else ""
        lines.append(f"- {project.slug}: {project.name}{marker}")
    return "\n".join(lines)


def format_active_project(root: Path) -> str:
    project = active_project(root)
    if project is None:
        return "No active project."
    return "\n".join(
        [
            "Zordon project",
            f"- name: {project.name}",
            f"- slug: {project.slug}",
            f"- folder: {project.path}",
        ]
    )


def read_project_brief(root: Path) -> tuple[Project, str]:
    project = require_active_project(root)
    path = project_brief_path(project)
    if not path.exists():
        return project, "No project brief saved yet."
    content = path.read_text(encoding="utf-8").strip()
    return project, content if content else "No project brief saved yet."


def write_project_brief(root: Path, content: str) -> Path:
    project = require_active_project(root)
    content = _required_text(content, "project brief")
    path = project_brief_path(project)
    path.write_text(content + "\n", encoding="utf-8")
    return path


def write_content_plan(root: Path, content: str) -> Path:
    project = require_active_project(root)
    content = _required_text(content, "content plan")
    path = content_plan_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def read_content_plan(root: Path) -> tuple[Project, str]:
    project = require_active_project(root)
    path = content_plan_path(project)
    if not path.exists():
        return project, "No content plan saved yet."
    content = path.read_text(encoding="utf-8").strip()
    return project, content if content else "No content plan saved yet."


def list_project_drafts(root: Path) -> list[str]:
    project = require_active_project(root)
    directory = project_drafts_dir(project)
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".txt", ".md"}
    )


def list_project_images(root: Path) -> list[str]:
    project = require_active_project(root)
    directory = project_images_dir(project)
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def latest_project_draft(root: Path) -> Path:
    project = require_active_project(root)
    directory = project_drafts_dir(project)
    if not directory.exists():
        raise FileNotFoundError("No project drafts saved yet.")
    drafts = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".txt", ".md"}
    ]
    if not drafts:
        raise FileNotFoundError("No project drafts saved yet.")
    return max(drafts, key=lambda path: path.stat().st_mtime)


def latest_project_image(root: Path) -> Path:
    project = require_active_project(root)
    directory = project_images_dir(project)
    if not directory.exists():
        raise FileNotFoundError("No project images saved yet.")
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not images:
        raise FileNotFoundError("No project images saved yet.")
    return max(images, key=lambda path: path.stat().st_mtime)


def copy_draft_to_active_project(root: Path, source: Path) -> Path:
    project = require_active_project(root)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"No draft found at {source}.")
    target_dir = project_drafts_dir(project)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists():
        target = _numbered_copy_path(target)
    shutil.copy2(source, target)
    return target


def copy_image_to_active_project(root: Path, source: Path) -> Path:
    project = require_active_project(root)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"No image found at {source}.")
    target_dir = project_images_dir(project)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists():
        target = _numbered_copy_path(target)
    shutil.copy2(source, target)
    return target


def require_active_project(root: Path) -> Project:
    project = active_project(root)
    if project is None:
        raise FileNotFoundError("No active project.")
    return project


def _numbered_copy_path(path: Path) -> Path:
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _project_to_dict(project: Project) -> dict[str, str]:
    return {
        "slug": project.slug,
        "name": project.name,
        "created_at": project.created_at,
    }


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
