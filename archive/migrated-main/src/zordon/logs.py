from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .config import ROOT, state_root


def logs_dir(root: Path) -> Path:
    base = state_root() if root == ROOT else root
    return base / "logs"


def list_logs(root: Path) -> list[str]:
    directory = logs_dir(root)
    if not directory.exists():
        return []
    return sorted(
        path.name for path in directory.iterdir() if path.is_file() and path.suffix == ".jsonl"
    )


def read_log_tail(root: Path, filename: str, line_count: int = 20) -> str:
    filename = safe_log_filename(filename)
    if line_count <= 0:
        raise ValueError("line_count must be positive")

    directory = logs_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No log found for {filename}.")

    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-line_count:]
    return "\n".join(tail) if tail else "(Log is empty.)"


def summarize_events(root: Path, filename: str | None = None) -> str:
    filename = filename or latest_log(root)
    if filename is None:
        return "No audit logs saved yet."
    filename = safe_log_filename(filename)
    directory = logs_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No log found for {filename}.")

    counts: Counter[str] = Counter()
    invalid_lines = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        name = event.get("event")
        if isinstance(name, str) and name.strip():
            counts[name] += 1
        else:
            invalid_lines += 1

    lines = [f"Events in {filename}:"]
    if counts:
        lines.extend(f"- {name}: {count}" for name, count in sorted(counts.items()))
    else:
        lines.append("- no events found")
    if invalid_lines:
        lines.append(f"- invalid_lines: {invalid_lines}")
    return "\n".join(lines)


def summarize_last_turn(root: Path, filename: str | None = None) -> str:
    filename = filename or latest_log(root)
    if filename is None:
        return "No audit logs saved yet."
    filename = safe_log_filename(filename)
    directory = logs_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No log found for {filename}.")

    last_user = ""
    last_assistant = ""
    for event in _iter_json_events(path):
        name = event.get("event")
        if name == "user_turn":
            last_user = _event_text(event)
        elif name == "assistant_reply":
            last_assistant = _event_text(event)

    lines = [f"Last turn in {filename}:"]
    lines.append(f"User: {last_user or '(none found)'}")
    lines.append(f"Assistant: {last_assistant or '(none found)'}")
    return "\n".join(lines)


def search_log(root: Path, term: str, filename: str | None = None, limit: int = 10) -> str:
    term = _required_text(term, "term")
    if limit <= 0:
        raise ValueError("limit must be positive")
    filename = filename or latest_log(root)
    if filename is None:
        return "No audit logs saved yet."
    filename = safe_log_filename(filename)
    directory = logs_dir(root)
    path = directory / filename
    _ensure_inside(directory, path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No log found for {filename}.")

    matches: list[str] = []
    lowered = term.lower()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if lowered not in line.lower():
            continue
        matches.append(_format_match(line_number, line))
        if len(matches) >= limit:
            break

    if not matches:
        return f"No matches for {term!r} in {filename}."
    lines = [f"Matches for {term!r} in {filename}:"]
    lines.extend(matches)
    return "\n".join(lines)


def latest_log(root: Path) -> str | None:
    logs = list_logs(root)
    return logs[-1] if logs else None


def safe_log_filename(filename: str) -> str:
    filename = _required_text(filename, "filename")
    if "/" in filename or "\\" in filename:
        raise ValueError("filename cannot include path separators")
    if filename in {".", ".."}:
        raise ValueError("filename is not valid")
    if not filename.endswith(".jsonl"):
        raise ValueError("filename must end in .jsonl")
    if not re.match(r"^[A-Za-z0-9._ -]+$", filename):
        raise ValueError(
            "filename can only contain letters, numbers, spaces, dots, underscores, and hyphens"
        )
    return filename


def _iter_json_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _event_text(event: dict) -> str:
    value = event.get("text") or event.get("output") or event.get("error") or ""
    return str(value).strip()


def _format_match(line_number: int, line: str) -> str:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return f"- line {line_number}: {_clip(line)}"
    if not isinstance(event, dict):
        return f"- line {line_number}: {_clip(line)}"

    name = event.get("event", "unknown")
    text = _event_text(event) or json.dumps(event, ensure_ascii=True)
    return f"- line {line_number} {name}: {_clip(text)}"


def _clip(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _ensure_inside(directory: Path, path: Path) -> None:
    resolved_directory = directory.resolve()
    resolved_path = path.resolve()
    if resolved_directory not in resolved_path.parents:
        raise ValueError("filename must stay inside the logs folder")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
