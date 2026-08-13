from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from .config import ROOT
from .drafts import safe_draft_filename, save_draft as write_draft
from .images import generate_image as create_image_file
from .memory import MemoryStore
from .projects import active_project, project_drafts_dir, project_images_dir


ToolHandler = Callable[[dict], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: ToolHandler
    requires_confirmation: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    output: str

    def as_model_text(self) -> str:
        status = "ok" if self.ok else "error"
        return f"{self.name} ({status}): {self.output}"


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict, confirmed: bool = False) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, output=f"No tool named {name!r} is registered.")
        if tool.requires_confirmation and not confirmed:
            return ToolResult(
                name=name,
                ok=False,
                output="This tool requires confirmation before it can run.",
            )

        try:
            return ToolResult(name=name, ok=True, output=tool.handler(arguments))
        except Exception as exc:
            return ToolResult(name=name, ok=False, output=str(exc))


def default_registry(memory: MemoryStore | None = None, root: Path = ROOT, config=None) -> ToolRegistry:
    tools = [
        Tool(
            name="refine_writing",
            description=(
                "Use this to improve a draft for clarity, flow, tone, or punch while preserving the user's intent."
            ),
            parameters=_object_schema(
                {
                    "draft": "The text to refine.",
                    "goal": "The desired result, audience, tone, or format.",
                }
            ),
            handler=refine_writing,
        ),
        Tool(
            name="generate_image",
            description=(
                "Use this when the user asks to generate, create, make, or produce an actual image file. "
                "This costs API credits and therefore requires confirmation."
            ),
            parameters=_object_schema(
                {
                    "prompt": "The final detailed image prompt to send to the image model.",
                    "filename": "Optional short PNG filename. Use an empty string to auto-name the file.",
                }
            ),
            handler=lambda args: generate_image(config, root, args),
            requires_confirmation=True,
        ),
        Tool(
            name="create_image_prompt",
            description=(
                "Use this only to draft prompt text when the user explicitly asks for an image prompt, concept, or wording."
            ),
            parameters=_object_schema(
                {
                    "idea": "The image idea or scene the user wants.",
                    "format": "The intended format, aspect ratio, style, or platform.",
                }
            ),
            handler=create_image_prompt,
        ),
        Tool(
            name="fetch_web_page",
            description=(
                "Use this to read a public web page when the user gives a URL and wants browsing or source-aware help."
            ),
            parameters=_object_schema(
                {
                    "url": "The full public URL to fetch, starting with http:// or https://.",
                    "question": "What to look for or summarize from the page.",
                }
            ),
            handler=fetch_web_page,
        ),
        Tool(
            name="save_draft",
            description=(
                "Save user-approved generated text to a local draft file. Use this only when the user asks to save a draft."
            ),
            parameters=_object_schema(
                {
                    "filename": "A short filename ending in .txt or .md. Path separators are not allowed.",
                    "content": "The draft text to save.",
                }
            ),
            handler=lambda args: save_draft(root, args),
            requires_confirmation=True,
        ),
    ]
    if memory is not None:
        tools.extend(memory_tools(memory))
    return ToolRegistry(tools)


def memory_tools(memory: MemoryStore) -> list[Tool]:
    return [
        Tool(
            name="remember_preference",
            description=(
                "Save a durable memory only when the user explicitly asks Zordon to remember a stable preference, "
                "personal fact, project detail, or recurring instruction."
            ),
            parameters=_object_schema(
                {
                    "key": "A short stable label for the memory, such as preferred_format.",
                    "value": "The exact preference, fact, or project detail to remember.",
                    "category": "A short category, such as preference, project, profile, or instruction.",
                }
            ),
            handler=lambda args: remember_preference(memory, args),
        ),
        Tool(
            name="list_memories",
            description="Use this when the user asks what Zordon remembers.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda args: list_memories(memory),
        ),
    ]


def _object_schema(properties: dict[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": description}
            for name, description in properties.items()
        },
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def refine_writing(arguments: dict) -> str:
    draft = _required_text(arguments, "draft")
    goal = _required_text(arguments, "goal")
    return (
        "Refined draft:\n"
        f"{_tighten_text(draft)}\n\n"
        "Editing note:\n"
        f"Optimized for {goal}. Keep the user's meaning, remove filler, and prefer concrete language."
    )


def create_image_prompt(arguments: dict) -> str:
    idea = _required_text(arguments, "idea")
    fmt = _required_text(arguments, "format")
    return (
        f"Image prompt for {fmt}:\n"
        f"{idea}. Cinematic composition, clear subject, intentional lighting, detailed environment, "
        "natural textures, strong focal point, cohesive color palette, high visual clarity. "
        "Avoid text artifacts, distorted anatomy, muddy details, and cluttered framing."
    )


def generate_image(config, root: Path, arguments: dict) -> str:
    if config is None:
        raise ValueError("image generation requires config")
    prompt = _required_text(arguments, "prompt")
    filename = arguments.get("filename")
    if isinstance(filename, str) and not filename.strip():
        filename = None
    project = active_project(root)
    output_dir = project_images_dir(project) if project is not None else None
    image = create_image_file(
        config,
        root,
        prompt,
        filename if isinstance(filename, str) else None,
        output_dir=output_dir,
    )
    detail = f"Generated image saved to {image.path}"
    if project is not None:
        detail += f"\nProject: {project.name}"
    if image.revised_prompt:
        detail += f"\nRevised prompt: {image.revised_prompt}"
    return detail


def remember_preference(memory: MemoryStore, arguments: dict) -> str:
    key = _required_text(arguments, "key")
    value = _required_text(arguments, "value")
    category = _required_text(arguments, "category")
    saved = memory.remember(key=key, value=value, category=category)
    return f"Remembered {saved.key}: {saved.value}"


def list_memories(memory: MemoryStore) -> str:
    summary = memory.summary(limit=50)
    if not summary:
        return "No durable memories saved yet."
    return summary


def save_draft(root: Path, arguments: dict) -> str:
    filename = _required_text(arguments, "filename")
    content = _required_text(arguments, "content")
    filename = safe_draft_filename(filename)
    project = active_project(root)
    if project is None:
        path = write_draft(root, filename, content)
        return f"Saved draft to {path}"

    directory = project_drafts_dir(project)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return f"Saved draft to {path}\nProject: {project.name}"


def fetch_web_page(arguments: dict) -> str:
    url = _required_text(arguments, "url")
    question = _required_text(arguments, "question")
    if not re.match(r"^https?://", url):
        raise ValueError("url must start with http:// or https://")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Zordon/0.1 (+local assistant harness)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            content_type = response.headers.get("content-type", "")
            raw = response.read(300_000)
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not fetch URL: {exc}") from exc

    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type or 'unknown'}")

    text = raw.decode("utf-8", errors="replace")
    if "text/html" in content_type:
        parser = ReadableHTMLParser()
        parser.feed(text)
        text = parser.text()

    text = re.sub(r"\s+", " ", text).strip()
    excerpt = text[:4000] if text else "(No readable text found.)"
    return json.dumps(
        {
            "url": url,
            "question": question,
            "excerpt": excerpt,
        },
        ensure_ascii=True,
    )


def _required_text(arguments: dict, key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _tighten_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "really ": "",
        "very ": "",
        "just ": "",
        "I think ": "",
    }
    for needle, replacement in replacements.items():
        cleaned = cleaned.replace(needle, replacement)
    return cleaned


class ReadableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)
