from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

from . import __status__, __version__
from .agent import Agent
from .audit import AuditLog
from .config import Config
from .doctor import doctor_text
from .drafts import (
    delete_draft,
    drafts_dir,
    list_drafts,
    read_draft,
    read_latest_draft,
    safe_draft_filename,
)
from .heartbeat import build_pulse
from .images import images_dir, latest_image, list_images, open_image
from .logs import list_logs, read_log_tail, search_log, summarize_events, summarize_last_turn
from .memory import MemoryStore
from .projects import (
    active_project,
    content_plan_path,
    copy_draft_to_active_project,
    copy_image_to_active_project,
    create_project,
    format_active_project,
    format_projects,
    latest_project_draft,
    latest_project_image,
    list_project_drafts,
    list_project_images,
    project_brief_path,
    project_drafts_dir,
    project_images_dir,
    read_content_plan,
    read_project_brief,
    require_active_project,
    set_active_project,
    write_content_plan,
    write_project_brief,
)
from .state import open_path, open_state_folder, state_text


def help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "/help - show this command list",
            "/version - show local build details",
            "/doctor - run offline health checks",
            "/state - show where Zordon stores memory, drafts, images, and logs",
            "/open-state - stage the Zordon state folder to open",
            "/open-drafts - stage the drafts folder to open",
            "/open-images - stage the images folder to open",
            "/open-logs - stage the logs folder to open",
            "/open-memory - stage the memory file to open",
            "/setup-voice - show the push-to-talk dependency install command",
            "/confirm - approve a pending action",
            "/cancel - cancel a pending action",
            "/commands - show the short voice command list",
            "/ready - show a compact startup readiness check",
            "/recent - show recent user and assistant turns",
            "/project - show the active project",
            "/project-status - show active project status",
            "/projects - list saved projects",
            "/new-project <name> - create and use a project",
            "/use-project <name> - switch active project",
            "/open-project - stage the active project folder to open",
            "/project-drafts - list drafts in the active project",
            "/project-images - list images in the active project",
            "/save-latest-draft-to-project - copy the latest draft into the active project",
            "/save-latest-image-to-project - copy the latest image into the active project",
            "/open-project-drafts - stage the active project drafts folder to open",
            "/open-project-images - stage the active project images folder to open",
            "/open-latest-project-draft - stage the latest project draft to open",
            "/open-latest-project-image - stage the latest project image to open",
            "/project-brief - show the active project brief",
            "/set-project-brief <text> - save the active project brief",
            "/open-project-brief - stage the active project brief to open",
            "/project-brief-next - turn the active project brief into one draft prompt",
            "/content-plan - build a local content plan for the active project",
            "/save-content-plan - stage the latest content plan to save",
            "/open-content-plan - stage the saved content plan to open",
            "/project-next - show the next concrete action for the active project",
            "/today - summarize today's local Zordon activity",
            "/pulse - show local state without calling the model",
            "/drafts - list saved drafts",
            "/latest-draft - print the most recently modified draft",
            "/open-latest-draft - stage the latest draft to open",
            "/show-draft <filename> - print a saved draft",
            "/delete-draft <filename> - stage a draft for deletion",
            "/images - list saved generated images",
            "/latest-image - show the most recently generated image path",
            "/open-latest-image - stage the latest generated image to open",
            "/logs - list audit logs",
            "/show-log <filename> - print the last 20 lines of an audit log",
            "/events [filename] - summarize event counts from an audit log",
            "/last [filename] - show the last user and assistant turn from an audit log",
            "/search-log <term> or /search-log <filename> <term> - search audit log text",
            "/memory - list saved memories",
            "/memory --raw - print raw memory JSON lines",
            "/remember <key> <value> - save a manual memory",
            "/forget <key> - delete a memory",
            "/reset - clear session history and pending actions",
            "/kill - clear pending state and quit immediately",
            "/exit - quit",
        ]
    )


@dataclass(frozen=True)
class CommandContext:
    config: Config
    agent: Agent
    memory: MemoryStore
    audit: AuditLog
    root: Path


@dataclass(frozen=True)
class CommandOutcome:
    handled: bool
    output: str = ""
    should_exit: bool = False


class CommandRouter:
    def __init__(self, context: CommandContext):
        self.context = context
        self.pending_delete_draft: str | None = None
        self.pending_open_image: Path | None = None
        self.pending_open_state = False
        self.pending_open_path: tuple[str, Path, bool] | None = None
        self.pending_content_plan: str | None = None
        self.last_content_plan: str | None = None

    def handle(self, user_text: str) -> CommandOutcome:
        lowered = user_text.lower()
        if lowered in {"/kill", "kill"}:
            return self._kill()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            return CommandOutcome(handled=True, output="Goodbye.", should_exit=True)
        if user_text == "/help":
            return CommandOutcome(handled=True, output=help_text())
        if user_text == "/version":
            return CommandOutcome(handled=True, output=version_text())
        if lowered in {"/commands", "commands"}:
            return CommandOutcome(handled=True, output=commands_text())
        if user_text == "/doctor":
            return CommandOutcome(
                handled=True,
                output=doctor_text(
                    self.context.config,
                    self.context.memory,
                    self.context.agent.tools,
                    self.context.root,
                ),
            )
        if user_text == "/state":
            return CommandOutcome(handled=True, output=state_text(self.context.root))
        if lowered in {"/open-state", "/open state"}:
            self.pending_open_state = True
            return CommandOutcome(
                handled=True, output="Open Zordon state folder? Say Confirm Or Deny."
            )
        if user_text == "/setup-voice":
            return CommandOutcome(handled=True, output=setup_voice_text(self.context.root))
        if user_text == "/reset":
            self.context.agent.reset()
            self.pending_delete_draft = None
            return CommandOutcome(handled=True, output="Session history cleared.")
        if lowered in {"/cancel", "cancel", "/deny", "deny"}:
            return CommandOutcome(handled=True, output=self._cancel())
        if lowered in {"/confirm", "confirm", "confrim"}:
            return CommandOutcome(handled=True, output=self._confirm())
        if lowered in {"/today", "today"}:
            return CommandOutcome(handled=True, output=self._today())
        if lowered in {"/ready", "ready"}:
            return CommandOutcome(handled=True, output=self._ready())
        if lowered in {"/recent", "recent"}:
            return CommandOutcome(handled=True, output=self._recent())
        if lowered in {"/project", "project"}:
            return CommandOutcome(handled=True, output=format_active_project(self.context.root))
        if lowered in {"/project-status", "/project status", "project status"}:
            return CommandOutcome(handled=True, output=self._project_status())
        if lowered in {"/projects", "projects"}:
            return CommandOutcome(handled=True, output=format_projects(self.context.root))
        if user_text.startswith("/new-project"):
            return CommandOutcome(handled=True, output=self._new_project(user_text))
        if user_text.startswith("/use-project"):
            return CommandOutcome(handled=True, output=self._use_project(user_text))
        if lowered in {"/open-project", "/open project"}:
            return CommandOutcome(handled=True, output=self._stage_open_project())
        if lowered in {"/project-drafts", "/project drafts"}:
            return CommandOutcome(handled=True, output=self._project_drafts())
        if lowered in {"/project-images", "/project images"}:
            return CommandOutcome(handled=True, output=self._project_images())
        if lowered in {
            "/open-project-drafts",
            "/open project drafts",
            "/open project draft folder",
        }:
            return CommandOutcome(handled=True, output=self._stage_open_project_drafts())
        if lowered in {
            "/open-project-images",
            "/open project images",
            "/open project image folder",
        }:
            return CommandOutcome(handled=True, output=self._stage_open_project_images())
        if lowered in {"/save-latest-draft-to-project", "/save latest draft to project"}:
            return CommandOutcome(handled=True, output=self._save_latest_draft_to_project())
        if lowered in {"/save-latest-image-to-project", "/save latest image to project"}:
            return CommandOutcome(handled=True, output=self._save_latest_image_to_project())
        if lowered in {"/open-latest-project-draft", "/open latest project draft"}:
            return CommandOutcome(handled=True, output=self._stage_open_latest_project_draft())
        if lowered in {"/open-latest-project-image", "/open latest project image"}:
            return CommandOutcome(handled=True, output=self._stage_open_latest_project_image())
        if lowered in {"/project-brief", "/project brief"}:
            return CommandOutcome(handled=True, output=self._project_brief())
        if user_text.startswith("/set-project-brief"):
            return CommandOutcome(handled=True, output=self._set_project_brief(user_text))
        if lowered in {"/open-project-brief", "/open project brief"}:
            return CommandOutcome(handled=True, output=self._stage_open_project_brief())
        if lowered in {"/project-brief-next", "/project brief next", "project brief next"}:
            return CommandOutcome(handled=True, output=self._project_brief_next())
        if lowered in {"/content-plan", "/content plan", "content plan"}:
            return CommandOutcome(handled=True, output=self._content_plan())
        if lowered in {"/save-content-plan", "/save content plan", "save content plan"}:
            return CommandOutcome(handled=True, output=self._stage_save_content_plan())
        if lowered in {"/open-content-plan", "/open content plan", "open content plan"}:
            return CommandOutcome(handled=True, output=self._stage_open_content_plan())
        if lowered in {"/project-next", "/project next", "project next"}:
            return CommandOutcome(handled=True, output=self._project_next())
        if user_text == "/pulse":
            pulse = build_pulse(
                self.context.config,
                self.context.memory,
                self.context.agent.tools,
                self.context.audit.path,
            )
            return CommandOutcome(handled=True, output=pulse.as_text())
        if lowered in {"/drafts", "/draft"}:
            drafts = list_drafts(self.context.root)
            return CommandOutcome(
                handled=True,
                output="\n".join(drafts) if drafts else "No drafts saved yet.",
            )
        if lowered in {"/open-drafts", "/open drafts", "/open-draft-folder", "/open draft folder"}:
            return CommandOutcome(
                handled=True,
                output=self._stage_open_path("drafts folder", drafts_dir(self.context.root), True),
            )
        if lowered in {"/latest-draft", "/latest-drafts", "/latest draft", "/latest drafts"}:
            return CommandOutcome(handled=True, output=self._latest_draft())
        if lowered in {"/open-latest-draft", "/open latest draft", "/open latest drafts"}:
            return CommandOutcome(handled=True, output=self._stage_open_latest_draft())
        if user_text.startswith("/show-draft"):
            return CommandOutcome(handled=True, output=self._show_draft(user_text))
        if user_text.startswith("/delete-draft"):
            return CommandOutcome(handled=True, output=self._stage_delete_draft(user_text))
        if lowered in {"/images", "/image"}:
            images = list_images(self.context.root)
            return CommandOutcome(
                handled=True,
                output="\n".join(images) if images else "No images saved yet.",
            )
        if lowered in {"/open-images", "/open images", "/open-image-folder", "/open image folder"}:
            return CommandOutcome(
                handled=True,
                output=self._stage_open_path("images folder", images_dir(self.context.root), True),
            )
        if lowered in {"/latest-image", "/latest-images", "/latest image", "/latest images"}:
            return CommandOutcome(handled=True, output=self._latest_image())
        if lowered in {
            "/open-latest-image",
            "/open latest image",
            "/open-latest-images",
            "/open latest images",
        }:
            return CommandOutcome(handled=True, output=self._stage_open_latest_image())
        if user_text == "/logs":
            logs = list_logs(self.context.root)
            return CommandOutcome(
                handled=True,
                output="\n".join(logs) if logs else "No audit logs saved yet.",
            )
        if lowered in {"/open-logs", "/open logs", "/open-log-folder", "/open log folder"}:
            return CommandOutcome(
                handled=True,
                output=self._stage_open_path("logs folder", self.context.root / "logs", True),
            )
        if user_text.startswith("/show-log"):
            return CommandOutcome(handled=True, output=self._show_log(user_text))
        if user_text.startswith("/events"):
            return CommandOutcome(handled=True, output=self._events(user_text))
        if user_text.startswith("/last"):
            return CommandOutcome(handled=True, output=self._last(user_text))
        if user_text.startswith("/search-log"):
            return CommandOutcome(handled=True, output=self._search_log(user_text))
        if user_text == "/memory":
            summary = self.context.memory.summary(limit=50)
            return CommandOutcome(
                handled=True,
                output=summary if summary else "No durable memories saved yet.",
            )
        if lowered in {"/open-memory", "/open memory", "/open memory file"}:
            if self.context.memory.path is None:
                return CommandOutcome(handled=True, output="No memory file is configured.")
            return CommandOutcome(
                handled=True,
                output=self._stage_open_path("memory file", self.context.memory.path, False),
            )
        if user_text == "/memory --raw":
            raw_lines = self.context.memory.raw_lines()
            return CommandOutcome(
                handled=True,
                output="\n".join(raw_lines) if raw_lines else "No durable memories saved yet.",
            )
        if user_text.startswith("/remember"):
            return CommandOutcome(handled=True, output=self._remember(user_text))
        if user_text.startswith("/forget"):
            return CommandOutcome(handled=True, output=self._forget(user_text))
        return CommandOutcome(handled=False)

    def _cancel(self) -> str:
        if self.pending_delete_draft is not None:
            filename = self.pending_delete_draft
            self.pending_delete_draft = None
            return f"Cancelled delete-draft {filename}."
        if self.pending_open_image is not None:
            path = self.pending_open_image
            self.pending_open_image = None
            return f"Cancelled open-image {path.name}."
        if self.pending_open_state:
            self.pending_open_state = False
            return "Cancelled open-state."
        if self.pending_open_path is not None:
            label, _, _ = self.pending_open_path
            self.pending_open_path = None
            return f"Cancelled open-{label}."
        if self.pending_content_plan is not None:
            self.pending_content_plan = None
            return "Cancelled save-content-plan."
        return self.context.agent.cancel_pending()

    def _kill(self) -> CommandOutcome:
        self.pending_delete_draft = None
        self.pending_open_image = None
        self.pending_open_state = False
        self.pending_open_path = None
        self.pending_content_plan = None
        self.context.agent.reset()
        self.context.audit.record("kill_switch")
        return CommandOutcome(
            handled=True,
            output="Kill switch engaged. Pending state cleared. Goodbye.",
            should_exit=True,
        )

    def _confirm(self) -> str:
        if self.pending_delete_draft is not None:
            filename = self.pending_delete_draft
            self.pending_delete_draft = None
            try:
                deleted_path = delete_draft(self.context.root, filename)
                return f"Deleted draft {deleted_path.name}."
            except (FileNotFoundError, ValueError) as exc:
                return str(exc)
        if self.pending_open_image is not None:
            path = self.pending_open_image
            self.pending_open_image = None
            try:
                open_image(path)
                return f"Opened image {path}."
            except OSError as exc:
                return f"Could not open image {path}: {exc}"
        if self.pending_open_state:
            self.pending_open_state = False
            try:
                path = open_state_folder(self.context.root)
                return f"Opened state folder {path}."
            except OSError as exc:
                return f"Could not open state folder: {exc}"
        if self.pending_open_path is not None:
            label, path, create_directory = self.pending_open_path
            self.pending_open_path = None
            try:
                opened = open_path(path, create_directory=create_directory)
                return f"Opened {label} {opened}."
            except OSError as exc:
                return f"Could not open {label}: {exc}"
        if self.pending_content_plan is not None:
            content = self.pending_content_plan
            self.pending_content_plan = None
            try:
                path = write_content_plan(self.context.root, content)
            except (FileNotFoundError, ValueError) as exc:
                return str(exc)
            return f"Saved content plan to {path}."

        chunks = list(self.context.agent.confirm_pending())
        body = "".join(chunks) if chunks else "(No text received.)"
        return f"{self.context.config.assistant_name}: {body}"

    def _show_draft(self, user_text: str) -> str:
        _, _, filename = user_text.partition(" ")
        if not filename.strip():
            return "Usage: /show-draft <filename>"
        try:
            return read_draft(self.context.root, filename.strip())
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)

    def _today(self) -> str:
        lines = ["Zordon today"]
        lines.append(f"- audit log: {self.context.audit.path or '(in-memory only)'}")
        lines.append(f"- events today: {self._today_event_count()}")
        lines.append(f"- memories: {len(self.context.memory.all())}")
        lines.append(f"- latest draft: {self._latest_draft_name()}")
        lines.append(f"- latest image: {self._latest_image_name()}")
        return "\n".join(lines)

    def _ready(self) -> str:
        tool_count = len(self.context.agent.tools.names())
        state_status = (
            "durable" if "location: durable" in state_text(self.context.root) else "check state"
        )
        return "\n".join(
            [
                "Zordon ready",
                f"- state: {state_status}",
                f"- memory: {len(self.context.memory.all())} entries",
                f"- latest draft: {self._latest_draft_name()}",
                f"- latest image: {self._latest_image_name()}",
                f"- tools: {tool_count} loaded",
                f"- project: {self._active_project_name()}",
            ]
        )

    def _recent(self, limit: int = 6) -> str:
        events = self._recent_turn_events(limit)
        if not events:
            return "No recent turns found."
        lines = ["Zordon recent"]
        lines.extend(events)
        return "\n".join(lines)

    def _recent_turn_events(self, limit: int) -> list[str]:
        path = self.context.audit.path
        if path is None or not path.exists() or not path.is_file():
            source_events = self.context.audit.events or []
        else:
            source_events = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    source_events.append(event)

        turns: list[str] = []
        for event in source_events:
            name = event.get("event")
            text = str(event.get("text") or event.get("output") or "").strip()
            if not text:
                continue
            if name in {"user_turn", "voice_transcript"}:
                turns.append(f"- You: {self._clip_recent(text)}")
            elif name == "assistant_reply":
                turns.append(f"- Zordon: {self._clip_recent(text)}")
        return turns[-limit:]

    @staticmethod
    def _clip_recent(text: str, limit: int = 140) -> str:
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _today_event_count(self) -> int:
        path = self.context.audit.path
        if path is None or not path.exists() or not path.is_file():
            return len(self.context.audit.events or [])
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
        return count

    def _latest_draft_name(self) -> str:
        try:
            path, _ = read_latest_draft(self.context.root)
            return path.name
        except FileNotFoundError:
            return "(none)"

    def _latest_image_name(self) -> str:
        try:
            return latest_image(self.context.root).name
        except FileNotFoundError:
            return "(none)"

    def _active_project_name(self) -> str:
        project = active_project(self.context.root)
        return project.name if project else "(none)"

    def _project_status(self) -> str:
        project = active_project(self.context.root)
        if project is None:
            return "No active project."
        drafts = list_project_drafts(self.context.root)
        images = list_project_images(self.context.root)
        _, brief = read_project_brief(self.context.root)
        brief_status = "saved" if brief != "No project brief saved yet." else "missing"
        latest_draft = latest_project_draft(self.context.root).name if drafts else "(none)"
        latest_image_name = latest_project_image(self.context.root).name if images else "(none)"
        return "\n".join(
            [
                "Project status",
                f"- name: {project.name}",
                f"- brief: {brief_status}",
                f"- drafts: {len(drafts)}",
                f"- latest draft: {latest_draft}",
                f"- images: {len(images)}",
                f"- latest image: {latest_image_name}",
                f"- folder: {project.path}",
            ]
        )

    def _new_project(self, user_text: str) -> str:
        _, _, name = user_text.partition(" ")
        if not name.strip():
            return "Usage: /new-project <name>"
        try:
            project = create_project(self.context.root, name.strip())
        except ValueError as exc:
            return str(exc)
        return f"Created project {project.name} ({project.slug})."

    def _use_project(self, user_text: str) -> str:
        _, _, name = user_text.partition(" ")
        if not name.strip():
            return "Usage: /use-project <name>"
        try:
            project = set_active_project(self.context.root, name.strip())
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)
        return f"Using project {project.name} ({project.slug})."

    def _stage_open_project(self) -> str:
        project = active_project(self.context.root)
        if project is None:
            return "No active project."
        return self._stage_open_path("project folder", project.path, True)

    def _project_drafts(self) -> str:
        try:
            drafts = list_project_drafts(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        return "\n".join(drafts) if drafts else "No project drafts saved yet."

    def _project_images(self) -> str:
        try:
            images = list_project_images(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        return "\n".join(images) if images else "No project images saved yet."

    def _stage_open_project_drafts(self) -> str:
        try:
            project = require_active_project(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        return self._stage_open_path("project drafts folder", project_drafts_dir(project), True)

    def _stage_open_project_images(self) -> str:
        try:
            project = require_active_project(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        return self._stage_open_path("project images folder", project_images_dir(project), True)

    def _save_latest_draft_to_project(self) -> str:
        try:
            source, _ = read_latest_draft(self.context.root)
            target = copy_draft_to_active_project(self.context.root, source)
        except FileNotFoundError as exc:
            return str(exc)
        return f"Saved latest draft to project: {target.name}."

    def _save_latest_image_to_project(self) -> str:
        try:
            source = latest_image(self.context.root)
            target = copy_image_to_active_project(self.context.root, source)
        except FileNotFoundError as exc:
            return str(exc)
        return f"Saved latest image to project: {target.name}."

    def _stage_open_latest_project_draft(self) -> str:
        try:
            path = latest_project_draft(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        self.pending_open_path = ("latest project draft", path, False)
        return f"Open latest project draft {path.name}? Say Confirm Or Deny."

    def _stage_open_latest_project_image(self) -> str:
        try:
            path = latest_project_image(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        self.pending_open_path = ("latest project image", path, False)
        return f"Open latest project image {path.name}? Say Confirm Or Deny."

    def _project_brief(self) -> str:
        try:
            project, content = read_project_brief(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        return f"Project brief: {project.name}\n{content}"

    def _set_project_brief(self, user_text: str) -> str:
        _, _, content = user_text.partition(" ")
        if not content.strip():
            return "Usage: /set-project-brief <text>"
        try:
            path = write_project_brief(self.context.root, content.strip())
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)
        return f"Saved project brief to {path.name}."

    def _stage_open_project_brief(self) -> str:
        try:
            project = require_active_project(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        path = project_brief_path(project)
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return self._stage_open_path("project brief", path, False)

    def _project_brief_next(self) -> str:
        try:
            project, brief = read_project_brief(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        if brief == "No project brief saved yet.":
            return "No project brief saved yet. Use /set-project-brief <text> first."
        return "\n".join(
            [
                "Project brief next",
                f"- project: {project.name}",
                "- prompt:",
                (
                    f"Draft a short content piece for {project.name}. Base it on this project brief: "
                    f"{brief} Keep the tone clear, calm, and useful. Include a strong opening line, "
                    "one concise body section, and a simple closing line."
                ),
            ]
        )

    def _content_plan(self) -> str:
        try:
            plan = self._build_content_plan()
        except FileNotFoundError as exc:
            return str(exc)
        self.last_content_plan = plan
        return plan

    def _stage_save_content_plan(self) -> str:
        try:
            self.pending_content_plan = self.last_content_plan or self._build_content_plan()
        except FileNotFoundError as exc:
            return str(exc)
        return "Save content plan to content-plan.md? Say Confirm Or Deny."

    def _stage_open_content_plan(self) -> str:
        try:
            project = require_active_project(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        path = content_plan_path(project)
        if not path.exists():
            return "No content plan saved yet."
        return self._stage_open_path("content plan", path, False)

    def _build_content_plan(self) -> str:
        project = require_active_project(self.context.root)
        _, brief = read_project_brief(self.context.root)
        drafts = list_project_drafts(self.context.root)
        images = list_project_images(self.context.root)
        goal = brief
        if brief == "No project brief saved yet.":
            goal = "Define the project promise, audience, and first publishable asset."
        latest_draft = latest_project_draft(self.context.root).name if drafts else "none yet"
        latest_image = latest_project_image(self.context.root).name if images else "none yet"
        return "\n".join(
            [
                f"Content plan: {project.name}",
                "",
                "Project goal",
                goal,
                "",
                "Content ideas",
                "1. Project origin post: introduce the idea, tone, and why it exists.",
                "2. Process post: show one draft, image, or decision from the project folder.",
                "3. Drop announcement: turn the strongest asset into a short launch/update post.",
                "",
                "Image concepts",
                "1. Signature cover art using the project color, title, and central visual symbol.",
                "2. Behind-the-scenes visual showing the project as an active creative workspace.",
                "3. Square social teaser that can pair with the next draft or announcement.",
                "",
                "Caption angles",
                "1. Calm authority: direct, clean, and confident.",
                "2. Creator log: what changed today and what is next.",
                "3. Audience hook: ask people what direction the project should take next.",
                "",
                "Next draft",
                f"Write a short project intro for {project.name}. Use latest draft: {latest_draft}. Use latest image: {latest_image}.",
            ]
        )

    def _project_next(self) -> str:
        try:
            project = require_active_project(self.context.root)
            _, brief = read_project_brief(self.context.root)
            _, content_plan = read_content_plan(self.context.root)
            drafts = list_project_drafts(self.context.root)
            images = list_project_images(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)

        if brief == "No project brief saved yet.":
            action = f"Set a short project brief for {project.name}."
            command = "/set-project-brief <one sentence>"
            reason = "Zordon needs the project goal before it can make useful content decisions."
        elif content_plan == "No content plan saved yet.":
            action = f"Create and save a content plan for {project.name}."
            command = "/content-plan, then /save-content-plan"
            reason = "The brief exists, but there is no saved plan to work from."
        elif not drafts:
            action = f"Write the first project draft for {project.name}."
            command = "Say: Draft a short project intro for this project."
            reason = "The content plan is saved, but the project has no draft asset yet."
        elif not images:
            action = f"Create the first project image for {project.name}."
            command = "Say: Create an image for this project."
            reason = "The project has writing started, but no image asset yet."
        else:
            latest_draft = latest_project_draft(self.context.root).name
            latest_image = latest_project_image(self.context.root).name
            action = f"Turn {latest_draft} and {latest_image} into one publishable post."
            command = "Say: Refine this into a social post."
            reason = "The project has a brief, plan, draft, and image. The next useful move is packaging."

        return "\n".join(
            [
                "Project next",
                f"- project: {project.name}",
                f"- action: {action}",
                f"- command: {command}",
                f"- why: {reason}",
            ]
        )

    def _latest_draft(self) -> str:
        try:
            path, content = read_latest_draft(self.context.root)
            return f"{path.name}\n{content}"
        except FileNotFoundError as exc:
            return str(exc)

    def _stage_open_latest_draft(self) -> str:
        try:
            path, _ = read_latest_draft(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        self.pending_open_path = ("latest draft", path, False)
        return f"Open latest draft {path.name}? Say Confirm Or Deny."

    def _latest_image(self) -> str:
        try:
            path = latest_image(self.context.root)
            return str(path)
        except FileNotFoundError as exc:
            return str(exc)

    def _stage_open_latest_image(self) -> str:
        try:
            path = latest_image(self.context.root)
        except FileNotFoundError as exc:
            return str(exc)
        self.pending_open_image = path
        return f"Open latest image {path.name}? Say Confirm Or Deny."

    def _stage_open_path(self, label: str, path: Path, create_directory: bool) -> str:
        self.pending_open_path = (label, path, create_directory)
        return f"Open {label}? Say Confirm Or Deny."

    def _stage_delete_draft(self, user_text: str) -> str:
        _, _, filename = user_text.partition(" ")
        if not filename.strip():
            return "Usage: /delete-draft <filename>"
        try:
            self.pending_delete_draft = safe_draft_filename(filename.strip())
        except ValueError as exc:
            return str(exc)
        return f"Delete draft {self.pending_delete_draft}? Say Confirm Or Deny."

    def _show_log(self, user_text: str) -> str:
        _, _, filename = user_text.partition(" ")
        if not filename.strip():
            return "Usage: /show-log <filename>"
        try:
            return read_log_tail(self.context.root, filename.strip())
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)

    def _events(self, user_text: str) -> str:
        _, _, filename = user_text.partition(" ")
        try:
            return summarize_events(self.context.root, filename.strip() or None)
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)

    def _last(self, user_text: str) -> str:
        _, _, filename = user_text.partition(" ")
        try:
            return summarize_last_turn(self.context.root, filename.strip() or None)
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)

    def _search_log(self, user_text: str) -> str:
        _, _, payload = user_text.partition(" ")
        payload = payload.strip()
        if not payload:
            return "Usage: /search-log <term> or /search-log <filename> <term>"
        first, _, rest = payload.partition(" ")
        filename = first if first.endswith(".jsonl") else None
        term = rest if filename else payload
        if not term.strip():
            return "Usage: /search-log <term> or /search-log <filename> <term>"
        try:
            return search_log(self.context.root, term.strip(), filename)
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)

    def _remember(self, user_text: str) -> str:
        _, _, payload = user_text.partition(" ")
        key, _, value = payload.strip().partition(" ")
        if not key.strip() or not value.strip():
            return "Usage: /remember <memory_key> <memory_value>"
        saved = self.context.memory.remember(key.strip(), value.strip(), "manual")
        return f"Remembered {saved.key}: {saved.value}"

    def _forget(self, user_text: str) -> str:
        _, _, key = user_text.partition(" ")
        if not key.strip():
            return "Usage: /forget <memory_key>"
        key = key.strip()
        if self.context.memory.forget(key):
            return f"Forgot {key}."
        return f"No memory found for {key}."


def version_text() -> str:
    command_count = len(help_text().splitlines()) - 1
    return "\n".join(
        [
            f"Zordon {__version__}",
            f"- status: {__status__}",
            f"- commands: {command_count}",
            f"- python: {platform.python_version()}",
        ]
    )


def commands_text() -> str:
    return "\n".join(
        [
            "Zordon commands",
            "/ready",
            "/today",
            "/recent",
            "/state",
            "/doctor",
            "/drafts",
            "/images",
            "/open-state",
            "/open-drafts",
            "/open-images",
            "/open-logs",
            "/open-memory",
            "/open-latest-draft",
            "/open-latest-image",
            "/project",
            "/project-status",
            "/projects",
            "/new-project <name>",
            "/use-project <name>",
            "/open-project",
            "/project-drafts",
            "/project-images",
            "/save-latest-draft-to-project",
            "/save-latest-image-to-project",
            "/open-project-drafts",
            "/open-project-images",
            "/open-latest-project-draft",
            "/open-latest-project-image",
            "/project-brief",
            "/set-project-brief <text>",
            "/open-project-brief",
            "/project-brief-next",
            "/content-plan",
            "/save-content-plan",
            "/open-content-plan",
            "/project-next",
        ]
    )


def setup_voice_text(root: Path) -> str:
    requirements = root / "requirements-voice.txt"
    return "\n".join(
        [
            "Voice setup:",
            "Install push-to-talk dependencies with:",
            (
                "& 'C:\\Users\\Jones\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' "
                f"-m pip install -r '{requirements}'"
            ),
            "Then run /doctor to confirm sounddevice and keyboard are available.",
        ]
    )
