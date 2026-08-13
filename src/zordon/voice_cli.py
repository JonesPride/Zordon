from __future__ import annotations

import argparse
import re
import time

from .agent import Agent
from .audit import default_audit_log
from .audio_io import AudioError, cue_sound, record_while_key_held
from .audio_provider import OpenAIAudioProvider
from .commands import CommandContext, CommandRouter
from .config import ROOT, ROOT as CONFIG_ROOT, load_config, load_dotenv, state_root
from .memory import default_memory_store
from .openai_provider import OpenAIProvider, ProviderError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Zordon with push-to-talk voice.")
    parser.add_argument(
        "--typed",
        action="store_true",
        help="Use the voice-mode agent with typed input instead of microphone capture.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config()
    runtime_root = state_root() if ROOT == CONFIG_ROOT else ROOT
    audit = default_audit_log(runtime_root)
    memory = default_memory_store(runtime_root)
    agent = Agent(config=config, provider=OpenAIProvider(config), audit=audit, memory=memory)
    router = CommandRouter(
        CommandContext(
            config=config,
            agent=agent,
            memory=memory,
            audit=audit,
            root=runtime_root,
        )
    )
    audio = OpenAIAudioProvider(config)

    print(f"{config.assistant_name} push-to-talk")
    print(f"Hold {config.push_to_talk_key.upper()} to speak. Press Ctrl+C to exit.")
    print("Use --typed for the same voice-mode agent without microphone capture.")
    print("Typed fallback supports slash commands. Use /help for the command list.")
    print("Transcript will print before each answer while we build.")
    print(f"Audit log: {audit.path}")
    print(f"Memory file: {memory.path}")

    while True:
        try:
            turn_started = time.perf_counter()
            if args.typed:
                try:
                    transcript = input("\nType fallback: ").strip()
                except EOFError:
                    print("\nGoodbye.")
                    return 0
                if transcript.lower() in {"exit", "quit", "/exit", "/quit"}:
                    print("Goodbye.")
                    return 0
                if not transcript:
                    continue
                record_seconds = 0.0
                transcribe_seconds = 0.0
            else:
                record_started = time.perf_counter()
                wav_path = record_while_key_held(config.push_to_talk_key, config.audio_sample_rate)
                record_seconds = time.perf_counter() - record_started
                cue_sound("thinking")
                transcribe_started = time.perf_counter()
                print("Thinking...")
                transcript = audio.transcribe(wav_path)
                transcribe_seconds = time.perf_counter() - transcribe_started

            if not transcript:
                print("Transcript was empty. Try again.")
                continue

            command_text = normalize_spoken_command(transcript)
            command = router.handle(command_text)
            if command.handled:
                if command_text != transcript:
                    print(f"Command: {command_text}")
                print(command.output)
                if command.should_exit:
                    return 0
                continue
            if is_command_candidate(command_text):
                if command_text != transcript:
                    print(f"Command: {command_text}")
                print(f"Unknown command: {command_text}. Use /help for the command list.")
                continue

            print(f"You said: {transcript}")
            audit.record("voice_transcript", text=transcript)

            print(f"{config.assistant_name}: ", end="", flush=True)
            chunks: list[str] = []
            think_started = time.perf_counter()
            for chunk in agent.respond(transcript):
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            print()
            think_seconds = time.perf_counter() - think_started

            reply = "".join(chunks).strip()
            if reply:
                speak_started = time.perf_counter()
                _, interrupted = audio.speak(reply, interrupt_key=config.push_to_talk_key)
                speak_seconds = time.perf_counter() - speak_started
                if interrupted:
                    audit.record("tts_interrupted", key=config.push_to_talk_key)
                    print("Speech interrupted.")
                else:
                    cue_sound("done")
            else:
                speak_seconds = 0.0

            audit.record(
                "voice_turn_timing",
                record_seconds=round(record_seconds, 3),
                transcribe_seconds=round(transcribe_seconds, 3),
                think_seconds=round(think_seconds, 3),
                speak_seconds=round(speak_seconds, 3),
                total_seconds=round(time.perf_counter() - turn_started, 3),
            )
        except KeyboardInterrupt:
            print("\nGoodbye.")
            return 0
        except (AudioError, ProviderError) as exc:
            if "No audio was captured" in str(exc):
                print("No audio captured. Hold the key a little longer and try again.")
                continue
            cue_sound("error")
            audit.record("voice_error", error=str(exc))
            print(f"\nVoice turn failed: {exc}")
            print("Try again when ready, or restart with --typed for a no-microphone fallback.")


def normalize_spoken_command(transcript: str) -> str:
    text = transcript.strip()
    lowered = re.sub(r"[.,!?]+$", "", text.lower()).strip()
    lowered = re.sub(r"[,;:]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = _split_compact_slash_command(lowered)

    slash_aliases = {
        "forward slash": "/",
        "slash": "/",
    }
    for prefix, replacement in slash_aliases.items():
        if lowered.startswith(prefix + " "):
            lowered = replacement + lowered.removeprefix(prefix).strip()
            break

    dynamic_command = _dynamic_spoken_command(lowered)
    if dynamic_command is not None:
        return dynamic_command

    simple_commands = {
        "confirm": "confirm",
        "/confirm": "/confirm",
        "confirmed": "confirm",
        "yes confirm": "confirm",
        "cancel": "cancel",
        "/cancel": "/cancel",
        "cancel it": "cancel",
        "deny": "deny",
        "/deny": "/deny",
        "help": "/help",
        "/help": "/help",
        "commands": "/commands",
        "/commands": "/commands",
        "command": "/commands",
        "/command": "/commands",
        "command list": "/commands",
        "list command": "/commands",
        "list commands": "/commands",
        "doctor": "/doctor",
        "doktor": "/doctor",
        "/doctor": "/doctor",
        "/doktor": "/doctor",
        "state": "/state",
        "/state": "/state",
        "open state": "/open-state",
        "/open state": "/open-state",
        "open state folder": "/open-state",
        "open zordon state": "/open-state",
        "open drafts": "/open-drafts",
        "/open drafts": "/open-drafts",
        "open draft folder": "/open-drafts",
        "open drafts folder": "/open-drafts",
        "open images": "/open-images",
        "/open images": "/open-images",
        "open image folder": "/open-images",
        "open images folder": "/open-images",
        "open logs": "/open-logs",
        "/open logs": "/open-logs",
        "open log folder": "/open-logs",
        "open logs folder": "/open-logs",
        "open memory": "/open-memory",
        "/open memory": "/open-memory",
        "open memory file": "/open-memory",
        "today": "/today",
        "/today": "/today",
        "today summary": "/today",
        "daily summary": "/today",
        "ready": "/ready",
        "/ready": "/ready",
        "recent": "/recent",
        "/recent": "/recent",
        "recent turns": "/recent",
        "recent activity": "/recent",
        "project": "/project",
        "/project": "/project",
        "current project": "/project",
        "project status": "/project-status",
        "/project status": "/project-status",
        "status project": "/project-status",
        "projects": "/projects",
        "/projects": "/projects",
        "flash projects": "/projects",
        "list projects": "/projects",
        "open project": "/open-project",
        "/open project": "/open-project",
        "project drafts": "/project-drafts",
        "/project drafts": "/project-drafts",
        "list project drafts": "/project-drafts",
        "project images": "/project-images",
        "/project images": "/project-images",
        "list project images": "/project-images",
        "open project drafts": "/open-project-drafts",
        "/open project drafts": "/open-project-drafts",
        "open project draft folder": "/open-project-drafts",
        "open project images": "/open-project-images",
        "/open project images": "/open-project-images",
        "open project image folder": "/open-project-images",
        "save latest draft to project": "/save-latest-draft-to-project",
        "/save latest draft to project": "/save-latest-draft-to-project",
        "save latest image to project": "/save-latest-image-to-project",
        "/save latest image to project": "/save-latest-image-to-project",
        "open latest project draft": "/open-latest-project-draft",
        "/open latest project draft": "/open-latest-project-draft",
        "open latest project image": "/open-latest-project-image",
        "/open latest project image": "/open-latest-project-image",
        "project brief": "/project-brief",
        "/project brief": "/project-brief",
        "open project brief": "/open-project-brief",
        "/open project brief": "/open-project-brief",
        "project brief next": "/project-brief-next",
        "/project brief next": "/project-brief-next",
        "brief next": "/project-brief-next",
        "next project brief": "/project-brief-next",
        "content plan": "/content-plan",
        "/content plan": "/content-plan",
        "make content plan": "/content-plan",
        "create content plan": "/content-plan",
        "project content plan": "/content-plan",
        "save content plan": "/save-content-plan",
        "/save content plan": "/save-content-plan",
        "open content plan": "/open-content-plan",
        "/open content plan": "/open-content-plan",
        "project next": "/project-next",
        "/project next": "/project-next",
        "next project": "/project-next",
        "what's next project": "/project-next",
        "whats next project": "/project-next",
        "pulse": "/pulse",
        "/pulse": "/pulse",
        "drafts": "/drafts",
        "/drafts": "/drafts",
        "list drafts": "/drafts",
        "latest draft": "/latest-draft",
        "/latest draft": "/latest-draft",
        "/latest drafts": "/latest-draft",
        "latest drafts": "/latest-draft",
        "read latest draft": "/latest-draft",
        "read latest drafts": "/latest-draft",
        "read my latest draft": "/latest-draft",
        "show latest draft": "/latest-draft",
        "show latest drafts": "/latest-draft",
        "show my latest draft": "/latest-draft",
        "open latest draft": "/open-latest-draft",
        "open latest drafts": "/open-latest-draft",
        "open my latest draft": "/open-latest-draft",
        "/open latest draft": "/open-latest-draft",
        "/open latest drafts": "/open-latest-draft",
        "images": "/images",
        "/images": "/images",
        "image": "/images",
        "/image": "/images",
        "list images": "/images",
        "show images": "/images",
        "latest image": "/latest-image",
        "/latest image": "/latest-image",
        "/latest images": "/latest-image",
        "latest images": "/latest-image",
        "show latest image": "/latest-image",
        "show latest images": "/latest-image",
        "open latest image": "/open-latest-image",
        "open latest images": "/open-latest-image",
        "open my latest image": "/open-latest-image",
        "/open latest image": "/open-latest-image",
        "/open latest images": "/open-latest-image",
        "where is the image": "/latest-image",
        "memory": "/memory",
        "/memory": "/memory",
        "memories": "/memory",
        "what do you remember": "/memory",
        "logs": "/logs",
        "/logs": "/logs",
        "exit": "exit",
        "/exit": "/exit",
        "quit": "quit",
        "/quit": "/quit",
    }
    if lowered in simple_commands:
        return simple_commands[lowered]
    if _looks_like_list_drafts(lowered):
        return "/drafts"
    if _looks_like_latest_draft(lowered):
        return "/latest-draft"
    if lowered.startswith("/"):
        return _normalize_unknown_slash_command(lowered)
    return text


def _dynamic_spoken_command(text: str) -> str | None:
    for prefix, command in [
        ("/new project ", "/new-project"),
        ("new project ", "/new-project"),
        ("create project ", "/new-project"),
        ("/use project ", "/use-project"),
        ("use project ", "/use-project"),
        ("switch project ", "/use-project"),
        ("/set project brief ", "/set-project-brief"),
        ("set project brief ", "/set-project-brief"),
        ("project brief is ", "/set-project-brief"),
    ]:
        if text.startswith(prefix):
            name = text.removeprefix(prefix).strip()
            if name:
                return f"{command} {name}"
    return None


def is_command_candidate(text: str) -> bool:
    return text.strip().startswith("/")


def _split_compact_slash_command(text: str) -> str:
    compact_aliases = {
        "slashconfirm": "slash confirm",
        "slashcancel": "slash cancel",
        "slashdeny": "slash deny",
        "slashhelp": "slash help",
        "slashcommands": "slash commands",
        "slashdoctor": "slash doctor",
        "slashdoktor": "slash doktor",
        "slashstate": "slash state",
        "slashopenstate": "slash open state",
        "slashopendrafts": "slash open drafts",
        "slashopenimages": "slash open images",
        "slashopenlogs": "slash open logs",
        "slashopenmemory": "slash open memory",
        "slashopenlatestdraft": "slash open latest draft",
        "slashtoday": "slash today",
        "slashready": "slash ready",
        "slashrecent": "slash recent",
        "slashproject": "slash project",
        "slashprojectstatus": "slash project status",
        "slashprojects": "slash projects",
        "flashprojects": "slash projects",
        "slashopenproject": "slash open project",
        "slashprojectdrafts": "slash project drafts",
        "slashprojectimages": "slash project images",
        "slashopenprojectdrafts": "slash open project drafts",
        "slashopenprojectimages": "slash open project images",
        "slashsavelatestdrafttoproject": "slash save latest draft to project",
        "slashsavelatestimagetoproject": "slash save latest image to project",
        "slashopenlatestprojectdraft": "slash open latest project draft",
        "slashopenlatestprojectimage": "slash open latest project image",
        "slashprojectbrief": "slash project brief",
        "slashopenprojectbrief": "slash open project brief",
        "slashprojectbriefnext": "slash project brief next",
        "slashcontentplan": "slash content plan",
        "slashsavecontentplan": "slash save content plan",
        "slashopencontentplan": "slash open content plan",
        "slashprojectnext": "slash project next",
        "slashpulse": "slash pulse",
        "slashdraft": "slash draft",
        "slashdrafts": "slash drafts",
        "slashimages": "slash images",
        "slashimage": "slash image",
        "slashmemory": "slash memory",
        "slashlogs": "slash logs",
        "slashexit": "slash exit",
        "slashquit": "slash quit",
    }
    return compact_aliases.get(text, text)


def _normalize_unknown_slash_command(text: str) -> str:
    command = text.strip().removeprefix("/").strip()
    command = re.sub(r"[^a-z0-9]+", "-", command).strip("-")
    return f"/{command}" if command else "/"


def _looks_like_list_drafts(text: str) -> bool:
    draft_words = {"draft", "drafts", "drafs", "draftes", "drachs", "drachts", "giraffes"}
    if text in draft_words:
        return True
    if text.startswith("list "):
        requested = text.removeprefix("list ").strip()
        return requested in draft_words
    if text.startswith("show "):
        requested = text.removeprefix("show ").strip()
        return requested in draft_words
    return False


def _looks_like_latest_draft(text: str) -> bool:
    draft_words = {"draft", "drafts", "drafs", "draftes", "drachs", "drachts"}
    prefixes = {"read", "show", "open", "latest", "read my", "show my", "open my"}
    for prefix in prefixes:
        if text.startswith(prefix + " "):
            requested = text.removeprefix(prefix).strip()
            return requested.startswith("latest ") and requested.removeprefix("latest ").strip() in draft_words
    return False


if __name__ == "__main__":
    raise SystemExit(main())
