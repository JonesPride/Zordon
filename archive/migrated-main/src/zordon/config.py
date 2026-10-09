from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_HISTORY_MESSAGE_LIMIT = 40
DEFAULT_OUTPUT_TOKEN_LIMIT = 2048
ReasoningEffort = Literal["low", "medium", "high", "xhigh"]
DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"
_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})


class ConfigurationError(ValueError):
    """Raised when Zordon cannot start safely from its configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    history_message_limit: int = DEFAULT_HISTORY_MESSAGE_LIMIT
    output_token_limit: int = DEFAULT_OUTPUT_TOKEN_LIMIT
    reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT
    debug_log_path: Path | None = None


def _bounded_integer(
    environ: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw_value = environ.get(name, str(default)).strip() or str(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an integer between {minimum} and {maximum}."
        ) from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return value


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    if environ is None:
        load_dotenv()
        environ = os.environ

    api_key = environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. On Windows PowerShell, get a key from "
            "https://platform.openai.com/api-keys, run "
            "if (-not (Test-Path .env)) { Copy-Item .env.example .env }, then "
            "open .env in a text editor and "
            "replace the placeholder with the key. Paste the key only into the "
            "file to keep it out of PowerShell command history."
        )

    model = environ.get("ZORDON_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    raw_timeout = environ.get(
        "ZORDON_REQUEST_TIMEOUT_SECONDS",
        str(DEFAULT_TIMEOUT_SECONDS),
    ).strip() or str(DEFAULT_TIMEOUT_SECONDS)

    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ConfigurationError(
            "ZORDON_REQUEST_TIMEOUT_SECONDS must be a positive number."
        ) from exc

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConfigurationError("ZORDON_REQUEST_TIMEOUT_SECONDS must be a positive number.")

    history_message_limit = _bounded_integer(
        environ, "ZORDON_HISTORY_MESSAGE_LIMIT", DEFAULT_HISTORY_MESSAGE_LIMIT, 2, 200
    )
    if history_message_limit % 2:
        raise ConfigurationError(
            "ZORDON_HISTORY_MESSAGE_LIMIT must be an even integer between 2 and 200."
        )
    output_token_limit = _bounded_integer(
        environ, "ZORDON_OUTPUT_TOKEN_LIMIT", DEFAULT_OUTPUT_TOKEN_LIMIT, 1, 100_000
    )
    raw_reasoning_effort = (
        environ.get("ZORDON_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip().lower()
        or DEFAULT_REASONING_EFFORT
    )
    if raw_reasoning_effort not in _REASONING_EFFORTS:
        choices = ", ".join(sorted(_REASONING_EFFORTS))
        raise ConfigurationError(f"ZORDON_REASONING_EFFORT must be one of: {choices}.")
    raw_debug_log = environ.get("ZORDON_DEBUG_LOG", "").strip()

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        history_message_limit=history_message_limit,
        output_token_limit=output_token_limit,
        reasoning_effort=cast(ReasoningEffort, raw_reasoning_effort),
        debug_log_path=Path(raw_debug_log).expanduser() if raw_debug_log else None,
    )


# Compatibility surface for migrated voice-agent modules.
# Keep this additive so the newer Settings/load_settings API continues to work.

import json
import tempfile

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Config:
    assistant_name: str
    model: str
    transcription_model: str
    tts_model: str
    tts_voice: str
    push_to_talk_key: str
    audio_sample_rate: int
    temperature: float
    request_timeout_seconds: int
    image_model: str = "gpt-image-1"
    image_size: str = "1024x1024"
    image_quality: str = "low"


def state_root() -> Path:
    configured = os.getenv("ZORDON_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    for candidate in state_root_candidates():
        if _can_write_to(candidate):
            return candidate

    return Path(tempfile.gettempdir()) / "Zordon"


def state_root_candidates() -> list[Path]:
    candidates: list[Path] = []

    appdata = os.getenv("APPDATA", "").strip()
    if appdata:
        candidates.append(Path(appdata) / "Zordon")

    local_appdata = os.getenv("LOCALAPPDATA", "").strip()
    if local_appdata:
        candidates.append(Path(local_appdata) / "Zordon")

    candidates.append(ROOT)
    return candidates


def is_temp_state_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(tempfile.gettempdir()).resolve())
        return True
    except ValueError:
        return False


def _can_write_to(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".zordon-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def load_config(path: Path = ROOT / "config.json") -> Config:
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

    settings = None
    try:
        settings = load_settings()
    except ConfigurationError:
        pass

    return Config(
        assistant_name=data.get("assistant_name", "Zordon"),
        model=data.get("model", settings.model if settings else DEFAULT_MODEL),
        image_model=data.get("image_model", "gpt-image-1"),
        image_size=data.get("image_size", "1024x1024"),
        image_quality=data.get("image_quality", "low"),
        transcription_model=data.get("transcription_model", "gpt-4o-transcribe"),
        tts_model=data.get("tts_model", "gpt-4o-mini-tts"),
        tts_voice=data.get("tts_voice", "verse"),
        push_to_talk_key=data.get("push_to_talk_key", "space"),
        audio_sample_rate=int(data.get("audio_sample_rate", 16000)),
        temperature=float(data.get("temperature", 0.7)),
        request_timeout_seconds=int(
            data.get(
                "request_timeout_seconds",
                settings.timeout_seconds if settings else DEFAULT_TIMEOUT_SECONDS,
            )
        ),
    )
