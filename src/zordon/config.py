from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_HISTORY_MESSAGE_LIMIT = 40
DEFAULT_OUTPUT_TOKEN_LIMIT = 2048


class ConfigurationError(ValueError):
    """Raised when Zordon cannot start safely from its configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    history_message_limit: int = DEFAULT_HISTORY_MESSAGE_LIMIT
    output_token_limit: int = DEFAULT_OUTPUT_TOKEN_LIMIT
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
    raw_debug_log = environ.get("ZORDON_DEBUG_LOG", "").strip()

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        history_message_limit=history_message_limit,
        output_token_limit=output_token_limit,
        debug_log_path=Path(raw_debug_log).expanduser() if raw_debug_log else None,
    )
