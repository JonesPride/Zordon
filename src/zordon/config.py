from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_TIMEOUT_SECONDS = 60.0


class ConfigurationError(ValueError):
    """Raised when Zordon cannot start safely from its configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


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
    raw_timeout = (
        environ.get(
            "ZORDON_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_TIMEOUT_SECONDS),
        ).strip()
        or str(DEFAULT_TIMEOUT_SECONDS)
    )

    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ConfigurationError(
            "ZORDON_REQUEST_TIMEOUT_SECONDS must be a positive number."
        ) from exc

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConfigurationError(
            "ZORDON_REQUEST_TIMEOUT_SECONDS must be a positive number."
        )

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
