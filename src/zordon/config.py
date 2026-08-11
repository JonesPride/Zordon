from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_TIMEOUT_SECONDS = 60.0
_FOLDER_KEY_PREFIX = "ZORDON_FOLDER_"
_FOLDER_SUFFIX_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,31}")


class ConfigurationError(ValueError):
    """Raised when Zordon cannot start safely from its configuration."""


@dataclass(frozen=True, slots=True)
class Tier2Limits:
    document_scan: int = 5_000
    audio_scan: int = 20_000
    search_results: int = 10
    document_read_chars: int = 12_000


@dataclass(frozen=True, slots=True)
class ApprovedRoot:
    folder_id: str
    path: Path = field(repr=False)


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    approved_roots: tuple[ApprovedRoot, ...] = ()
    tier2_limits: Tier2Limits = Tier2Limits()


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

    approved_roots = _parse_approved_roots(environ)
    tier2_limits = Tier2Limits(
        document_scan=_parse_bounded_integer(
            environ, "ZORDON_DOCUMENT_SCAN_LIMIT", 5_000, 100, 50_000
        ),
        audio_scan=_parse_bounded_integer(
            environ, "ZORDON_AUDIO_SCAN_LIMIT", 20_000, 100, 100_000
        ),
        search_results=_parse_bounded_integer(
            environ, "ZORDON_SEARCH_RESULT_LIMIT", 10, 1, 25
        ),
        document_read_chars=_parse_bounded_integer(
            environ, "ZORDON_DOCUMENT_READ_CHARS", 12_000, 1_000, 20_000
        ),
    )

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        approved_roots=approved_roots,
        tier2_limits=tier2_limits,
    )


def _parse_approved_roots(environ: Mapping[str, str]) -> tuple[ApprovedRoot, ...]:
    roots: list[ApprovedRoot] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for key, raw_path in sorted(environ.items(), key=lambda item: item[0].casefold()):
        if not key.startswith(_FOLDER_KEY_PREFIX):
            continue
        suffix = key.removeprefix(_FOLDER_KEY_PREFIX)
        if _FOLDER_SUFFIX_PATTERN.fullmatch(suffix) is None:
            raise ConfigurationError(
                f"{key} is not a valid approved folder setting. Use "
                "ZORDON_FOLDER_ followed by an uppercase letter, then uppercase "
                "letters, digits, or underscores."
            )
        folder_id = suffix.lower()
        if folder_id in seen_ids:
            raise ConfigurationError(f"Duplicate approved folder ID '{folder_id}'.")
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError(
                f"The approved folder '{folder_id}' does not exist or cannot be resolved."
            ) from exc
        if not path.is_dir():
            raise ConfigurationError(
                f"The approved folder '{folder_id}' must refer to a directory."
            )
        canonical = os.path.normcase(str(path))
        if canonical in seen_paths:
            raise ConfigurationError(
                f"Multiple settings refer to the same approved folder ('{folder_id}')."
            )
        seen_ids.add(folder_id)
        seen_paths.add(canonical)
        roots.append(ApprovedRoot(folder_id, path))
    return tuple(roots)


def _parse_bounded_integer(
    environ: Mapping[str, str],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = environ.get(key, str(default)).strip() or str(default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{key} must be an integer from {minimum} through {maximum}."
        ) from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{key} must be an integer from {minimum} through {maximum}."
        )
    return value
