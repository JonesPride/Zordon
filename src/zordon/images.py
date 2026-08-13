from __future__ import annotations

import base64
import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import Config, ROOT, state_root
from .openai_provider import ProviderError


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    revised_prompt: str | None = None


def images_dir(root: Path) -> Path:
    base = state_root() if root == ROOT else root
    return base / "images"


def list_images(root: Path) -> list[str]:
    directory = images_dir(root)
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def latest_image(root: Path) -> Path:
    directory = images_dir(root)
    if not directory.exists():
        raise FileNotFoundError("No images saved yet.")
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not images:
        raise FileNotFoundError("No images saved yet.")
    return max(images, key=lambda path: path.stat().st_mtime)


def open_image(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No image found at {path}.")
    os.startfile(str(path))


def generate_image(
    config: Config,
    root: Path,
    prompt: str,
    filename: str | None = None,
    output_dir: Path | None = None,
) -> GeneratedImage:
    prompt = _required_text(prompt, "prompt")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("OPENAI_API_KEY is not set. Check .env.local.")

    payload = {
        "model": config.image_model,
        "prompt": prompt,
        "size": config.image_size,
        "quality": config.image_quality,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Image generation failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise ProviderError(f"Image generation service is unreachable: {exc}") from exc

    item = data.get("data", [{}])[0]
    b64_json = item.get("b64_json")
    if not isinstance(b64_json, str) or not b64_json:
        raise ProviderError("Image generation response did not include image data.")

    directory = output_dir or images_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_image_filename(filename or _default_filename(prompt))
    path = directory / safe_name
    path.write_bytes(base64.b64decode(b64_json))
    return GeneratedImage(path=path, revised_prompt=item.get("revised_prompt"))


def _default_filename(prompt: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", prompt.lower()).strip("-")[:48]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{slug or 'image'}.png"


def _safe_image_filename(filename: str) -> str:
    filename = _required_text(filename, "filename")
    if "/" in filename or "\\" in filename:
        raise ValueError("filename cannot include path separators")
    if not filename.lower().endswith(".png"):
        filename += ".png"
    if not re.match(r"^[A-Za-z0-9._ -]+$", filename):
        raise ValueError("filename can only contain letters, numbers, spaces, dots, underscores, and hyphens")
    return filename


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
