from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .audio_io import play_wav
from .config import Config
from .openai_provider import ProviderError


class OpenAIAudioProvider:
    def __init__(self, config: Config):
        self.config = config

    def transcribe(self, wav_path: Path) -> str:
        api_key = _api_key()
        boundary = f"----zordon-{uuid.uuid4().hex}"
        body = _multipart_body(
            boundary,
            fields={"model": self.config.transcription_model},
            files={"file": (wav_path.name, "audio/wav", wav_path.read_bytes())},
        )
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Transcription failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"Transcription service is unreachable: {exc}") from exc

        return str(data.get("text", "")).strip()

    def speak(self, text: str, interrupt_key: str | None = None) -> tuple[Path, bool]:
        api_key = _api_key()
        payload = {
            "model": self.config.tts_model,
            "voice": self.config.tts_voice,
            "input": text,
            "response_format": "wav",
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Speech generation failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"Speech service is unreachable: {exc}") from exc

        path = Path(tempfile.gettempdir()) / "zordon-reply.wav"
        path.write_bytes(audio)
        interrupted = play_wav(path, interrupt_key=interrupt_key)
        return path, interrupted


def _api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("OPENAI_API_KEY is not set. Check .env.local.")
    return api_key


def _multipart_body(
    boundary: str, fields: dict[str, str], files: dict[str, tuple[str, str, bytes]]
) -> bytes:
    lines: list[bytes] = []
    for name, value in fields.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )
    for name, (filename, content_type, content) in files.items():
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines)
