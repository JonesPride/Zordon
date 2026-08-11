from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

from zordon.audio import AudioCatalog, AudioEntry
from zordon.tools.base import ToolResult


class PlaybackUnavailable(RuntimeError):
    """Raised when the local playback runtime cannot be initialized."""


class PlaybackBackend(Protocol):
    def play(self, path: Path) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


BackendFactory = Callable[[], PlaybackBackend]


class PlaybackController:
    """Manage one local audio player without exposing private file paths."""

    def __init__(
        self,
        catalog: AudioCatalog,
        *,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self._catalog = catalog
        self._backend_factory = backend_factory or VLCBackend
        self._backend: PlaybackBackend | None = None
        self._current: AudioEntry | None = None
        self._state = "stopped"
        self._closed = False

    def play(self, audio_id: str, turn_number: int) -> ToolResult:
        if self._closed:
            return _unavailable()
        resolved = self._catalog.resolve_for_playback(audio_id, turn_number)
        if isinstance(resolved, ToolResult):
            return resolved
        try:
            backend = self._get_backend()
        except PlaybackUnavailable:
            return _unavailable()
        except Exception:  # noqa: BLE001 - backend details are private
            return _failed()

        if self._state != "stopped":
            try:
                backend.stop()
            except Exception:  # noqa: BLE001 - backend details are private
                return _failed()
            self._state = "stopped"
            self._current = None
        try:
            backend.play(resolved.path)
        except Exception:  # noqa: BLE001 - backend details are private
            self._state = "stopped"
            self._current = None
            return _failed()
        self._current = resolved
        self._state = "playing"
        return ToolResult.success("Audio playback started.", self._status_data())

    def pause(self) -> ToolResult:
        if self._state != "playing":
            return ToolResult.success("Playback state unchanged.", self._status_data())
        try:
            assert self._backend is not None
            self._backend.pause()
        except Exception:  # noqa: BLE001 - backend details are private
            return _failed()
        self._state = "paused"
        return ToolResult.success("Audio playback paused.", self._status_data())

    def resume(self) -> ToolResult:
        if self._state != "paused":
            return ToolResult.success("Playback state unchanged.", self._status_data())
        try:
            assert self._backend is not None
            self._backend.resume()
        except Exception:  # noqa: BLE001 - backend details are private
            return _failed()
        self._state = "playing"
        return ToolResult.success("Audio playback resumed.", self._status_data())

    def stop(self) -> ToolResult:
        if self._state == "stopped":
            return ToolResult.success("Playback is stopped.", self._status_data())
        try:
            assert self._backend is not None
            self._backend.stop()
        except Exception:  # noqa: BLE001 - backend details are private
            return _failed()
        self._state = "stopped"
        self._current = None
        return ToolResult.success("Audio playback stopped.", self._status_data())

    def now_playing(self) -> ToolResult:
        return ToolResult.success("Playback status retrieved.", self._status_data())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        backend = self._backend
        if backend is not None:
            if self._state != "stopped":
                with suppress(Exception):
                    backend.stop()
            with suppress(Exception):
                backend.close()
        self._state = "stopped"
        self._current = None

    def _get_backend(self) -> PlaybackBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def _status_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"state": self._state}
        if self._current is not None:
            data["audio"] = {
                "audio_id": self._current.audio_id,
                "folder_id": self._current.folder_id,
                "relative_path": self._current.relative_path,
                **self._current.metadata,
                "untrusted_data": True,
            }
        return data


class VLCBackend:
    """A lazily imported VLC backend with one instance and media player."""

    def __init__(self) -> None:
        try:
            import vlc  # type: ignore[import-not-found]

            self._instance = vlc.Instance()
            self._player = self._instance.media_player_new()
        except Exception as exc:
            raise PlaybackUnavailable(
                "VLC 3.x and the matching python-vlc runtime are required."
            ) from exc
        self._closed = False

    def play(self, path: Path) -> None:
        media = self._instance.media_new_path(str(path))
        self._player.set_media(media)
        if self._player.play() == -1:
            raise RuntimeError("VLC could not start playback.")

    def pause(self) -> None:
        self._player.pause()

    def resume(self) -> None:
        if self._player.play() == -1:
            raise RuntimeError("VLC could not resume playback.")

    def stop(self) -> None:
        self._player.stop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._player.stop()
        finally:
            try:
                self._player.release()
            finally:
                self._instance.release()


def _unavailable() -> ToolResult:
    return ToolResult.failure(
        "playback_unavailable",
        "Playback is unavailable. Install 64-bit VLC 3.x and restart Zordon.",
    )


def _failed() -> ToolResult:
    return ToolResult.failure(
        "playback_failed",
        "The playback command could not be completed.",
    )
