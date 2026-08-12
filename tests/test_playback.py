from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from zordon.audio import AudioCatalog
from zordon.config import ApprovedRoot, Tier2Limits
from zordon.paths import ApprovedFolders
from zordon.playback import PlaybackController, PlaybackUnavailable, VLCBackend


class RecordingBackend:
    def __init__(self, *, fail: str | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail = fail

    def _record(self, name: str, *arguments: Any) -> None:
        self.calls.append((name, *arguments))
        if self.fail == name:
            raise RuntimeError("private backend failure")

    def play(self, path: Path) -> None:
        self._record("play", path)

    def pause(self) -> None:
        self._record("pause")

    def resume(self) -> None:
        self._record("resume")

    def stop(self) -> None:
        self._record("stop")

    def close(self) -> None:
        self._record("close")


def make_catalog(tmp_path: Path, names: tuple[str, ...]) -> tuple[AudioCatalog, Path]:
    root = tmp_path / "private-music"
    root.mkdir()
    metadata: dict[str, Any] = {}
    for name in names:
        (root / name).write_bytes(b"audio")
        metadata[name] = SimpleNamespace(
            tags={"title": [Path(name).stem], "artist": ["Test Artist"]},
            info=SimpleNamespace(length=120.0),
        )
    catalog = AudioCatalog(
        ApprovedFolders((ApprovedRoot("music", root.resolve()),)),
        Tier2Limits(),
        metadata_loader=lambda path: metadata[path.name],
        id_factory=(f"audio-{index}" for index in range(1, 100)).__next__,
    )
    return catalog, root


def search_id(catalog: AudioCatalog, query: str, turn: int) -> str:
    result = catalog.search(query, turn_number=turn)
    return str(result.data["results"][0]["audio_id"])


def test_backend_is_created_lazily_after_valid_audio_resolution(tmp_path: Path) -> None:
    catalog, _ = make_catalog(tmp_path, ("Track.mp3",))
    backend = RecordingBackend()
    created: list[bool] = []
    controller = PlaybackController(
        catalog, backend_factory=lambda: (created.append(True), backend)[1]
    )

    assert controller.now_playing().data == {"state": "stopped"}
    assert controller.play("invalid", turn_number=1).code == "audio_id_invalid"
    assert created == []

    audio_id = search_id(catalog, "track", 1)
    assert controller.play(audio_id, turn_number=1).ok is True
    assert created == [True]


def test_same_turn_ambiguous_audio_never_reaches_backend(tmp_path: Path) -> None:
    catalog, _ = make_catalog(tmp_path, ("Mix One.mp3", "Mix Two.mp3"))
    audio_id = search_id(catalog, "mix", 3)
    backend = RecordingBackend()
    controller = PlaybackController(catalog, backend_factory=lambda: backend)

    result = controller.play(audio_id, turn_number=3)

    assert result.code == "audio_selection_required"
    assert backend.calls == []


def test_play_pause_resume_stop_transitions_and_now_playing_are_safe(
    tmp_path: Path,
) -> None:
    catalog, root = make_catalog(tmp_path, ("Track.mp3",))
    audio_id = search_id(catalog, "track", 1)
    backend = RecordingBackend()
    controller = PlaybackController(catalog, backend_factory=lambda: backend)

    assert controller.play(audio_id, turn_number=1).ok
    playing = controller.now_playing()
    assert playing.data["state"] == "playing"
    assert playing.data["audio"]["title"] == "Track"
    assert playing.data["audio"]["relative_path"] == "Track.mp3"
    assert str(root) not in str(playing.data)
    assert controller.pause().data["state"] == "paused"
    assert controller.resume().data["state"] == "playing"
    assert controller.stop().data == {"state": "stopped"}
    assert controller.now_playing().data == {"state": "stopped"}
    assert backend.calls == [
        ("play", root / "Track.mp3"),
        ("pause",),
        ("resume",),
        ("stop",),
    ]


def test_redundant_controls_are_successful_without_backend_calls(
    tmp_path: Path,
) -> None:
    catalog, _ = make_catalog(tmp_path, ("Track.mp3",))
    backend = RecordingBackend()
    controller = PlaybackController(catalog, backend_factory=lambda: backend)

    assert controller.pause().ok
    assert controller.resume().ok
    assert controller.stop().ok
    audio_id = search_id(catalog, "track", 1)
    controller.play(audio_id, turn_number=1)
    controller.pause()
    assert controller.pause().ok
    controller.resume()
    assert controller.resume().ok
    assert backend.calls == [
        ("play", tmp_path / "private-music" / "Track.mp3"),
        ("pause",),
        ("resume",),
    ]


def test_starting_new_audio_stops_current_audio_first(tmp_path: Path) -> None:
    catalog, root = make_catalog(tmp_path, ("One.mp3", "Two.mp3"))
    first = search_id(catalog, "one", 1)
    second = search_id(catalog, "two", 2)
    backend = RecordingBackend()
    controller = PlaybackController(catalog, backend_factory=lambda: backend)

    assert controller.play(first, turn_number=1).ok
    assert controller.play(second, turn_number=2).ok

    assert backend.calls == [
        ("play", root / "One.mp3"),
        ("stop",),
        ("play", root / "Two.mp3"),
    ]


@pytest.mark.parametrize("operation", ["play", "pause", "resume", "stop"])
def test_backend_failures_return_safe_playback_failed(tmp_path: Path, operation: str) -> None:
    catalog, root = make_catalog(tmp_path, ("Track.mp3",))
    audio_id = search_id(catalog, "track", 1)
    backend = RecordingBackend(fail=operation)
    controller = PlaybackController(catalog, backend_factory=lambda: backend)
    if operation != "play":
        backend.fail = None
        assert controller.play(audio_id, turn_number=1).ok
        if operation == "resume":
            assert controller.pause().ok
        backend.fail = operation

    result = getattr(controller, operation)(*(audio_id, 1) if operation == "play" else ())

    assert result.code == "playback_failed"
    assert "private" not in result.summary
    assert str(root) not in str(result.data)


def test_unavailable_backend_returns_actionable_safe_error(tmp_path: Path) -> None:
    catalog, _ = make_catalog(tmp_path, ("Track.mp3",))
    audio_id = search_id(catalog, "track", 1)
    controller = PlaybackController(
        catalog,
        backend_factory=lambda: (_ for _ in ()).throw(PlaybackUnavailable("private DLL detail")),
    )

    result = controller.play(audio_id, turn_number=1)

    assert result.code == "playback_unavailable"
    assert "VLC" in result.summary
    assert "private" not in result.summary
    assert controller.now_playing().data == {"state": "stopped"}


def test_close_stops_playback_closes_once_and_blocks_future_play(
    tmp_path: Path,
) -> None:
    catalog, _ = make_catalog(tmp_path, ("Track.mp3",))
    audio_id = search_id(catalog, "track", 1)
    backend = RecordingBackend()
    controller = PlaybackController(catalog, backend_factory=lambda: backend)
    controller.play(audio_id, turn_number=1)

    controller.close()
    controller.close()

    assert backend.calls[-2:] == [("stop",), ("close",)]
    assert controller.now_playing().data == {"state": "stopped"}
    assert controller.play(audio_id, turn_number=2).code == "playback_unavailable"


def test_close_without_play_does_not_create_backend(tmp_path: Path) -> None:
    catalog, _ = make_catalog(tmp_path, ("Track.mp3",))
    created: list[bool] = []
    controller = PlaybackController(
        catalog,
        backend_factory=lambda: created.append(True),  # type: ignore[arg-type,func-returns-value]
    )

    controller.close()

    assert created == []


def test_vlc_backend_import_is_lazy_and_missing_module_is_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_vlc(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vlc":
            raise ModuleNotFoundError("private missing module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_vlc)

    with pytest.raises(PlaybackUnavailable, match="VLC"):
        VLCBackend()


def test_vlc_backend_uses_one_instance_and_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class Player:
        def set_media(self, media: Any) -> None:
            calls.append(("set_media", media))

        def play(self) -> int:
            calls.append(("play",))
            return 0

        def pause(self) -> None:
            calls.append(("pause",))

        def stop(self) -> None:
            calls.append(("stop",))

        def release(self) -> None:
            calls.append(("player_release",))

    player = Player()

    class Instance:
        def media_player_new(self) -> Player:
            calls.append(("player_new",))
            return player

        def media_new_path(self, path: str) -> str:
            calls.append(("media", path))
            return "media"

        def release(self) -> None:
            calls.append(("instance_release",))

    fake_vlc = SimpleNamespace(Instance=lambda: Instance())
    monkeypatch.setitem(__import__("sys").modules, "vlc", fake_vlc)
    backend = VLCBackend()

    backend.play(Path("song.mp3"))
    backend.pause()
    backend.resume()
    backend.stop()
    backend.close()

    assert calls == [
        ("player_new",),
        ("media", "song.mp3"),
        ("set_media", "media"),
        ("play",),
        ("pause",),
        ("play",),
        ("stop",),
        ("stop",),
        ("player_release",),
        ("instance_release",),
    ]


def test_vlc_play_minus_one_is_a_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = SimpleNamespace(set_media=lambda media: None, play=lambda: -1)
    instance = SimpleNamespace(
        media_player_new=lambda: player,
        media_new_path=lambda path: object(),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "vlc", SimpleNamespace(Instance=lambda: instance)
    )

    backend = VLCBackend()

    with pytest.raises(RuntimeError):
        backend.play(Path("song.mp3"))
