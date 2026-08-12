from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from zordon.application import ApplicationError, build_application
from zordon.config import Settings
from zordon.messages import ModelItem
from zordon.playback import PlaybackUnavailable
from zordon.providers.base import ProviderEvent, ResponseCompleted, TextDelta
from zordon.tools import Tool


class FakeProvider:
    def __init__(self) -> None:
        self.output_limits: list[int] = []

    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
        max_output_tokens: int = 2048,
    ) -> Iterator[ProviderEvent]:
        del system_prompt, items, tools
        self.output_limits.append(max_output_tokens)
        yield TextDelta("ready")
        yield ResponseCompleted()


class RecordingBackend:
    def __init__(self) -> None:
        self.close_calls = 0

    def play(self, path):
        del path

    def pause(self):
        pass

    def resume(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.close_calls += 1


def settings() -> Settings:
    return Settings(api_key="test")


def test_build_starts_without_folders_or_vlc_and_non_playback_works() -> None:
    backend_calls = 0

    def unavailable_backend():
        nonlocal backend_calls
        backend_calls += 1
        raise PlaybackUnavailable

    app = build_application(
        settings(),
        provider_factory=lambda _: FakeProvider(),
        backend_factory=unavailable_backend,
    )

    assert list(app.agent.stream_turn("hello")) == ["ready"]
    assert backend_calls == 0
    assert len(app.registry.list_tools()) == 10
    app.close()


def test_application_owns_one_shared_catalog_and_controller() -> None:
    app = build_application(settings(), provider_factory=lambda _: FakeProvider())

    assert app.playback._catalog is app.audio_catalog
    app.close()


def test_build_wires_hardened_provider_and_agent_controls(monkeypatch) -> None:
    provider = FakeProvider()
    provider_arguments: dict[str, object] = {}

    def build_provider(**kwargs):
        provider_arguments.update(kwargs)
        return provider

    monkeypatch.setattr("zordon.application.OpenAIProvider", build_provider)
    app = build_application(
        Settings(
            api_key="test",
            history_message_limit=6,
            output_token_limit=777,
            reasoning_effort="high",
        )
    )

    list(app.agent.stream_turn("hello"))

    assert provider_arguments["reasoning_effort"] == "high"
    assert provider.output_limits == [777]
    app.close()


def test_application_close_is_idempotent() -> None:
    backend = RecordingBackend()
    app = build_application(
        settings(),
        provider_factory=lambda _: FakeProvider(),
        backend_factory=lambda: backend,
    )
    app.playback._backend = backend

    app.close()
    app.close()

    assert backend.close_calls == 1


def test_build_closes_controller_when_later_composition_fails(monkeypatch) -> None:
    closed: list[bool] = []

    def fail_registry(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("composition failed")

    monkeypatch.setattr("zordon.application.build_builtin_registry", fail_registry)
    monkeypatch.setattr(
        "zordon.application.PlaybackController.close", lambda self: closed.append(True)
    )

    with pytest.raises(ApplicationError, match="could not initialize"):
        build_application(settings(), provider_factory=lambda _: FakeProvider())

    assert closed == [True]
