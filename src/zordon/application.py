from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zordon.agent import Agent
from zordon.audio import AudioCatalog
from zordon.config import Settings
from zordon.documents import DocumentService
from zordon.paths import ApprovedFolders
from zordon.playback import BackendFactory, PlaybackController
from zordon.providers.base import ModelProvider
from zordon.providers.openai_provider import OpenAIProvider
from zordon.tools.builtin import build_builtin_registry, build_system_prompt
from zordon.tools.registry import ToolRegistry

ProviderFactory = Callable[[Settings], ModelProvider]


class ApplicationError(RuntimeError):
    """A composition failure safe to report without internal details."""


@dataclass(slots=True)
class Application:
    agent: Agent
    playback: PlaybackController
    registry: ToolRegistry
    audio_catalog: AudioCatalog
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.playback.close()


def build_application(
    settings: Settings,
    *,
    provider_factory: ProviderFactory | None = None,
    backend_factory: BackendFactory | None = None,
) -> Application:
    folders = ApprovedFolders(settings.approved_roots)
    documents = DocumentService(folders, settings.tier2_limits)
    audio = AudioCatalog(folders, settings.tier2_limits)
    playback = PlaybackController(audio, backend_factory=backend_factory)
    try:
        registry = build_builtin_registry(documents, audio, playback, settings.tier2_limits)
        provider = (
            provider_factory(settings)
            if provider_factory is not None
            else OpenAIProvider(
                api_key=settings.api_key,
                model=settings.model,
                timeout_seconds=settings.timeout_seconds,
            )
        )
        agent = Agent(provider, registry, build_system_prompt(registry))
        return Application(agent, playback, registry, audio)
    except Exception as exc:
        playback.close()
        raise ApplicationError("Zordon could not initialize its local services.") from exc
