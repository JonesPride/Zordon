from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from zordon.messages import JSONValue, ModelItem
from zordon.tools import Tool


class ProviderError(RuntimeError):
    """A model-provider failure that can be safely explained to the user."""


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ResponseCompleted:
    pass


ProviderEvent = TextDelta | ToolCall | ResponseCompleted


class ModelProvider(Protocol):
    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
        max_output_tokens: int = 2048,
    ) -> Iterator[ProviderEvent]:
        """Yield provider-neutral events for one complete model response."""
        raise NotImplementedError
