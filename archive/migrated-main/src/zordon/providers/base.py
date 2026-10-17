from collections.abc import Iterator, Sequence
from typing import Protocol

from zordon.messages import Message


class ProviderError(RuntimeError):
    """A model-provider failure that can be safely explained to the user."""


class ModelProvider(Protocol):
    def stream_reply(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        max_output_tokens: int,
    ) -> Iterator[str]:
        """Yield user-visible text chunks for one candidate conversation."""
        raise NotImplementedError
