from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from zordon.config import ReasoningEffort
from zordon.messages import Message
from zordon.providers.base import ProviderError

_VISIBLE_DELTA_EVENTS = {
    "response.output_text.delta",
    "response.refusal.delta",
}
_FAILURE_EVENTS = {
    "error",
    "response.failed",
    "response.incomplete",
}


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        reasoning_effort: ReasoningEffort = "medium",
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    def stream_reply(
        self,
        system_prompt: str,
        messages: Sequence[Message],
        max_output_tokens: int,
    ) -> Iterator[str]:
        payload = [{"role": message.role, "content": message.content} for message in messages]

        try:
            stream = self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=payload,
                stream=True,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": self._reasoning_effort},
            )
            completed = False
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type in _VISIBLE_DELTA_EVENTS:
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield delta
                elif event_type == "response.completed":
                    completed = True
                elif event_type in _FAILURE_EVENTS:
                    raise ProviderError(
                        "The model response was incomplete or failed before it completed."
                    )
            if not completed:
                raise ProviderError(
                    "The model response was incomplete or failed before it completed."
                )
        except ProviderError:
            raise
        except AuthenticationError as exc:
            raise ProviderError(
                "OpenAI rejected the API key. Run the secure key setup again."
            ) from exc
        except RateLimitError as exc:
            raise ProviderError(
                "The model is temporarily rate-limited. Wait a moment and retry."
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise ProviderError(
                "The model could not be reached. Check the connection and retry."
            ) from exc
        except APIError as exc:
            raise ProviderError("The model provider returned an error. Please retry.") from exc
        except Exception as exc:
            raise ProviderError(
                "An unexpected model-provider error occurred. Please retry."
            ) from exc
