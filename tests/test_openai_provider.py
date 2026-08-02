from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from zordon.agent import Agent
from zordon.messages import Message
from zordon.providers.base import ProviderError
from zordon.providers.openai_provider import OpenAIProvider


class FakeResponses:
    def __init__(
        self,
        events: list[SimpleNamespace] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.events = events or []
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return iter(self.events)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def event(event_type: str, delta: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=event_type, delta=delta)


def test_adapter_maps_messages_and_yields_only_visible_text() -> None:
    responses = FakeResponses(
        [
            event("response.created"),
            event("response.output_text.delta", "Hello"),
            event("response.output_text.delta", " there"),
            event("response.refusal.delta", "."),
            event("response.completed"),
        ]
    )
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )

    chunks = list(
        provider.stream_reply(
            "System instructions",
            [
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello"),
                Message(role="user", content="Continue"),
            ],
        )
    )

    assert chunks == ["Hello", " there", "."]
    assert responses.calls == [
        {
            "model": "gpt-5.6-terra",
            "instructions": "System instructions",
            "input": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Continue"},
            ],
            "stream": True,
        }
    ]


def test_adapter_turns_failed_stream_event_into_provider_error() -> None:
    responses = FakeResponses([event("response.failed")])
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )

    with pytest.raises(ProviderError, match="failed"):
        list(provider.stream_reply("System", [Message("user", "Hi")]))


def test_adapter_turns_incomplete_stream_event_into_provider_error() -> None:
    responses = FakeResponses(
        [
            event("response.output_text.delta", "Partial reply"),
            event("response.incomplete"),
        ]
    )
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )

    stream = provider.stream_reply("System", [Message("user", "Hi")])

    assert next(stream) == "Partial reply"
    with pytest.raises(ProviderError, match="incomplete"):
        next(stream)


def test_stream_eof_without_completed_event_does_not_commit_agent_history() -> None:
    responses = FakeResponses(
        [event("response.output_text.delta", "Partial reply")]
    )
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )

    agent = Agent(provider)

    stream = agent.stream_turn("Hi")

    assert next(stream) == "Partial reply"
    with pytest.raises(ProviderError, match="before it completed"):
        next(stream)
    assert agent.history == ()


def test_incomplete_stream_does_not_commit_partial_turn_to_agent_history() -> None:
    responses = FakeResponses(
        [
            event("response.output_text.delta", "Partial reply"),
            event("response.incomplete"),
        ]
    )
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )
    agent = Agent(provider)

    stream = agent.stream_turn("Keep this turn clean")

    assert next(stream) == "Partial reply"
    with pytest.raises(ProviderError, match="incomplete"):
        next(stream)
    assert agent.history == ()


def test_adapter_translates_unexpected_sdk_failure() -> None:
    responses = FakeResponses(failure=RuntimeError("internal detail"))
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )

    with pytest.raises(ProviderError, match="unexpected model-provider"):
        list(provider.stream_reply("System", [Message("user", "Hi")]))


@pytest.mark.parametrize(
    ("sdk_error", "safe_message"),
    [
        (
            AuthenticationError(
                "invalid credential",
                response=httpx.Response(
                    401,
                    request=httpx.Request("POST", "/responses"),
                ),
                body=None,
            ),
            "rejected the API key",
        ),
        (
            RateLimitError(
                "too many requests",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "/responses"),
                ),
                body=None,
            ),
            "temporarily rate-limited",
        ),
        (
            APITimeoutError(
                httpx.Request("POST", "/responses")
            ),
            "could not be reached",
        ),
        (
            APIConnectionError(
                request=httpx.Request("POST", "/responses")
            ),
            "could not be reached",
        ),
        (
            APIError(
                "provider detail",
                httpx.Request("POST", "/responses"),
                body=None,
            ),
            "provider returned an error",
        ),
    ],
)
def test_adapter_maps_sdk_exceptions_to_safe_provider_errors(
    sdk_error: Exception, safe_message: str
) -> None:
    responses = FakeResponses(failure=sdk_error)
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-5.6-terra",
        timeout_seconds=12.0,
        client=FakeClient(responses),
    )

    with pytest.raises(ProviderError, match=safe_message) as error:
        list(provider.stream_reply("System", [Message("user", "Hi")]))

    assert error.value.__cause__ is sdk_error
