from collections.abc import Iterator, Sequence

import pytest

from zordon.agent import SYSTEM_PROMPT, Agent
from zordon.messages import Message
from zordon.providers.base import ProviderError


class RecordingProvider:
    def __init__(self, replies: list[list[str]]) -> None:
        self._replies = iter(replies)
        self.calls: list[tuple[str, tuple[Message, ...]]] = []

    def stream_reply(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> Iterator[str]:
        self.calls.append((system_prompt, tuple(messages)))
        yield from next(self._replies)


class FailingProvider:
    def stream_reply(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> Iterator[str]:
        del system_prompt, messages
        yield "unfinished"
        raise ProviderError("The model connection was interrupted.")


def test_system_prompt_defines_zordon_and_current_limits() -> None:
    lowered = SYSTEM_PROMPT.lower()

    assert "zordon" in lowered
    assert "cool mentor" in lowered
    assert "calm" in lowered
    assert "text conversation" in lowered
    assert "do not claim" in lowered


def test_successful_turn_streams_chunks_and_commits_both_messages() -> None:
    provider = RecordingProvider([["Calm", " and ready."]])
    agent = Agent(provider)

    chunks = list(agent.stream_turn("  Hello Zordon  "))

    assert chunks == ["Calm", " and ready."]
    assert agent.history == (
        Message(role="user", content="Hello Zordon"),
        Message(role="assistant", content="Calm and ready."),
    )


def test_later_turn_sends_prior_successful_history_to_provider() -> None:
    provider = RecordingProvider(
        [
            ["Blue noted."],
            ["Your temporary favorite color is blue."],
        ]
    )
    agent = Agent(provider)

    list(agent.stream_turn("My temporary favorite color is blue."))
    list(agent.stream_turn("What color did I tell you?"))

    _, second_messages = provider.calls[1]
    assert second_messages == (
        Message(role="user", content="My temporary favorite color is blue."),
        Message(role="assistant", content="Blue noted."),
        Message(role="user", content="What color did I tell you?"),
    )


def test_failed_stream_does_not_commit_partial_turn() -> None:
    agent = Agent(FailingProvider())

    stream = agent.stream_turn("Keep this turn clean")
    assert next(stream) == "unfinished"
    with pytest.raises(ProviderError, match="interrupted"):
        next(stream)

    assert agent.history == ()


def test_empty_model_reply_is_a_provider_error_and_is_not_committed() -> None:
    agent = Agent(RecordingProvider([[]]))

    with pytest.raises(ProviderError, match="no text"):
        list(agent.stream_turn("Are you there?"))

    assert agent.history == ()
