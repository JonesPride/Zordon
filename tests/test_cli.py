from collections.abc import Iterator, Sequence
from io import StringIO

import pytest

from zordon.agent import Agent
from zordon.cli import run
from zordon.messages import ModelItem
from zordon.providers.base import (
    ProviderError,
    ProviderEvent,
    ResponseCompleted,
    TextDelta,
)
from zordon.tools import Tool


class ChunkProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
    ) -> Iterator[ProviderEvent]:
        del system_prompt, items, tools
        self.calls += 1
        yield TextDelta("Calm")
        yield TextDelta(" response")
        yield ResponseCompleted()


class FailThenRecoverProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
    ) -> Iterator[ProviderEvent]:
        del system_prompt, items, tools
        self.calls += 1
        if self.calls == 1:
            yield TextDelta("partial")
            raise ProviderError("Connection interrupted.")
        yield TextDelta("recovered")
        yield ResponseCompleted()


class InterruptedStreamProvider:
    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
    ) -> Iterator[ProviderEvent]:
        del system_prompt, items, tools
        yield TextDelta("partial")
        raise KeyboardInterrupt


class IncompleteReplyProvider:
    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
    ) -> Iterator[ProviderEvent]:
        del system_prompt, items, tools
        yield TextDelta("partial")
        raise ProviderError("The model response ended before response.completed.")


class RecordingOutput(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


class InterruptOnceOutput(RecordingOutput):
    def __init__(self, interrupted_text: str) -> None:
        super().__init__()
        self._interrupted_text = interrupted_text
        self._interrupted = False

    def write(self, text: str) -> int:
        if not self._interrupted and text == self._interrupted_text:
            self._interrupted = True
            raise KeyboardInterrupt
        return super().write(text)


class FakeApplication:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def inputs(*values: str):
    iterator = iter(values)

    def read(prompt: str) -> str:
        del prompt
        return next(iterator)

    return read


@pytest.mark.parametrize("ending", ["/exit", "exit", "quit"])
def test_cli_always_closes_application_after_exit(ending: str) -> None:
    app = FakeApplication(Agent(ChunkProvider()))

    assert run(app, input_fn=inputs(ending), output=RecordingOutput()) == 0
    assert app.close_calls == 1


def test_cli_closes_application_after_eof() -> None:
    app = FakeApplication(Agent(ChunkProvider()))

    def eof(prompt: str) -> str:
        del prompt
        raise EOFError

    assert run(app, input_fn=eof, output=RecordingOutput()) == 0
    assert app.close_calls == 1


def test_cli_closes_application_after_keyboard_interrupt() -> None:
    app = FakeApplication(Agent(ChunkProvider()))

    def interrupt(prompt: str) -> str:
        del prompt
        raise KeyboardInterrupt

    assert run(app, input_fn=interrupt, output=RecordingOutput()) == 0
    assert app.close_calls == 1


def test_cli_closes_after_provider_failure_then_exit() -> None:
    app = FakeApplication(Agent(IncompleteReplyProvider()))

    assert run(
        app, input_fn=inputs("hello", "exit"), output=RecordingOutput()
    ) == 0
    assert app.close_calls == 1


def test_cli_closes_application_when_output_fails() -> None:
    app = FakeApplication(Agent(ChunkProvider()))

    class BrokenOutput(RecordingOutput):
        def write(self, text: str) -> int:
            del text
            raise OSError("terminal unavailable")

    with pytest.raises(OSError, match="terminal unavailable"):
        run(app, output=BrokenOutput())

    assert app.close_calls == 1


def test_cli_ignores_blank_input_streams_chunks_and_exits() -> None:
    provider = ChunkProvider()
    output = RecordingOutput()

    exit_code = run(
        Agent(provider),
        input_fn=inputs("", "Hello", "/exit"),
        output=output,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert provider.calls == 1
    assert "Zordon online" in rendered
    assert "Zordon: Calm response" in rendered
    assert "Zordon offline" in rendered
    assert output.flush_count >= 3


def test_cli_labels_partial_failure_and_accepts_the_next_turn() -> None:
    provider = FailThenRecoverProvider()
    output = RecordingOutput()

    exit_code = run(
        Agent(provider),
        input_fn=inputs("First", "Second", "quit"),
        output=output,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "partial" in rendered
    assert "Reply incomplete: Connection interrupted." in rendered
    assert "Zordon: recovered" in rendered


def test_cli_labels_partial_reply_as_incomplete_and_keeps_history_clean() -> None:
    agent = Agent(IncompleteReplyProvider())
    output = RecordingOutput()

    exit_code = run(
        agent,
        input_fn=inputs("Hello", "/exit"),
        output=output,
    )

    assert exit_code == 0
    assert "Zordon: partial\n[Reply incomplete:" in output.getvalue()
    assert agent.history == ()


def test_cli_handles_keyboard_interrupt_without_traceback() -> None:
    output = RecordingOutput()

    def interrupt(prompt: str) -> str:
        del prompt
        raise KeyboardInterrupt

    exit_code = run(Agent(ChunkProvider()), input_fn=interrupt, output=output)

    assert exit_code == 0
    assert "Zordon offline" in output.getvalue()


def test_cli_handles_keyboard_interrupt_while_printing_greeting() -> None:
    output = InterruptOnceOutput("Zordon online. Type /exit when you're finished.\n")

    exit_code = run(Agent(ChunkProvider()), output=output)

    assert exit_code == 0
    assert output.getvalue() == "\nZordon offline.\n"


def test_cli_handles_keyboard_interrupt_during_streaming() -> None:
    agent = Agent(InterruptedStreamProvider())
    output = RecordingOutput()

    exit_code = run(agent, input_fn=inputs("Hello"), output=output)

    assert exit_code == 0
    assert output.getvalue().endswith("Zordon: partial\nZordon offline.\n")
    assert output.flush_count >= 3
    assert agent.history == ()


def test_cli_handles_keyboard_interrupt_while_printing_final_newline() -> None:
    output = InterruptOnceOutput("\n")

    exit_code = run(
        Agent(ChunkProvider()),
        input_fn=inputs("Hello"),
        output=output,
    )

    assert exit_code == 0
    assert output.getvalue().endswith("Zordon: Calm response\nZordon offline.\n")
