# Zordon Tier 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependable streaming text conversation loop that remembers successful turns for the current process and fails cleanly without corrupting history.

**Architecture:** A provider-neutral `Agent` owns the system prompt and in-memory history, while an `OpenAIProvider` translates that contract to the Responses API. A thin CLI streams chunks immediately and handles expected configuration and provider failures without duplicating agent logic.

**Tech Stack:** Python 3.12, official OpenAI Python SDK, python-dotenv, pytest, setuptools

## Global Constraints

- Support Python 3.12.
- Treat Windows PowerShell as the primary command environment.
- Keep the package compatible with a later Raspberry Pi deployment.
- Use GPT-5.6 Terra as the configurable default model.
- Keep all OpenAI-specific objects and exceptions inside `src/zordon/providers/openai_provider.py`.
- Do not add tools, audio, durable memory, scheduling, an agent framework, a database, or a web server.
- Never place a real API key in a tracked file, test fixture, command, or document.
- Commit a user turn and assistant turn to history only after the response stream completes successfully.
- Do not begin Tier 2 until automated checks pass and the user completes and approves live Tier 1 verification.

---

## File map

- `pyproject.toml`: package metadata, runtime dependencies, development dependencies, pytest settings, and the `zordon` console command.
- `.gitignore`: local secrets, virtual environments, caches, build outputs, coverage data, and logs.
- `.env.example`: names and safe placeholder values for supported environment variables.
- `README.md`: PowerShell installation, secure API-access handoff, run commands, and Tier 1 verification.
- `src/zordon/__init__.py`: package version.
- `src/zordon/config.py`: environment loading and validation.
- `src/zordon/messages.py`: provider-neutral message value object.
- `src/zordon/providers/base.py`: provider protocol and application-level provider error.
- `src/zordon/providers/openai_provider.py`: OpenAI Responses streaming adapter and error translation.
- `src/zordon/agent.py`: system prompt, successful history, and streaming turn transaction.
- `src/zordon/cli.py`: terminal adapter and composition root.
- `tests/test_config.py`: configuration defaults, validation, and secret-safe representation.
- `tests/test_agent.py`: prompt, streaming, history recall, and rollback behavior.
- `tests/test_openai_provider.py`: request translation, event filtering, and provider failure behavior.
- `tests/test_cli.py`: blank input, streaming flushes, expected-failure recovery, and clean shutdown.

---

### Task 1: Package scaffold and safe configuration

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/zordon/__init__.py`
- Create: `src/zordon/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: process environment variables and an optional injected `Mapping[str, str]`.
- Produces: `Settings`, `ConfigurationError`, and `load_settings(environ=None)`.

- [ ] **Step 1: Add the package metadata and failing configuration tests**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "zordon-assistant"
version = "0.1.0"
description = "The text-first core of the Zordon personal AI assistant"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "openai>=2.0,<3.0",
    "python-dotenv>=1.0,<2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<10.0",
]

[project.scripts]
zordon = "zordon.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

Create the minimal `README.md` required by the package metadata:

```markdown
# Zordon

Zordon is being built one independently testable tier at a time.
```

Create `tests/test_config.py`:

```python
import pytest

from zordon.config import ConfigurationError, load_settings


def test_load_settings_uses_tier_one_defaults() -> None:
    settings = load_settings({"OPENAI_API_KEY": "test-key"})

    assert settings.api_key == "test-key"
    assert settings.model == "gpt-5.6-terra"
    assert settings.timeout_seconds == 60.0


def test_load_settings_accepts_supported_overrides() -> None:
    settings = load_settings(
        {
            "OPENAI_API_KEY": "test-key",
            "ZORDON_MODEL": "gpt-5.6-sol",
            "ZORDON_REQUEST_TIMEOUT_SECONDS": "15.5",
        }
    )

    assert settings.model == "gpt-5.6-sol"
    assert settings.timeout_seconds == 15.5


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "OPENAI_API_KEY"),
        ({"OPENAI_API_KEY": "   "}, "OPENAI_API_KEY"),
        (
            {
                "OPENAI_API_KEY": "test-key",
                "ZORDON_REQUEST_TIMEOUT_SECONDS": "zero",
            },
            "positive number",
        ),
        (
            {
                "OPENAI_API_KEY": "test-key",
                "ZORDON_REQUEST_TIMEOUT_SECONDS": "0",
            },
            "positive number",
        ),
    ],
)
def test_load_settings_rejects_invalid_configuration(
    environ: dict[str, str], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(environ)


def test_settings_repr_does_not_reveal_api_key() -> None:
    settings = load_settings({"OPENAI_API_KEY": "super-secret-test-key"})

    assert "super-secret-test-key" not in repr(settings)
```

- [ ] **Step 2: Run the configuration tests and confirm the expected failure**

Run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests/test_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'zordon.config'`.

- [ ] **Step 3: Implement minimal safe configuration**

Create `.gitignore`:

```gitignore
.env
.env.*
!.env.example
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
build/
dist/
*.egg-info/
*.log
```

Create `.env.example`:

```dotenv
OPENAI_API_KEY=replace-with-a-key-from-the-secure-setup-flow
ZORDON_MODEL=gpt-5.6-terra
ZORDON_REQUEST_TIMEOUT_SECONDS=60
```

Create `src/zordon/__init__.py`:

```python
"""Zordon's provider-neutral assistant core."""

__version__ = "0.1.0"
```

Create `src/zordon/config.py`:

```python
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_TIMEOUT_SECONDS = 60.0


class ConfigurationError(ValueError):
    """Raised when Zordon cannot start safely from its configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    if environ is None:
        load_dotenv()
        environ = os.environ

    api_key = environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. Use Zordon's secure API-key setup first."
        )

    model = environ.get("ZORDON_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    raw_timeout = (
        environ.get(
            "ZORDON_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_TIMEOUT_SECONDS),
        ).strip()
        or str(DEFAULT_TIMEOUT_SECONDS)
    )

    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ConfigurationError(
            "ZORDON_REQUEST_TIMEOUT_SECONDS must be a positive number."
        ) from exc

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConfigurationError(
            "ZORDON_REQUEST_TIMEOUT_SECONDS must be a positive number."
        )

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
```

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```powershell
python -m pytest tests/test_config.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit the configuration slice**

```powershell
git add pyproject.toml README.md .gitignore .env.example src/zordon/__init__.py src/zordon/config.py tests/test_config.py
git commit -m "feat: add safe Tier 1 configuration"
```

---

### Task 2: Provider-neutral messages and streaming agent core

**Files:**
- Create: `src/zordon/messages.py`
- Create: `src/zordon/providers/__init__.py`
- Create: `src/zordon/providers/base.py`
- Create: `src/zordon/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: a `ModelProvider` whose `stream_reply(system_prompt, messages)` method yields text chunks.
- Produces: immutable `Message`, `ProviderError`, `ModelProvider`, `SYSTEM_PROMPT`, and `Agent.stream_turn(user_text)`.

- [ ] **Step 1: Write failing agent transaction tests**

Create `tests/test_agent.py`:

```python
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
```

- [ ] **Step 2: Run the agent tests and confirm the expected failure**

Run:

```powershell
python -m pytest tests/test_agent.py -q
```

Expected: collection fails because `zordon.agent`, `zordon.messages`, and `zordon.providers.base` do not exist.

- [ ] **Step 3: Implement the provider-neutral contracts and agent**

Create `src/zordon/messages.py`:

```python
from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
```

Create `src/zordon/providers/__init__.py`:

```python
"""Model-provider adapters for Zordon."""
```

Create `src/zordon/providers/base.py`:

```python
from collections.abc import Iterator, Sequence
from typing import Protocol

from zordon.messages import Message


class ProviderError(RuntimeError):
    """A model-provider failure that can be safely explained to the user."""


class ModelProvider(Protocol):
    def stream_reply(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> Iterator[str]:
        """Yield user-visible text chunks for one candidate conversation."""
        raise NotImplementedError
```

Create `src/zordon/agent.py`:

```python
from __future__ import annotations

from collections.abc import Iterator

from zordon.messages import Message
from zordon.providers.base import ModelProvider, ProviderError

SYSTEM_PROMPT = """\
You are Zordon, the user's personal AI assistant.

Your purpose is to remember useful context during the current conversation,
help the user think clearly, and keep their goals and projects moving.

Sound like a cool mentor: calm, capable, realistic, concise, and plain-spoken.
Be warm without being theatrical or overly familiar.

At this stage, you can only hold a text conversation. You do not yet have
tools, voice, persistent memory, reminders, file access, or background
abilities. Do not claim that you performed actions or accessed information
outside the conversation.
"""


class Agent:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def stream_turn(self, user_text: str) -> Iterator[str]:
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("A user turn cannot be blank.")

        user_message = Message(role="user", content=clean_text)
        candidate_history = (*self._history, user_message)
        reply_parts: list[str] = []

        for chunk in self._provider.stream_reply(
            SYSTEM_PROMPT,
            candidate_history,
        ):
            if not chunk:
                continue
            reply_parts.append(chunk)
            yield chunk

        reply = "".join(reply_parts)
        if not reply.strip():
            raise ProviderError("The model returned no text. Please try again.")

        self._history.extend(
            (
                user_message,
                Message(role="assistant", content=reply),
            )
        )
```

- [ ] **Step 4: Run the focused tests and then the current suite**

Run:

```powershell
python -m pytest tests/test_agent.py -q
python -m pytest -q
```

Expected: `5 passed` for the focused file and `12 passed` for the current suite.

- [ ] **Step 5: Commit the agent-core slice**

```powershell
git add src/zordon/messages.py src/zordon/providers/__init__.py src/zordon/providers/base.py src/zordon/agent.py tests/test_agent.py
git commit -m "feat: add streaming agent core"
```

---

### Task 3: OpenAI Responses streaming adapter

**Files:**
- Create: `src/zordon/providers/openai_provider.py`
- Test: `tests/test_openai_provider.py`

**Interfaces:**
- Consumes: `Settings` values and provider-neutral `Message` objects.
- Produces: `OpenAIProvider.stream_reply(system_prompt, messages)` yielding only visible text and raising `ProviderError` for expected provider failures.

- [ ] **Step 1: Write failing adapter tests at the SDK boundary**

Create `tests/test_openai_provider.py`:

```python
from types import SimpleNamespace
from typing import Any

import pytest

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
```

- [ ] **Step 2: Run the adapter tests and confirm the expected failure**

Run:

```powershell
python -m pytest tests/test_openai_provider.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'zordon.providers.openai_provider'`.

- [ ] **Step 3: Implement the OpenAI-only adapter**

Create `src/zordon/providers/openai_provider.py`:

```python
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

from zordon.messages import Message
from zordon.providers.base import ProviderError

_VISIBLE_DELTA_EVENTS = {
    "response.output_text.delta",
    "response.refusal.delta",
}


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    def stream_reply(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> Iterator[str]:
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

        try:
            stream = self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=payload,
                stream=True,
            )
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type in _VISIBLE_DELTA_EVENTS:
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield delta
                elif event_type in {"error", "response.failed"}:
                    raise ProviderError(
                        "The model response failed before it completed."
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
            raise ProviderError(
                "The model provider returned an error. Please retry."
            ) from exc
        except Exception as exc:
            raise ProviderError(
                "An unexpected model-provider error occurred. Please retry."
            ) from exc
```

- [ ] **Step 4: Run focused and full automated tests**

Run:

```powershell
python -m pytest tests/test_openai_provider.py -q
python -m pytest -q
```

Expected: `3 passed` for the focused file and `15 passed` for the current suite.

- [ ] **Step 5: Commit the provider slice**

```powershell
git add src/zordon/providers/openai_provider.py tests/test_openai_provider.py
git commit -m "feat: add OpenAI streaming provider"
```

---

### Task 4: Streaming PowerShell-friendly terminal interface

**Files:**
- Create: `src/zordon/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Agent.stream_turn()`, validated `Settings`, terminal input, and a writable text stream.
- Produces: `run(agent, input_fn=input, output=sys.stdout)` and the `zordon` console command.

- [ ] **Step 1: Write failing CLI behavior tests**

Create `tests/test_cli.py`:

```python
from collections.abc import Iterator, Sequence
from io import StringIO

from zordon.agent import Agent
from zordon.messages import Message
from zordon.providers.base import ProviderError
from zordon.cli import run


class ChunkProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream_reply(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> Iterator[str]:
        del system_prompt, messages
        self.calls += 1
        yield "Calm"
        yield " response"


class FailThenRecoverProvider:
    def __init__(self) -> None:
        self.calls = 0

    def stream_reply(
        self, system_prompt: str, messages: Sequence[Message]
    ) -> Iterator[str]:
        del system_prompt, messages
        self.calls += 1
        if self.calls == 1:
            yield "partial"
            raise ProviderError("Connection interrupted.")
        yield "recovered"


class RecordingOutput(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def inputs(*values: str):
    iterator = iter(values)

    def read(prompt: str) -> str:
        del prompt
        return next(iterator)

    return read


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


def test_cli_handles_keyboard_interrupt_without_traceback() -> None:
    output = RecordingOutput()

    def interrupt(prompt: str) -> str:
        del prompt
        raise KeyboardInterrupt

    exit_code = run(Agent(ChunkProvider()), input_fn=interrupt, output=output)

    assert exit_code == 0
    assert "Zordon offline" in output.getvalue()
```

- [ ] **Step 2: Run the CLI tests and confirm the expected failure**

Run:

```powershell
python -m pytest tests/test_cli.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'zordon.cli'`.

- [ ] **Step 3: Implement the CLI and PowerShell runbook**

Create `src/zordon/cli.py`:

```python
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from zordon.agent import Agent
from zordon.config import ConfigurationError, load_settings
from zordon.providers.base import ProviderError
from zordon.providers.openai_provider import OpenAIProvider

_EXIT_COMMANDS = {"/exit", "exit", "quit"}


def run(
    agent: Agent,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    output.write("Zordon online. Type /exit when you're finished.\n")
    output.flush()

    while True:
        try:
            raw_text = input_fn("You: ")
        except (EOFError, KeyboardInterrupt):
            output.write("\nZordon offline.\n")
            output.flush()
            return 0

        user_text = raw_text.strip()
        if not user_text:
            continue
        if user_text.lower() in _EXIT_COMMANDS:
            output.write("Zordon offline.\n")
            output.flush()
            return 0

        output.write("Zordon: ")
        output.flush()
        printed_chunk = False

        try:
            for chunk in agent.stream_turn(user_text):
                output.write(chunk)
                output.flush()
                printed_chunk = True
        except ProviderError as exc:
            if printed_chunk:
                output.write(f"\n[Reply incomplete: {exc}]\n")
            else:
                output.write(f"[Unable to answer: {exc}]\n")
            output.flush()
            continue

        output.write("\n")
        output.flush()


def main() -> int:
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    provider = OpenAIProvider(
        api_key=settings.api_key,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
    )
    return run(Agent(provider))


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `README.md`:

```markdown
# Zordon

Zordon is a personal AI assistant built one independently testable tier at a
time. Tier 1 is a streaming text conversation that remembers successful turns
until the process exits.

## Requirements

- Windows
- PowerShell
- Python 3.12
- OpenAI API access configured through Zordon's secure setup flow

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The secure setup flow stores `OPENAI_API_KEY` locally. Zordon can also read
`ZORDON_MODEL` and `ZORDON_REQUEST_TIMEOUT_SECONDS` from the environment. Never
commit `.env`.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
zordon
```

Type `/exit` to stop. The text interface remains part of Zordon after voice is
added because it is the dependable debugging and fallback path.

## Automated checks

```powershell
python -m pytest -q
python -m compileall -q src tests
python -c "from zordon.cli import main; print('Zordon import OK')"
```

## Tier 1 live verification

1. Tell Zordon a temporary fact.
2. Talk about something else for at least two turns.
3. Ask Zordon to recall the temporary fact.
4. Confirm the answer appeared incrementally.
5. Exit and restart Zordon.
6. Confirm the temporary fact is forgotten after restart.
7. Disconnect the network for one turn and confirm Zordon reports the problem
   without a traceback, then reconnect and continue.

Tier 2 does not begin until these checks pass and the user approves Tier 1.
```

- [ ] **Step 4: Run the CLI tests and the full suite**

Run:

```powershell
python -m pytest tests/test_cli.py -q
python -m pytest -q
```

Expected: `3 passed` for the focused file and `18 passed` for the full suite.

- [ ] **Step 5: Verify the no-key startup path**

Run in a fresh PowerShell process where `OPENAI_API_KEY` is absent:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
python -m zordon.cli
```

Expected: one actionable `Configuration error:` line, no traceback, and process exit code `2`.

- [ ] **Step 6: Commit the CLI slice**

```powershell
git add src/zordon/cli.py tests/test_cli.py README.md
git commit -m "feat: add streaming text interface"
```

---

### Task 5: Tier 1 automated acceptance gate

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-zordon-tier-1.md` only to mark completed checkboxes during execution.

**Interfaces:**
- Consumes: the completed package and test suite.
- Produces: fresh evidence that Tier 1 is ready for the user's live verification.

- [ ] **Step 1: Recreate the supported environment cleanly**

Run from the project root in PowerShell:

```powershell
deactivate 2>$null
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Expected: installation succeeds on Python 3.12 with no dependency conflict.

- [ ] **Step 2: Run the complete automated verification**

Run:

```powershell
python -m pytest -q
python -m compileall -q src tests
python -c "from zordon.agent import Agent; from zordon.cli import main; print('Zordon import OK')"
```

Expected:

```text
18 passed
Zordon import OK
```

`compileall` produces no error output.

- [ ] **Step 3: Verify secret and repository hygiene**

Run:

```powershell
git diff --check
git ls-files .env
git grep -nE "sk-[A-Za-z0-9_-]{20,}" -- . ":!docs/superpowers/plans/*"
git status --short
```

Expected:

- `git diff --check` prints nothing.
- `git ls-files .env` prints nothing.
- the API-key-pattern search prints nothing.
- `git status --short` shows only the intentional implementation-plan checkbox update, if checkboxes were marked.

- [ ] **Step 4: Review Tier 1 against the approved specification**

Confirm each statement with a file or test:

- the CLI streams output chunks and flushes each chunk;
- successful history is resent on later turns;
- failed and empty replies do not change history;
- only the OpenAI adapter imports the OpenAI SDK;
- expected configuration and network/provider failures avoid tracebacks;
- no tool, audio, durable-memory, heartbeat, or safety-gate implementation has entered Tier 1;
- the PowerShell runbook includes the live verification procedure.

Expected: every statement has direct test or source evidence and no Tier 1
requirement remains uncovered.

- [ ] **Step 5: Commit execution tracking, if it changed**

```powershell
git add docs/superpowers/plans/2026-07-31-zordon-tier-1.md
git commit -m "docs: record Tier 1 implementation verification"
```

Skip this commit only if the plan file has no execution-tracking changes.

- [ ] **Step 6: Hand off to the user without starting Tier 2**

Provide:

- the exact automated test count;
- the project path;
- the secure OpenAI API-access step;
- the PowerShell install and run commands;
- the eight live verification checks from the approved design.

Wait for the user's explicit Tier 1 approval before designing or implementing
Tier 2.

---

## Plan self-review

- Every Tier 1 requirement maps to an implementation step, automated test, or
  live user check.
- Production code follows a test that is expected to fail for the missing
  behavior.
- Function names, signatures, message fields, model defaults, and test counts
  are consistent across tasks.
- All provider-specific types remain in the OpenAI adapter.
- The history transaction prevents partial or failed turns from being saved.
- The plan contains no implementation work from later tiers.
