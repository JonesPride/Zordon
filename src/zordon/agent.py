from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import cast

from zordon.messages import JSONValue, Message, ModelItem, ToolCallItem, ToolResultItem
from zordon.providers.base import (
    ModelProvider,
    ProviderError,
    ResponseCompleted,
    TextDelta,
    ToolCall,
)
from zordon.tools import (
    InvalidArgumentsError,
    Tool,
    ToolContext,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 8

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


class AgentError(ProviderError):
    """An orchestration failure that is safe to show without internal details."""


class Agent:
    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry or ToolRegistry()
        self._history: list[ModelItem] = []
        self._turn_number = 0

    @property
    def history(self) -> tuple[ModelItem, ...]:
        return tuple(self._history)

    def stream_turn(self, user_text: str) -> Iterator[str]:
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("A user turn cannot be blank.")

        self._turn_number += 1
        context = ToolContext(turn_number=self._turn_number)

        candidate: list[ModelItem] = [
            *self._history,
            Message(role="user", content=clean_text),
        ]
        tool_rounds = 0
        call_count = 0
        force_final = False

        while True:
            tools: tuple[Tool, ...] = () if force_final else self._registry.list_tools()
            text, calls = yield from self._stream_batch(candidate, tools)

            if not calls:
                if not text.strip():
                    raise ProviderError("The model returned no text. Please try again.")
                candidate.append(Message(role="assistant", content=text))
                self._history = candidate
                return

            if force_final:
                raise ProviderError(
                    "The model's final response requested a tool. Please retry."
                )

            if text.strip():
                candidate.append(Message(role="assistant", content=text))

            tool_rounds += 1
            if call_count + len(calls) > MAX_TOOL_CALLS:
                for call in calls:
                    candidate.extend(
                        (
                            _call_item(call),
                            ToolResultItem(
                                call.call_id,
                                call.name,
                                _result_dict(
                                    ToolResult.failure(
                                        "tool_limit_reached",
                                        "The per-turn tool-call limit was reached.",
                                    )
                                ),
                            ),
                        )
                    )
                force_final = True
                continue

            call_count += len(calls)
            for call in calls:
                candidate.append(_call_item(call))
                result = self._execute_tool(call, context)
                candidate.append(
                    ToolResultItem(
                        call.call_id,
                        call.name,
                        _result_dict(result),
                    )
                )

            if tool_rounds >= MAX_TOOL_ROUNDS:
                force_final = True

    def _stream_batch(
        self,
        candidate: list[ModelItem],
        tools: tuple[Tool, ...],
    ) -> Generator[str, None, tuple[str, list[ToolCall]]]:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        completed = False

        for event in self._provider.stream_response(
            SYSTEM_PROMPT,
            candidate,
            tools,
        ):
            if isinstance(event, ResponseCompleted):
                if completed:
                    raise ProviderError(
                        "The model completed the response more than once."
                    )
                completed = True
            elif completed:
                raise ProviderError(
                    "The model emitted a response event after completion."
                )
            elif isinstance(event, TextDelta):
                if event.text:
                    text_parts.append(event.text)
                    yield event.text
            elif isinstance(event, ToolCall):
                calls.append(event)
            else:
                raise ProviderError("The model emitted an unsupported response event.")

        if not completed:
            raise ProviderError("The model response ended before completion.")
        return "".join(text_parts), calls

    def _execute_tool(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            return self._registry.execute(call.name, call.arguments, context)
        except UnknownToolError as exc:
            return ToolResult.failure("unknown_tool", str(exc))
        except InvalidArgumentsError as exc:
            return ToolResult.failure("invalid_arguments", str(exc))
        except ToolExecutionError as exc:
            raise AgentError("A local tool failed unexpectedly. Please retry.") from exc


def _call_item(call: ToolCall) -> ToolCallItem:
    return ToolCallItem(call.call_id, call.name, call.arguments)


def _result_dict(result: ToolResult) -> dict[str, JSONValue]:
    value = {
        "ok": result.ok,
        "code": result.code,
        "summary": result.summary,
        "data": dict(result.data),
    }
    return cast(dict[str, JSONValue], value)
