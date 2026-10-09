from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from typing import cast

from zordon.debug_logging import DebugLogger
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
        system_prompt: str = SYSTEM_PROMPT,
        history_message_limit: int = 40,
        output_token_limit: int = 2048,
        debug_logger: DebugLogger | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry or ToolRegistry()
        self._history: list[ModelItem] = []
        self._turn_number = 0
        self._system_prompt = system_prompt
        self._history_message_limit = history_message_limit
        self._output_token_limit = output_token_limit
        self._debug_logger = debug_logger or DebugLogger(None)

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
        self._debug_logger.event("turn_started", message_count=_message_count(candidate))

        try:
            while True:
                tools: tuple[Tool, ...] = () if force_final else self._registry.list_tools()
                text, calls = yield from self._stream_batch(candidate, tools)

                if not calls:
                    if not text.strip():
                        raise ProviderError("The model returned no text. Please try again.")
                    candidate.append(Message(role="assistant", content=text))
                    self._history = _bounded_history(candidate, self._history_message_limit)
                    self._debug_logger.event(
                        "turn_completed",
                        message_count=_message_count(self._history),
                        output_chars=len(text),
                    )
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
        except Exception:
            self._debug_logger.event("turn_failed", message_count=_message_count(candidate))
            raise

    def _stream_batch(
        self,
        candidate: list[ModelItem],
        tools: tuple[Tool, ...],
    ) -> Generator[str, None, tuple[str, list[ToolCall]]]:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        completed = False

        for event in self._provider.stream_response(
            self._system_prompt,
            candidate,
            tools,
            self._output_token_limit,
        ):
            if isinstance(event, ResponseCompleted):
                if completed:
                    raise ProviderError("The model completed the response more than once.")
                completed = True
            elif completed:
                raise ProviderError("The model emitted a response event after completion.")
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


def _message_count(items: Sequence[ModelItem]) -> int:
    return sum(isinstance(item, Message) for item in items)


def _bounded_history(items: list[ModelItem], message_limit: int) -> list[ModelItem]:
    turn_starts = [
        index
        for index, item in enumerate(items)
        if isinstance(item, Message) and item.role == "user"
    ]
    if len(turn_starts) < 2 or _message_count(items) <= message_limit:
        return items

    keep_from = turn_starts[-1]
    for start in reversed(turn_starts[:-1]):
        if _message_count(items[start:]) > message_limit:
            break
        keep_from = start
    return items[keep_from:]


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
