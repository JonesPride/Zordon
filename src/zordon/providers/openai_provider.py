from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from openai.types.responses import ResponseInputParam
from openai.types.shared_params import Reasoning

from zordon.config import ReasoningEffort
from zordon.messages import Message, ModelItem, ToolCallItem, ToolResultItem
from zordon.providers.base import (
    ProviderError,
    ProviderEvent,
    ResponseCompleted,
    TextDelta,
    ToolCall,
)
from zordon.tools import Tool

_VISIBLE_DELTA_EVENTS = {
    "response.output_text.delta",
    "response.refusal.delta",
}
_FAILURE_EVENTS = {
    "error",
    "response.failed",
    "response.incomplete",
}
_CALL_DELTA_EVENT = "response.function_call_arguments.delta"
_CALL_ARGUMENTS_DONE_EVENT = "response.function_call_arguments.done"
_OUTPUT_ITEM_ADDED_EVENT = "response.output_item.added"
_OUTPUT_ITEM_DONE_EVENT = "response.output_item.done"
_RESPONSE_COMPLETED_EVENT = "response.completed"

_INCOMPLETE_RESPONSE_MESSAGE = "The model response was incomplete or failed before it completed."


@dataclass(slots=True)
class _FunctionCallState:
    item_id: str
    call_id: str
    name: str
    argument_parts: list[str]
    finalized_arguments: str | None = None
    output_done: bool = False


def _build_response_input(items: Sequence[ModelItem]) -> ResponseInputParam:
    return cast(ResponseInputParam, [_map_item(item) for item in items])


def _build_reasoning(effort: ReasoningEffort) -> Reasoning:
    return Reasoning(effort=effort)


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
        self._reasoning_effort: ReasoningEffort = reasoning_effort
        self._client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
        max_output_tokens: int = 2048,
    ) -> Iterator[ProviderEvent]:
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": system_prompt,
            "input": _build_response_input(items),
            "stream": True,
            "max_output_tokens": max_output_tokens,
            "reasoning": _build_reasoning(self._reasoning_effort),
        }
        if tools:
            request["tools"] = [_map_tool(tool) for tool in tools]

        try:
            stream = self._client.responses.create(**request)
            states: dict[str, _FunctionCallState] = {}
            call_ids: set[str] = set()
            completed = False

            for event in stream:
                event_type = getattr(event, "type", "")
                if completed:
                    if _is_semantic_event(event_type):
                        raise ProviderError("The model emitted a response event after completion.")
                    continue

                if event_type in _VISIBLE_DELTA_EVENTS:
                    delta = getattr(event, "delta", "")
                    if delta:
                        yield TextDelta(delta)
                elif event_type == _OUTPUT_ITEM_ADDED_EVENT:
                    _add_function_call(event, states, call_ids, tools_exposed=bool(tools))
                elif event_type == _CALL_DELTA_EVENT:
                    state = _get_call_state(event, states)
                    if state.finalized_arguments is not None:
                        raise ProviderError("The model emitted tool arguments after completion.")
                    state.argument_parts.append(_require_string(event, "delta"))
                elif event_type == _CALL_ARGUMENTS_DONE_EVENT:
                    state = _get_call_state(event, states)
                    if state.finalized_arguments is not None:
                        raise ProviderError("The model finalized tool arguments more than once.")
                    finalized = _require_string(event, "arguments")
                    if finalized != "".join(state.argument_parts):
                        raise ProviderError(
                            "The model's finalized tool arguments did not match the stream."
                        )
                    state.finalized_arguments = finalized
                elif event_type == _OUTPUT_ITEM_DONE_EVENT:
                    call = _finish_function_call(event, states)
                    if call is not None:
                        yield call
                elif event_type == _RESPONSE_COMPLETED_EVENT:
                    if any(not state.output_done for state in states.values()):
                        raise ProviderError("The model response's function call was incomplete.")
                    completed = True
                    yield ResponseCompleted()
                elif event_type in _FAILURE_EVENTS:
                    raise ProviderError(_INCOMPLETE_RESPONSE_MESSAGE)

            if not completed:
                raise ProviderError(_INCOMPLETE_RESPONSE_MESSAGE)
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


def _map_item(item: ModelItem) -> dict[str, object]:
    if isinstance(item, Message):
        return {"role": item.role, "content": item.content}
    if isinstance(item, ToolCallItem):
        return {
            "type": "function_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": _json_dump(item.arguments),
        }
    if isinstance(item, ToolResultItem):
        return {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": _json_dump(item.result),
        }
    raise TypeError("Unsupported model item.")


def _map_tool(tool: Tool) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": True,
    }


def _json_dump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _add_function_call(
    event: object,
    states: dict[str, _FunctionCallState],
    call_ids: set[str],
    *,
    tools_exposed: bool,
) -> None:
    item = getattr(event, "item", None)
    if getattr(item, "type", "") != "function_call":
        return
    if not tools_exposed:
        raise ProviderError("The model requested a tool even though no tools were exposed.")
    item_id = _require_string(item, "id")
    call_id = _require_string(item, "call_id")
    name = _require_string(item, "name")
    if item_id in states:
        raise ProviderError("The model emitted a duplicate function-call item ID.")
    if call_id in call_ids:
        raise ProviderError("The model emitted a duplicate tool call ID.")
    states[item_id] = _FunctionCallState(item_id, call_id, name, [])
    call_ids.add(call_id)


def _get_call_state(event: object, states: dict[str, _FunctionCallState]) -> _FunctionCallState:
    item_id = _require_string(event, "item_id")
    try:
        return states[item_id]
    except KeyError as exc:
        raise ProviderError("The model referenced an unknown function-call item.") from exc


def _finish_function_call(event: object, states: dict[str, _FunctionCallState]) -> ToolCall | None:
    item = getattr(event, "item", None)
    if getattr(item, "type", "") != "function_call":
        return None
    item_id = _require_string(item, "id")
    try:
        state = states[item_id]
    except KeyError as exc:
        raise ProviderError("The model referenced an unknown function-call item.") from exc
    if state.output_done:
        raise ProviderError("The model completed a function-call item more than once.")
    if state.finalized_arguments is None:
        raise ProviderError("The model response's function call was incomplete.")
    if _require_string(item, "call_id") != state.call_id:
        raise ProviderError("The model changed a tool call ID during streaming.")
    if _require_string(item, "name") != state.name:
        raise ProviderError("The model changed a tool name during streaming.")
    if _require_string(item, "arguments") != state.finalized_arguments:
        raise ProviderError(
            "The model's output tool arguments did not match the finalized arguments."
        )

    try:
        arguments = json.loads(state.finalized_arguments)
    except json.JSONDecodeError as exc:
        raise ProviderError("The model tool arguments were not valid JSON.") from exc
    if not isinstance(arguments, dict):
        raise ProviderError("The model tool arguments must be a JSON object.")
    state.output_done = True
    return ToolCall(state.call_id, state.name, arguments)


def _require_string(value: object, attribute: str) -> str:
    found = getattr(value, attribute, None)
    if not isinstance(found, str):
        raise ProviderError("The model emitted a malformed tool-call event.")
    return found


def _is_semantic_event(event_type: str) -> bool:
    return event_type in (
        _VISIBLE_DELTA_EVENTS
        | _FAILURE_EVENTS
        | {
            _CALL_DELTA_EVENT,
            _CALL_ARGUMENTS_DONE_EVENT,
            _OUTPUT_ITEM_ADDED_EVENT,
            _OUTPUT_ITEM_DONE_EVENT,
            _RESPONSE_COMPLETED_EVENT,
        }
    )
