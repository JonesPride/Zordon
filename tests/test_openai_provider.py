import json
from types import SimpleNamespace
from typing import Any, assert_type

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from openai.types.responses import ResponseInputParam
from openai.types.shared_params import Reasoning

from zordon.agent import Agent
from zordon.messages import Message, ToolCallItem, ToolResultItem
from zordon.providers import openai_provider
from zordon.providers.base import (
    ProviderError,
    ResponseCompleted,
    TextDelta,
    ToolCall,
)
from zordon.providers.openai_provider import OpenAIProvider
from zordon.tools import Tool, ToolResult


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


def event(event_type: str, **values: object) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **values)


def function_item(
    item_id: str,
    call_id: str,
    name: str = "calculate",
    arguments: str | None = None,
) -> SimpleNamespace:
    values: dict[str, object] = {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
    }
    if arguments is not None:
        values["arguments"] = arguments
    return SimpleNamespace(**values)


def completed_call_events(
    *,
    item_id: str = "item_1",
    call_id: str = "call_1",
    name: str = "calculate",
    arguments: str = '{"expression":"2+2"}',
) -> list[SimpleNamespace]:
    return [
        event(
            "response.output_item.added",
            item=function_item(item_id, call_id, name),
        ),
        event(
            "response.function_call_arguments.delta",
            item_id=item_id,
            delta=arguments,
        ),
        event(
            "response.function_call_arguments.done",
            item_id=item_id,
            arguments=arguments,
        ),
        event(
            "response.output_item.done",
            item=function_item(item_id, call_id, name, arguments),
        ),
    ]


CALCULATE = Tool(
    name="calculate",
    description="Evaluate a restricted expression.",
    input_schema={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    execute=lambda arguments, context: ToolResult.success("unused"),
)


def make_provider(
    events: list[SimpleNamespace] | None = None,
    failure: Exception | None = None,
) -> tuple[OpenAIProvider, FakeResponses]:
    responses = FakeResponses(events, failure)
    return (
        OpenAIProvider(
            api_key="test-key",
            model="gpt-5.6-terra",
            timeout_seconds=12.0,
            client=FakeClient(responses),
        ),
        responses,
    )


def test_sdk_payload_builders_preserve_request_shapes() -> None:
    response_input = openai_provider._build_response_input(
        [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]
    )
    reasoning = openai_provider._build_reasoning("high")

    assert_type(response_input, ResponseInputParam)
    assert_type(reasoning, Reasoning)
    assert response_input == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert reasoning == {"effort": "high"}


def test_maps_model_items_and_strict_tool_definitions() -> None:
    provider, responses = make_provider([event("response.completed")])
    result = {
        "ok": True,
        "code": "calculated",
        "summary": "Done",
        "data": {"result": "4"},
    }

    assert list(
        provider.stream_response(
            "System instructions",
            [
                Message("user", "Math"),
                ToolCallItem("call_1", "calculate", {"expression": "2+2"}),
                ToolResultItem("call_1", "calculate", result),
            ],
            [CALCULATE],
        )
    ) == [ResponseCompleted()]

    assert responses.calls == [
        {
            "model": "gpt-5.6-terra",
            "instructions": "System instructions",
            "input": [
                {"role": "user", "content": "Math"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "calculate",
                    "arguments": '{"expression":"2+2"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": json.dumps(result, separators=(",", ":"), sort_keys=True),
                },
            ],
            "stream": True,
            "max_output_tokens": 2048,
            "reasoning": {"effort": "medium"},
            "tools": [
                {
                    "type": "function",
                    "name": "calculate",
                    "description": "Evaluate a restricted expression.",
                    "parameters": CALCULATE.input_schema,
                    "strict": True,
                }
            ],
        }
    ]


def test_omits_tools_for_forced_final_request() -> None:
    provider, responses = make_provider([event("response.completed")])

    list(provider.stream_response("System", [Message("user", "Finish")], []))

    assert "tools" not in responses.calls[0]


def test_emits_text_deltas_and_explicit_completion() -> None:
    provider, _ = make_provider(
        [
            event("response.created"),
            event("response.output_text.delta", delta="Hello"),
            event("response.refusal.delta", delta=" there"),
            event("response.completed"),
        ]
    )

    assert list(provider.stream_response("System", [], [])) == [
        TextDelta("Hello"),
        TextDelta(" there"),
        ResponseCompleted(),
    ]


def test_emits_call_only_after_arguments_and_output_item_complete() -> None:
    events = completed_call_events(arguments='{"expression":"2')
    events[1:2] = [
        event(
            "response.function_call_arguments.delta",
            item_id="item_1",
            delta='{"expression":"2',
        ),
        event(
            "response.function_call_arguments.delta",
            item_id="item_1",
            delta='+2"}',
        ),
    ]
    events[3].arguments = '{"expression":"2+2"}'
    events[4].item.arguments = '{"expression":"2+2"}'
    events.append(event("response.completed"))
    provider, _ = make_provider(events)

    assert list(provider.stream_response("System", [], [CALCULATE])) == [
        ToolCall("call_1", "calculate", {"expression": "2+2"}),
        ResponseCompleted(),
    ]


def test_emits_multiple_calls_in_output_completion_order() -> None:
    events = [
        *completed_call_events(
            item_id="item_1", call_id="call_1", arguments='{"expression":"1+1"}'
        ),
        *completed_call_events(
            item_id="item_2", call_id="call_2", arguments='{"expression":"2+2"}'
        ),
        event("response.completed"),
    ]
    provider, _ = make_provider(events)

    assert list(provider.stream_response("System", [], [CALCULATE])) == [
        ToolCall("call_1", "calculate", {"expression": "1+1"}),
        ToolCall("call_2", "calculate", {"expression": "2+2"}),
        ResponseCompleted(),
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [("not json", "valid JSON"), ("[]", "JSON object")],
)
def test_rejects_invalid_final_arguments(arguments: str, message: str) -> None:
    provider, _ = make_provider(
        [*completed_call_events(arguments=arguments), event("response.completed")]
    )

    with pytest.raises(ProviderError, match=message):
        list(provider.stream_response("System", [], [CALCULATE]))


def test_rejects_duplicate_call_ids() -> None:
    provider, _ = make_provider(
        [
            *completed_call_events(item_id="item_1", call_id="duplicate"),
            event(
                "response.output_item.added",
                item=function_item("item_2", "duplicate"),
            ),
        ]
    )

    with pytest.raises(ProviderError, match="duplicate"):
        list(provider.stream_response("System", [], [CALCULATE]))


@pytest.mark.parametrize(
    "events",
    [
        [event("response.function_call_arguments.delta", item_id="missing", delta="{}")],
        [
            event(
                "response.function_call_arguments.done",
                item_id="missing",
                arguments="{}",
            )
        ],
        [
            event(
                "response.output_item.done",
                item=function_item("missing", "call_1", arguments="{}"),
            )
        ],
    ],
)
def test_rejects_call_events_without_matching_added_item(
    events: list[SimpleNamespace],
) -> None:
    provider, _ = make_provider(events)

    with pytest.raises(ProviderError, match="unknown function-call item"):
        list(provider.stream_response("System", [], [CALCULATE]))


def test_rejects_argument_delta_after_arguments_done() -> None:
    provider, _ = make_provider(
        [
            event(
                "response.output_item.added",
                item=function_item("item_1", "call_1"),
            ),
            event(
                "response.function_call_arguments.done",
                item_id="item_1",
                arguments="",
            ),
            event(
                "response.function_call_arguments.delta",
                item_id="item_1",
                delta="{}",
            ),
        ]
    )

    with pytest.raises(ProviderError, match="after completion"):
        list(provider.stream_response("System", [], [CALCULATE]))


def test_rejects_argument_finalization_mismatch() -> None:
    events = completed_call_events()
    events[2].arguments = '{"expression":"3+3"}'
    provider, _ = make_provider(events)

    with pytest.raises(ProviderError, match="did not match"):
        list(provider.stream_response("System", [], [CALCULATE]))


def test_rejects_incomplete_function_item_at_response_completion() -> None:
    provider, _ = make_provider(
        [
            event(
                "response.output_item.added",
                item=function_item("item_1", "call_1"),
            ),
            event("response.completed"),
        ]
    )

    with pytest.raises(ProviderError, match="function call was incomplete"):
        list(provider.stream_response("System", [], [CALCULATE]))


def test_rejects_function_call_when_no_tools_are_exposed() -> None:
    provider, _ = make_provider(
        [
            event(
                "response.output_item.added",
                item=function_item("item_1", "call_1"),
            )
        ]
    )

    with pytest.raises(ProviderError, match="no tools were exposed"):
        list(provider.stream_response("System", [], []))


def test_rejects_eof_without_explicit_completion() -> None:
    provider, _ = make_provider([event("response.output_text.delta", delta="Partial")])

    with pytest.raises(ProviderError, match="before it completed"):
        list(provider.stream_response("System", [], []))


def test_rejects_semantic_events_after_completion() -> None:
    provider, _ = make_provider(
        [
            event("response.completed"),
            event("response.output_text.delta", delta="late"),
        ]
    )

    stream = provider.stream_response("System", [], [])
    assert next(stream) == ResponseCompleted()
    with pytest.raises(ProviderError, match="after completion"):
        next(stream)


@pytest.mark.parametrize("event_type", ["error", "response.failed", "response.incomplete"])
def test_rejects_provider_failure_events(event_type: str) -> None:
    provider, _ = make_provider([event(event_type)])

    with pytest.raises(ProviderError, match="incomplete or failed"):
        list(provider.stream_response("System", [], []))


def test_tier_one_text_agent_remains_compatible() -> None:
    provider, _ = make_provider(
        [
            event("response.output_text.delta", delta="Still works"),
            event("response.completed"),
        ]
    )
    agent = Agent(provider)

    assert list(agent.stream_turn("Hello")) == ["Still works"]
    assert agent.history == (
        Message("user", "Hello"),
        Message("assistant", "Still works"),
    )


def test_translates_unexpected_sdk_failure_without_leaking_details() -> None:
    provider, _ = make_provider(failure=RuntimeError("internal detail"))

    with pytest.raises(ProviderError, match="unexpected model-provider") as error:
        list(provider.stream_response("System", [], []))

    assert "internal detail" not in str(error.value)


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
            APITimeoutError(httpx.Request("POST", "/responses")),
            "could not be reached",
        ),
        (
            APIConnectionError(request=httpx.Request("POST", "/responses")),
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
def test_maps_sdk_exceptions_to_safe_provider_errors(
    sdk_error: Exception, safe_message: str
) -> None:
    provider, _ = make_provider(failure=sdk_error)

    with pytest.raises(ProviderError, match=safe_message) as error:
        list(provider.stream_response("System", [], []))

    assert error.value.__cause__ is sdk_error
