from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from zordon.agent import (
    MAX_TOOL_CALLS,
    MAX_TOOL_ROUNDS,
    SYSTEM_PROMPT,
    Agent,
    AgentError,
)
from zordon.messages import JSONValue, Message, ModelItem, ToolCallItem, ToolResultItem
from zordon.providers.base import (
    ProviderError,
    ProviderEvent,
    ResponseCompleted,
    TextDelta,
    ToolCall,
)
from zordon.tools import Tool, ToolContext, ToolRegistry, ToolResult


class ScriptedProvider:
    def __init__(self, batches: Sequence[Sequence[ProviderEvent | Exception]]) -> None:
        self._batches = iter(batches)
        self.calls: list[tuple[str, tuple[ModelItem, ...], tuple[Tool, ...]]] = []

    def stream_response(
        self,
        system_prompt: str,
        items: Sequence[ModelItem],
        tools: Sequence[Tool],
    ) -> Iterator[ProviderEvent]:
        self.calls.append((system_prompt, tuple(items), tuple(tools)))
        for event in next(self._batches):
            if isinstance(event, Exception):
                raise event
            yield event


def make_tool(
    calls: list[dict[str, object]] | None = None,
    *,
    execute=None,
) -> Tool:
    def default_execute(
        arguments: dict[str, object], context: ToolContext
    ) -> ToolResult:
        del context
        if calls is not None:
            calls.append(arguments)
        return ToolResult.success("Tool completed.", {"value": arguments["value"]})

    return Tool(
        name="echo",
        description="Echo an integer.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        execute=execute or default_execute,
    )


def completed(*events: ProviderEvent) -> list[ProviderEvent]:
    return [*events, ResponseCompleted()]


def result_dict(
    *,
    ok: bool = True,
    code: str = "ok",
    summary: str = "Tool completed.",
    data: dict[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    return {"ok": ok, "code": code, "summary": summary, "data": data or {}}


def test_system_prompt_defines_zordon_and_current_limits() -> None:
    lowered = SYSTEM_PROMPT.lower()

    assert "zordon" in lowered
    assert "cool mentor" in lowered
    assert "calm" in lowered
    assert "text conversation" in lowered
    assert "do not claim" in lowered


def test_no_tool_turn_streams_and_commits_messages() -> None:
    provider = ScriptedProvider(
        [completed(TextDelta("Calm"), TextDelta(" and ready."))]
    )
    agent = Agent(provider)

    assert list(agent.stream_turn("  Hello Zordon  ")) == ["Calm", " and ready."]
    assert agent.history == (
        Message("user", "Hello Zordon"),
        Message("assistant", "Calm and ready."),
    )
    assert provider.calls[0][1] == (Message("user", "Hello Zordon"),)
    assert provider.calls[0][2] == ()


def test_one_tool_round_commits_complete_trace() -> None:
    provider = ScriptedProvider(
        [
            completed(ToolCall("c1", "echo", {"value": 4})),
            completed(TextDelta("Four.")),
        ]
    )
    tool = make_tool()
    agent = Agent(provider, ToolRegistry([tool]))

    assert list(agent.stream_turn("What is two plus two?")) == ["Four."]
    assert agent.history == (
        Message("user", "What is two plus two?"),
        ToolCallItem("c1", "echo", {"value": 4}),
        ToolResultItem("c1", "echo", result_dict(data={"value": 4})),
        Message("assistant", "Four."),
    )
    assert provider.calls[0][2] == (tool,)
    assert provider.calls[1][1] == agent.history[:-1]


def test_multiple_calls_execute_sequentially_in_model_order() -> None:
    seen: list[dict[str, object]] = []
    provider = ScriptedProvider(
        [
            completed(
                TextDelta("Checking. "),
                ToolCall("c1", "echo", {"value": 1}),
                ToolCall("c2", "echo", {"value": 2}),
            ),
            completed(TextDelta("Done.")),
        ]
    )
    agent = Agent(provider, ToolRegistry([make_tool(seen)]))

    assert list(agent.stream_turn("Use two tools")) == ["Checking. ", "Done."]
    assert seen == [{"value": 1}, {"value": 2}]
    assert agent.history[1] == Message("assistant", "Checking. ")
    assert [
        item.call_id for item in agent.history if isinstance(item, ToolCallItem)
    ] == [
        "c1",
        "c2",
    ]


def test_tool_context_uses_one_incrementing_number_per_user_turn() -> None:
    turn_numbers: list[int] = []

    def execute(arguments: dict[str, object], context: ToolContext) -> ToolResult:
        del arguments
        turn_numbers.append(context.turn_number)
        return ToolResult.success("Done.")

    provider = ScriptedProvider(
        [
            [ProviderError("First turn failed.")],
            completed(
                ToolCall("c1", "echo", {"value": 1}),
                ToolCall("c2", "echo", {"value": 2}),
            ),
            completed(TextDelta("Recovered.")),
        ]
    )
    agent = Agent(provider, ToolRegistry([make_tool(execute=execute)]))

    with pytest.raises(ProviderError, match="First turn failed"):
        list(agent.stream_turn("First"))
    list(agent.stream_turn("Second"))

    assert turn_numbers == [2, 2]


def test_multiple_tool_rounds_feed_complete_candidate_trace_back_to_provider() -> None:
    provider = ScriptedProvider(
        [
            completed(ToolCall("c1", "echo", {"value": 1})),
            completed(ToolCall("c2", "echo", {"value": 2})),
            completed(TextDelta("Finished.")),
        ]
    )
    agent = Agent(provider, ToolRegistry([make_tool()]))

    list(agent.stream_turn("Continue until done"))

    assert len(provider.calls) == 3
    assert provider.calls[2][1] == agent.history[:-1]


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (ToolCall("c1", "missing", {}), "unknown_tool"),
        (ToolCall("c1", "echo", {}), "invalid_arguments"),
    ],
)
def test_expected_tool_errors_are_returned_to_model(call: ToolCall, code: str) -> None:
    provider = ScriptedProvider(
        [completed(call), completed(TextDelta("I could not use that tool."))]
    )
    agent = Agent(provider, ToolRegistry([make_tool()]))

    list(agent.stream_turn("Try it"))

    second_items = provider.calls[1][1]
    result = next(item for item in second_items if isinstance(item, ToolResultItem))
    assert result.result["ok"] is False
    assert result.result["code"] == code


def test_exact_round_and_call_limits_execute_then_force_one_final_response() -> None:
    seen: list[dict[str, object]] = []
    batches: list[Sequence[ProviderEvent | Exception]] = []
    for round_number in range(MAX_TOOL_ROUNDS):
        batches.append(
            completed(
                ToolCall(
                    f"c{round_number * 2 + 1}", "echo", {"value": round_number * 2 + 1}
                ),
                ToolCall(
                    f"c{round_number * 2 + 2}", "echo", {"value": round_number * 2 + 2}
                ),
            )
        )
    batches.append(completed(TextDelta("Final answer.")))
    provider = ScriptedProvider(batches)
    tool = make_tool(seen)
    agent = Agent(provider, ToolRegistry([tool]))

    assert list(agent.stream_turn("Use the full budget")) == ["Final answer."]
    assert len(seen) == MAX_TOOL_CALLS
    assert len(provider.calls) == MAX_TOOL_ROUNDS + 1
    assert provider.calls[-2][2] == (tool,)
    assert provider.calls[-1][2] == ()


def test_batch_crossing_call_budget_executes_none_and_returns_limit_results() -> None:
    seen: list[dict[str, object]] = []
    provider = ScriptedProvider(
        [
            completed(*(ToolCall(f"c{i}", "echo", {"value": i}) for i in range(1, 8))),
            completed(
                ToolCall("c8", "echo", {"value": 8}),
                ToolCall("c9", "echo", {"value": 9}),
            ),
            completed(TextDelta("Limit reached.")),
        ]
    )
    agent = Agent(provider, ToolRegistry([make_tool(seen)]))

    list(agent.stream_turn("Use too many tools"))

    assert [arguments["value"] for arguments in seen] == list(range(1, 8))
    final_items = provider.calls[-1][1]
    limit_results = [
        item
        for item in final_items
        if isinstance(item, ToolResultItem)
        and item.result["code"] == "tool_limit_reached"
    ]
    assert [item.call_id for item in limit_results] == ["c8", "c9"]
    assert provider.calls[-1][2] == ()


def test_tool_failure_is_safe_and_rolls_back_after_visible_text() -> None:
    def explode(arguments: dict[str, object], context: ToolContext) -> ToolResult:
        del arguments, context
        raise RuntimeError("secret implementation detail")

    provider = ScriptedProvider(
        [completed(TextDelta("Working..."), ToolCall("c1", "echo", {"value": 1}))]
    )
    agent = Agent(provider, ToolRegistry([make_tool(execute=explode)]))

    stream = agent.stream_turn("Run it")
    assert next(stream) == "Working..."
    with pytest.raises(AgentError, match="local tool failed unexpectedly") as error:
        next(stream)

    assert "secret" not in str(error.value)
    assert agent.history == ()


def test_provider_failure_rolls_back_after_visible_text() -> None:
    provider = ScriptedProvider(
        [[TextDelta("Partial"), ProviderError("Connection interrupted.")]]
    )
    agent = Agent(provider)

    stream = agent.stream_turn("Keep this clean")
    assert next(stream) == "Partial"
    with pytest.raises(ProviderError, match="interrupted"):
        next(stream)
    assert agent.history == ()


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ([], "before completion"),
        ([ResponseCompleted(), ResponseCompleted()], "more than once"),
        ([TextDelta("late")], "before completion"),
    ],
)
def test_incomplete_or_duplicate_completion_rolls_back(
    batch: list[ProviderEvent], message: str
) -> None:
    agent = Agent(ScriptedProvider([batch]))

    with pytest.raises(ProviderError, match=message):
        list(agent.stream_turn("Hello"))
    assert agent.history == ()


def test_blank_final_response_is_rejected_and_rolled_back() -> None:
    provider = ScriptedProvider(
        [
            completed(ToolCall("c1", "echo", {"value": 1})),
            completed(TextDelta("   ")),
        ]
    )
    agent = Agent(provider, ToolRegistry([make_tool()]))

    with pytest.raises(ProviderError, match="no text"):
        list(agent.stream_turn("Are you there?"))
    assert agent.history == ()


def test_forced_final_response_cannot_request_another_tool() -> None:
    batches = [
        completed(ToolCall(f"c{i}", "echo", {"value": i}))
        for i in range(1, MAX_TOOL_ROUNDS + 1)
    ]
    batches.append(completed(ToolCall("too_late", "echo", {"value": 5})))
    provider = ScriptedProvider(batches)
    agent = Agent(provider, ToolRegistry([make_tool()]))

    with pytest.raises(ProviderError, match="final response requested a tool"):
        list(agent.stream_turn("Keep calling"))
    assert provider.calls[-1][2] == ()
    assert agent.history == ()


def test_later_turn_receives_prior_complete_tool_trace() -> None:
    provider = ScriptedProvider(
        [
            completed(ToolCall("c1", "echo", {"value": 1})),
            completed(TextDelta("First done.")),
            completed(TextDelta("I remember.")),
        ]
    )
    agent = Agent(provider, ToolRegistry([make_tool()]))

    list(agent.stream_turn("First"))
    first_history = agent.history
    list(agent.stream_turn("Second"))

    assert provider.calls[2][1] == (*first_history, Message("user", "Second"))


def test_blank_user_turn_is_rejected_without_provider_call() -> None:
    provider = ScriptedProvider([])
    agent = Agent(provider)

    with pytest.raises(ValueError, match="blank"):
        list(agent.stream_turn("   "))
    assert provider.calls == []
