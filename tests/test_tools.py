from typing import Any

import pytest

from zordon.tools import (
    DuplicateToolError,
    InvalidArgumentsError,
    Tool,
    ToolDefinitionError,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)


def make_add_tool(name: str = "add_numbers") -> Tool:
    def execute(arguments: dict[str, Any]) -> ToolResult:
        return ToolResult.success("calculated", {"total": arguments["left"] + arguments["right"]})

    return Tool(
        name=name,
        description="Add two integers.",
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def test_register_lookup_and_list_tools() -> None:
    registry = ToolRegistry()
    tool = make_add_tool()
    registry.register(tool)

    assert registry.get("add_numbers") is tool
    assert registry.list_tools() == (tool,)


def test_tool_names_are_normalized_consistently() -> None:
    registry = ToolRegistry([make_add_tool("  ADD_NUMBERS  ")])
    assert registry.get(" Add_Numbers ").name == "add_numbers"


def test_successful_execution_returns_predictable_result() -> None:
    result = ToolRegistry([make_add_tool()]).execute(
        "add_numbers", {"left": 4, "right": 7}
    )
    assert result == ToolResult(ok=True, code="ok", summary="calculated", data={"total": 11})


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"left": 1}, "missing required"),
        ({"left": 1, "right": 2, "extra": 3}, "unexpected argument"),
        ({"left": "1", "right": 2}, "must be an integer"),
    ],
)
def test_argument_validation(arguments: dict[str, Any], message: str) -> None:
    registry = ToolRegistry([make_add_tool()])
    with pytest.raises(InvalidArgumentsError, match=message):
        registry.execute("add_numbers", arguments)


def test_arguments_must_be_a_mapping() -> None:
    with pytest.raises(InvalidArgumentsError, match="object"):
        ToolRegistry([make_add_tool()]).execute("add_numbers", [])  # type: ignore[arg-type]


def test_duplicate_registration_is_rejected_after_normalization() -> None:
    registry = ToolRegistry([make_add_tool()])
    with pytest.raises(DuplicateToolError, match="add_numbers"):
        registry.register(make_add_tool("ADD_NUMBERS"))


def test_unknown_lookup_and_execution_are_safe_errors() -> None:
    registry = ToolRegistry()
    with pytest.raises(UnknownToolError, match="not registered"):
        registry.get("missing")
    with pytest.raises(UnknownToolError, match="not registered"):
        registry.execute("missing", {})


@pytest.mark.parametrize("name", ["", "two words", "bad-name", "9starts_wrong"])
def test_invalid_tool_names_are_rejected(name: str) -> None:
    with pytest.raises(ToolDefinitionError, match="name"):
        make_add_tool(name)


@pytest.mark.parametrize(
    "changes",
    [
        {"description": ""},
        {"input_schema": {"type": "array"}},
        {
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": ["missing"],
                "additionalProperties": False,
            }
        },
        {"execute": None},
    ],
)
def test_invalid_tool_definitions_are_rejected(changes: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "name": "valid_tool",
        "description": "Valid description.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "execute": lambda arguments: ToolResult.success("done"),
    }
    values.update(changes)
    with pytest.raises(ToolDefinitionError):
        Tool(**values)


def test_normal_exception_is_wrapped_without_internal_details() -> None:
    def explode(arguments: dict[str, Any]) -> ToolResult:
        del arguments
        raise RuntimeError("database password and traceback detail")

    tool = Tool(
        "explode",
        "Raise an internal exception.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        explode,
    )

    with pytest.raises(ToolExecutionError) as error:
        ToolRegistry([tool]).execute("explode", {})

    assert "database password" not in str(error.value)
    assert str(error.value) == "Tool 'explode' failed during execution."


def test_invalid_executor_result_is_wrapped_predictably() -> None:
    tool = Tool(
        "invalid_result",
        "Return the wrong result type.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        lambda arguments: "wrong",  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(ToolExecutionError, match="invalid result"):
        ToolRegistry([tool]).execute("invalid_result", {})


def test_model_definitions_export_independent_schema_copies() -> None:
    registry = ToolRegistry([make_add_tool()])
    definitions = registry.model_definitions()

    assert definitions == (
        {
            "name": "add_numbers",
            "description": "Add two integers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
        },
    )
    definitions[0]["input_schema"]["required"].append("mutated")  # type: ignore[index,union-attr]
    assert registry.model_definitions()[0]["input_schema"]["required"] == ["left", "right"]  # type: ignore[index]
