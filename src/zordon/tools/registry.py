from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zordon.tools.base import (
    DuplicateToolError,
    InvalidArgumentsError,
    Tool,
    ToolContext,
    ToolDefinitionError,
    ToolExecutionError,
    ToolResult,
    UnknownToolError,
    normalize_tool_name,
)


class ToolRegistry:
    """Register, describe, validate, and execute provider-neutral tools."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise ToolDefinitionError("Only Tool instances can be registered.")
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            normalized = normalize_tool_name(name)
        except ToolDefinitionError as exc:
            raise UnknownToolError("The requested tool is not registered.") from exc
        try:
            return self._tools[normalized]
        except KeyError as exc:
            raise UnknownToolError(f"Tool '{normalized}' is not registered.") from exc

    def list_tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def model_definitions(self) -> tuple[dict[str, Any], ...]:
        return tuple(tool.model_definition() for tool in self._tools.values())

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        tool = self.get(name)
        validated = _validate_arguments(tool, arguments)
        try:
            result = tool.execute(validated, context)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool '{tool.name}' failed during execution."
            ) from exc
        if not isinstance(result, ToolResult):
            raise ToolExecutionError(f"Tool '{tool.name}' returned an invalid result.")
        return result


def _validate_arguments(tool: Tool, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise InvalidArgumentsError("Tool arguments must be an object.")
    schema = tool.input_schema
    properties: dict[str, dict[str, Any]] = schema["properties"]
    required: list[str] = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise InvalidArgumentsError(
            f"Tool arguments are missing required values: {', '.join(missing)}."
        )
    unexpected = [name for name in arguments if name not in properties]
    if unexpected:
        raise InvalidArgumentsError(f"unexpected argument: {unexpected[0]}.")
    validated = dict(arguments)
    for name, value in validated.items():
        _validate_value(name, value, properties[name])
    return validated


def _validate_value(name: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected = schema["type"]
    valid = {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, Mapping),
        "string": isinstance(value, str),
    }[expected]
    if not valid:
        article = "an" if expected in {"array", "integer", "object"} else "a"
        raise InvalidArgumentsError(f"Argument '{name}' must be {article} {expected}.")
    if "enum" in schema and value not in schema["enum"]:
        raise InvalidArgumentsError(f"Argument '{name}' is not an allowed value.")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise InvalidArgumentsError(f"Argument '{name}' is too short.")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise InvalidArgumentsError(f"Argument '{name}' is too long.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise InvalidArgumentsError(f"Argument '{name}' is below the minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise InvalidArgumentsError(f"Argument '{name}' is above the maximum.")
    if isinstance(value, list) and "items" in schema:
        item_schema = schema["items"]
        for index, item in enumerate(value):
            _validate_value(f"{name}[{index}]", item, item_schema)
