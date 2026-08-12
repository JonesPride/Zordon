from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

ToolArguments = dict[str, Any]
ToolExecutor = Callable[[ToolArguments, "ToolContext"], "ToolResult"]

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ToolError(RuntimeError):
    """Base class for tool errors that are safe to surface to the agent."""


class ToolDefinitionError(ToolError, ValueError):
    """Raised when a tool contract is invalid."""


class DuplicateToolError(ToolError):
    """Raised when a normalized tool name is already registered."""


class UnknownToolError(ToolError):
    """Raised when a tool name is not registered."""


class InvalidArgumentsError(ToolError, ValueError):
    """Raised when structured arguments do not match a tool schema."""


class ToolExecutionError(ToolError):
    """Raised when a tool cannot produce a valid result."""


def normalize_tool_name(name: str) -> str:
    if not isinstance(name, str):
        raise ToolDefinitionError("Tool name must be a string.")
    normalized = name.strip().lower()
    if not _NAME_PATTERN.fullmatch(normalized):
        raise ToolDefinitionError(
            "Tool name must start with a letter and contain only lowercase letters, "
            "digits, and underscores."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ToolContext:
    turn_number: int


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    code: str
    summary: str
    data: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, summary: str, data: Mapping[str, Any] | None = None) -> ToolResult:
        return cls(ok=True, code="ok", summary=summary, data=data or {})

    @classmethod
    def failure(
        cls, code: str, summary: str, data: Mapping[str, Any] | None = None
    ) -> ToolResult:
        return cls(ok=False, code=code, summary=summary, data=data or {})


class Tool:
    """A named structured-input callable with a predictable result contract."""

    __slots__ = (
        "_description",
        "_execute",
        "_input_schema",
        "_name",
        "_safety_class",
    )

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Mapping[str, Any],
        execute: ToolExecutor,
        safety_class: str = "read_only",
    ) -> None:
        self._name = normalize_tool_name(name)
        if not isinstance(description, str) or not description.strip():
            raise ToolDefinitionError("Tool description must be a nonblank string.")
        if not callable(execute):
            raise ToolDefinitionError("Tool execution callable must be callable.")
        self._description = description.strip()
        self._input_schema = _validate_schema_definition(input_schema)
        self._execute = execute
        if safety_class not in {"read_only", "compute", "playback"}:
            raise ToolDefinitionError("Tool safety class is not supported.")
        self._safety_class = safety_class

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._input_schema)

    @property
    def safety_class(self) -> str:
        return self._safety_class

    def execute(self, arguments: ToolArguments, context: ToolContext) -> ToolResult:
        return self._execute(arguments, context)

    def model_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _validate_schema_definition(schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ToolDefinitionError("Tool input schema must describe an object.")
    properties = schema.get("properties")
    required = schema.get("required", [])
    if not isinstance(properties, Mapping):
        raise ToolDefinitionError("Tool input schema properties must be an object.")
    if not isinstance(required, list) or not all(
        isinstance(name, str) for name in required
    ):
        raise ToolDefinitionError("Tool input schema required must be a list of names.")
    if not set(required).issubset(properties):
        raise ToolDefinitionError("Required arguments must be declared in properties.")
    if schema.get("additionalProperties") is not False:
        raise ToolDefinitionError(
            "Tool input schema must reject additional properties."
        )
    for name, property_schema in properties.items():
        if not isinstance(name, str) or not isinstance(property_schema, Mapping):
            raise ToolDefinitionError("Every tool property needs a named schema.")
        if property_schema.get("type") not in {
            "array",
            "boolean",
            "integer",
            "number",
            "object",
            "string",
        }:
            raise ToolDefinitionError(
                f"Tool property '{name}' has an unsupported type."
            )
    return copy.deepcopy(dict(schema))
