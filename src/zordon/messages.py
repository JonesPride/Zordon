from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["user", "assistant"]
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ToolCallItem:
    call_id: str
    name: str
    arguments: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ToolResultItem:
    call_id: str
    name: str
    result: dict[str, JSONValue]


ModelItem = Message | ToolCallItem | ToolResultItem
