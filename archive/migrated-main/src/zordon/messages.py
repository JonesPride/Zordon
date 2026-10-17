from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
