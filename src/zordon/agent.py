from __future__ import annotations

from collections.abc import Iterator

from zordon.messages import Message
from zordon.providers.base import ModelProvider, ProviderError

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


class Agent:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def stream_turn(self, user_text: str) -> Iterator[str]:
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("A user turn cannot be blank.")

        user_message = Message(role="user", content=clean_text)
        candidate_history = (*self._history, user_message)
        reply_parts: list[str] = []

        for chunk in self._provider.stream_reply(
            SYSTEM_PROMPT,
            candidate_history,
        ):
            if not chunk:
                continue
            reply_parts.append(chunk)
            yield chunk

        reply = "".join(reply_parts)
        if not reply.strip():
            raise ProviderError("The model returned no text. Please try again.")

        self._history.extend(
            (
                user_message,
                Message(role="assistant", content=reply),
            )
        )
