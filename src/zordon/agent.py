from __future__ import annotations

from collections.abc import Iterator

from zordon.debug_logging import DebugLogger
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
    def __init__(
        self,
        provider: ModelProvider,
        history_message_limit: int = 40,
        output_token_limit: int = 2048,
        debug_logger: DebugLogger | None = None,
    ) -> None:
        self._provider = provider
        self._history: list[Message] = []
        self._history_message_limit = history_message_limit
        self._output_token_limit = output_token_limit
        self._debug_logger = debug_logger or DebugLogger(None)

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
        self._debug_logger.event("turn_started", message_count=len(candidate_history))

        try:
            for chunk in self._provider.stream_reply(
                SYSTEM_PROMPT,
                candidate_history,
                self._output_token_limit,
            ):
                if not chunk:
                    continue
                reply_parts.append(chunk)
                yield chunk
        except Exception:
            self._debug_logger.event("turn_failed", message_count=len(candidate_history))
            raise

        reply = "".join(reply_parts)
        if not reply.strip():
            raise ProviderError("The model returned no text. Please try again.")

        self._history.extend(
            (
                user_message,
                Message(role="assistant", content=reply),
            )
        )
        if len(self._history) > self._history_message_limit:
            del self._history[: -self._history_message_limit]
        self._debug_logger.event(
            "turn_completed",
            message_count=len(self._history),
            output_chars=len(reply),
        )
