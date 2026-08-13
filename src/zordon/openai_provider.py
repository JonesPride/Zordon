from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Protocol

from zordon.config import Config
from zordon.providers.base import ProviderError


class HistoryTurn(Protocol):
    role: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    call_id: str


@dataclass(frozen=True)
class OpenAIProvider:
    config: Config
    api_url: str = "https://api.openai.com/v1/responses"

    def stream_reply(
        self,
        system_prompt: str,
        input_text: str,
        tools: list[dict] | None = None,
    ) -> Iterable[str]:
        stream = self.stream_step(system_prompt=system_prompt, input_text=input_text, tools=tools)
        while True:
            try:
                yield next(stream)
            except StopIteration:
                break

    def stream_step(
        self,
        system_prompt: str,
        input_text: str,
        tools: list[dict] | None = None,
    ) -> Iterable[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set. Check .env.local.")

        payload = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": input_text,
            "stream": True,
            "store": False,
            "truncation": "auto",
            "text": {"format": {"type": "text"}},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                tool_calls = yield from self._read_sse(response)
                return tool_calls
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"OpenAI request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise ProviderError(f"OpenAI is unreachable right now: {exc}") from exc

    def _read_sse(self, response) -> Iterable[str]:
        tool_calls: list[ToolCall] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type.endswith("output_text.delta") and "delta" in event:
                yield event["delta"]
            elif event_type.endswith("output_item.done"):
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    tool_calls.append(self._parse_tool_call(item))
            elif event_type.endswith("response.failed"):
                error = event.get("response", {}).get("error") or event.get("error")
                raise ProviderError(f"OpenAI response failed: {error}")
        return tool_calls

    def _parse_tool_call(self, item: dict) -> ToolCall:
        raw_arguments = item.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        return ToolCall(
            name=item.get("name", ""),
            arguments=arguments,
            call_id=item.get("call_id", ""),
        )
