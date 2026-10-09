from __future__ import annotations

import os
from importlib.metadata import entry_points
from io import StringIO
from types import SimpleNamespace
from typing import Any

from zordon import cli
from zordon.application import Application
from zordon.messages import Message, ToolCallItem, ToolResultItem


class ScriptedResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            arguments = '{"expression":"2+2"}'
            item = SimpleNamespace(
                type="function_call", id="item_1", call_id="call_1", name="calculate"
            )
            yield SimpleNamespace(type="response.output_item.added", item=item)
            yield SimpleNamespace(
                type="response.function_call_arguments.delta", item_id="item_1", delta=arguments
            )
            yield SimpleNamespace(
                type="response.function_call_arguments.done", item_id="item_1", arguments=arguments
            )
            yield SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(**vars(item), arguments=arguments),
            )
        elif len(self.calls) == 3:
            yield SimpleNamespace(type="response.output_text.delta", delta="partial")
            raise RuntimeError("private provider detail: test-key")
        else:
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta="The answer is 4." if len(self.calls) == 2 else "Recovered.",
            )
        yield SimpleNamespace(type="response.completed")


def test_installed_entry_point_calculates_and_recovers_through_real_application(
    monkeypatch, tmp_path
) -> None:
    for name in tuple(os.environ):
        if name.startswith("ZORDON_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.setattr("zordon.config.load_dotenv", lambda: None)
    debug_log = tmp_path / "debug.jsonl"
    for name, value in {
        "OPENAI_API_KEY": "test-key",
        "ZORDON_MODEL": "test-model",
        "ZORDON_REQUEST_TIMEOUT_SECONDS": "12.5",
        "ZORDON_REASONING_EFFORT": "high",
        "ZORDON_OUTPUT_TOKEN_LIMIT": "777",
        "ZORDON_DEBUG_LOG": str(debug_log),
    }.items():
        monkeypatch.setenv(name, value)

    responses = ScriptedResponses()
    sdk_arguments: dict[str, Any] = {}

    def fake_sdk(**kwargs):
        sdk_arguments.update(kwargs)
        return SimpleNamespace(responses=responses)

    monkeypatch.setattr("zordon.providers.openai_provider.OpenAI", fake_sdk)
    applications: list[Application] = []
    build = cli.build_application

    def capture_application(settings):
        app = build(settings)
        applications.append(app)
        return app

    monkeypatch.setattr(cli, "build_application", capture_application)
    output = StringIO()
    requests = iter(("Calculate 2+2", "Failed request", "Continue", "/exit"))
    real_run = cli.run
    monkeypatch.setattr(
        cli, "run", lambda app: real_run(app, input_fn=lambda _: next(requests), output=output)
    )

    entry = next(ep for ep in entry_points(group="console_scripts") if ep.name == "zordon")
    assert entry.load() is cli.main
    assert entry.load()() == 0

    app = applications[0]
    assert app._closed is True
    assert len(responses.calls) == 4
    assert sdk_arguments == {"api_key": "test-key", "timeout": 12.5}
    for call in responses.calls:
        assert call["model"] == "test-model"
        assert call["max_output_tokens"] == 777
        assert call["reasoning"] == {"effort": "high"}
    returned = responses.calls[1]["input"]
    assert returned[1]["type"] == "function_call"
    assert returned[2]["call_id"] == "call_1"
    assert '"result":"4"' in returned[2]["output"]
    recovery = responses.calls[3]["input"]
    assert all(item.get("content") != "Failed request" for item in recovery)
    assert any(isinstance(item, ToolCallItem) for item in app.agent.history)
    assert any(isinstance(item, ToolResultItem) for item in app.agent.history)
    assert Message("user", "Continue") in app.agent.history
    assert Message("user", "Failed request") not in app.agent.history
    rendered = output.getvalue()
    assert "The answer is 4." in rendered
    assert "Reply incomplete:" in rendered
    assert "Recovered." in rendered
    for secret in ("test-key", "private provider detail", "Calculate 2+2", "The answer is 4."):
        assert secret not in debug_log.read_text()
    assert "test-key" not in rendered
    assert "private provider detail" not in rendered
