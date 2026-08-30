from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from typing import Any, TextIO

from zordon.agent import Agent
from zordon.audit import default_audit_log
from zordon.commands import CommandContext, CommandRouter
from zordon.config import ConfigurationError, load_config, load_dotenv, load_settings, state_root
from zordon.debug_logging import DebugLogger
from zordon.memory import default_memory_store
from zordon.openai_provider import OpenAIProvider as ToolOpenAIProvider
from zordon.providers.base import ProviderError
from zordon.providers.openai_provider import OpenAIProvider as StreamingOpenAIProvider

_EXIT_COMMANDS = {"/exit", "exit", "quit"}


def run(
    agent: Any,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    router: CommandRouter | None = None,
) -> int:
    try:
        output.write("Zordon online. Type /exit when you're finished.\n")
        output.flush()

        while True:
            try:
                raw_text = input_fn("You: ")
            except EOFError:
                output.write("\nZordon offline.\n")
                output.flush()
                return 0

            user_text = raw_text.strip()
            if not user_text:
                continue
            if user_text.lower() in _EXIT_COMMANDS:
                output.write("Zordon offline.\n")
                output.flush()
                return 0
            if router is not None:
                command = router.handle(user_text)
                if command.handled:
                    if command.output:
                        output.write(f"Zordon: {command.output}\n")
                    if command.should_exit:
                        output.write("Zordon offline.\n")
                        output.flush()
                        return 0
                    output.flush()
                    continue

            output.write("Zordon: ")
            output.flush()
            printed_chunk = False

            try:
                for chunk in _reply_stream(agent, user_text):
                    output.write(chunk)
                    output.flush()
                    printed_chunk = True
            except ProviderError as exc:
                if printed_chunk:
                    output.write(f"\n[Reply incomplete: {exc}]\n")
                else:
                    output.write(f"[Unable to answer: {exc}]\n")
                output.flush()
                continue

            output.write("\n")
            output.flush()
    except KeyboardInterrupt:
        output.write("\nZordon offline.\n")
        output.flush()
        return 0


def _reply_stream(agent: Any, user_text: str) -> Iterable[str]:
    if getattr(agent, "_simple_mode", False):
        return agent.stream_turn(user_text)
    return agent.respond(user_text)


def main() -> int:
    try:
        load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    load_dotenv()
    config = load_config()
    runtime_root = state_root()
    audit = default_audit_log(runtime_root)
    memory = default_memory_store(runtime_root)
    agent = Agent(
        config=config,
        provider=ToolOpenAIProvider(config),
        audit=audit,
        memory=memory,
    )
    router = CommandRouter(
        CommandContext(
            config=config,
            agent=agent,
            memory=memory,
            audit=audit,
            root=runtime_root,
        )
    )

    return run(agent, router=router)


def simple_agent(settings) -> Agent:
    provider = StreamingOpenAIProvider(
        api_key=settings.api_key,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
        reasoning_effort=settings.reasoning_effort,
    )
    return Agent(
        provider,
        history_message_limit=settings.history_message_limit,
        output_token_limit=settings.output_token_limit,
        debug_logger=DebugLogger(settings.debug_log_path),
    )


if __name__ == "__main__":
    raise SystemExit(main())
