from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol, TextIO

from zordon.agent import Agent
from zordon.application import ApplicationError, build_application
from zordon.config import ConfigurationError, load_settings
from zordon.providers.base import ProviderError

_EXIT_COMMANDS = {"/exit", "exit", "quit"}


class ApplicationLike(Protocol):
    agent: Agent

    def close(self) -> None: ...


def run(
    app: ApplicationLike | Agent,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> int:
    agent = app if isinstance(app, Agent) else app.agent
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

            output.write("Zordon: ")
            output.flush()
            printed_chunk = False

            try:
                for chunk in agent.stream_turn(user_text):
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
    finally:
        if not isinstance(app, Agent):
            app.close()


def main() -> int:
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        app = build_application(settings)
    except ApplicationError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2
    return run(app)


if __name__ == "__main__":
    raise SystemExit(main())
