# Zordon Tier 1 Design

**Date:** 2026-07-31
**Status:** Approved
**Scope:** Tier 1 only — the streaming, text-based brain

## Objective

Build the smallest dependable Zordon that can run in a terminal, accept typed
messages, stream coherent replies, and remember earlier turns during the same
process. Tier 1 deliberately excludes tools, audio, durable memory, scheduling,
and consequential actions.

Tier 1 is complete only after automated verification succeeds and the user
personally holds a short multi-turn conversation that demonstrates in-session
recall.

## Chosen approach

Use a small modular Python package rather than a single script or an agent
framework.

A single script would be quick initially but would tangle provider calls,
conversation state, terminal input, and error handling before later tiers
arrive. A large agent framework would add abstractions that are unnecessary for
the first tier and make the core behavior harder to inspect. The modular
package keeps the harness readable while establishing stable seams for later
tiers.

## Runtime and dependency constraints

- Python 3.12 is the supported runtime.
- Windows PowerShell is the primary command environment.
- The package must remain compatible with a later Raspberry Pi deployment.
- Use the official OpenAI Python SDK for the initial provider.
- Use GPT-5.6 Terra as the default model.
- The model identifier must be configurable rather than embedded throughout
  the code.
- Do not introduce an agent framework, database, audio dependency, or web
  server in Tier 1.
- Development dependencies may include pytest and narrowly scoped test tools.

## Project structure

```text
Zordon/
├── AGENT.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── src/
│   └── zordon/
│       ├── __init__.py
│       ├── agent.py
│       ├── cli.py
│       ├── config.py
│       ├── messages.py
│       └── providers/
│           ├── __init__.py
│           ├── base.py
│           └── openai_provider.py
└── tests/
    ├── test_agent.py
    ├── test_cli.py
    ├── test_config.py
    └── test_openai_provider.py
```

Each file has one responsibility:

- `agent.py` owns the system prompt, successful conversation history, and one
  shared typed-turn entry point.
- `messages.py` owns small provider-neutral conversation types.
- `providers/base.py` defines the streaming provider contract.
- `providers/openai_provider.py` is the only module that imports or understands
  the OpenAI SDK.
- `config.py` reads and validates runtime configuration.
- `cli.py` handles terminal input, streamed printing, friendly failures, and
  clean shutdown.

## Core interfaces

The provider contract accepts a system prompt and an ordered, provider-neutral
list of conversation messages. It returns an iterator of text chunks. The agent
core must not receive OpenAI response objects or event types.

The agent exposes one operation for a user turn. It builds a candidate history,
passes that history to the provider, yields reply chunks as they arrive, and
collects the complete reply. The user message and assistant reply are committed
to history together only after streaming finishes successfully.

This transaction-like behavior matters: if the provider fails halfway through
a response, Tier 1 reports the failure but does not save a truncated assistant
message or an unmatched user message into the durable in-process history.

## System prompt

The system prompt identifies the assistant as Zordon and states its present
purpose. It directs Zordon to sound like a cool mentor: calm, capable,
realistic, concise, and plain-spoken. It must not claim capabilities that Tier 1
does not have.

The prompt is application-owned and sent separately from conversation history.
It is never editable through ordinary conversation input.

## Conversation flow

1. The terminal displays a short Zordon greeting and a clear input prompt.
2. The user types a non-empty message.
3. The CLI sends the message to the shared agent entry point.
4. The agent constructs candidate history from prior successful turns plus the
   new user message.
5. The provider streams reply chunks.
6. The CLI prints each chunk immediately without waiting for the complete
   response.
7. After a successful stream, the agent commits both sides of the turn to
   in-memory history.
8. The CLI displays the next input prompt.

Blank input is ignored. A documented exit command and `Ctrl+C` both stop the
program cleanly without a traceback.

## Configuration and secrets

Tier 1 reads these values from environment variables:

- `OPENAI_API_KEY` — required for a live conversation;
- `ZORDON_MODEL` — optional, defaulting to `gpt-5.6-terra`;
- `ZORDON_REQUEST_TIMEOUT_SECONDS` — optional positive number with a safe
  default.

The real API key is never stored in a tracked file. `.env.example` contains
only placeholder values and documents the variable names. `.gitignore`
excludes `.env`, virtual environments, Python caches, test caches, coverage
outputs, build outputs, and local logs.

The application may support loading a local `.env` file for Windows
convenience, but environment variables remain the source seen by the
configuration layer.

## Provider behavior

The OpenAI provider uses the official Responses API with streaming enabled. It
translates provider-neutral messages immediately before the SDK call and emits
only user-visible text deltas.

The adapter owns provider-specific configuration, event filtering, timeouts,
and exception translation. Later providers must be replaceable by implementing
the same contract; no change to `agent.py` or `cli.py` should be necessary.

No automatic model fallback is included in Tier 1. Silently switching models
would make behavior and costs harder to reason about during foundational
testing.

## Error handling

Configuration errors produce a short actionable message and a non-zero process
exit without exposing secret values.

Provider authentication failures, rate limits, timeouts, connection failures,
and malformed provider responses are translated into a small application-level
error type. The terminal prints a calm explanation and returns to the next
prompt when recovery is possible. It does not display a Python traceback during
ordinary failures.

Unexpected programmer errors are not silently swallowed in the core. Tests and
development execution must still make defects diagnosable.

If streaming fails after printing partial text, the CLI clearly labels the
reply as incomplete. The failed turn is not committed to conversation history.

## Testing strategy

Implementation follows red-green-refactor. Production behavior is introduced
only after a test has failed for the expected reason.

Automated tests cover:

- the system prompt contains Zordon's name, purpose, and mentor tone;
- prior successful turns are sent back to the provider on later turns;
- reply chunks are yielded in their original order;
- a completed reply is committed to history;
- a failed or interrupted stream does not commit a partial turn;
- configuration applies defaults and rejects missing or invalid values;
- the OpenAI adapter maps provider-neutral messages and filters streaming
  events correctly;
- the CLI ignores blank input, exits cleanly, streams output immediately, and
  recovers from an expected provider failure;
- source code and fixtures contain no real API key.

Agent and CLI tests use deterministic fake providers. Provider-adapter tests
mock only the SDK boundary and never call the live API. Automated verification
therefore spends no API credits.

## Tier 1 verification

Automated verification must include:

1. Install the project in a clean Python 3.12 virtual environment.
2. Run the entire pytest suite and confirm zero failures.
3. Run a package/import check.
4. Confirm the repository contains no tracked `.env` file or obvious API-key
   pattern.

The user's live verification is:

1. Start Zordon from PowerShell.
2. Tell Zordon a temporary fact, such as a favorite color.
3. Discuss another subject for at least two turns.
4. Ask Zordon to recall the temporary fact.
5. Confirm the reply streams rather than appearing all at once.
6. Stop and restart Zordon.
7. Confirm it no longer remembers the fact, which is expected in Tier 1.
8. Temporarily remove network access or use an invalid configuration and
   confirm Zordon reports the problem without crashing into a traceback.

Tier 2 must not begin until these checks pass and the user explicitly approves
Tier 1.

## Deferred work

The following are intentionally deferred:

- tool schemas, tool execution, and multi-tool loops;
- reminders, project integrations, and note/file retrieval;
- microphone capture, transcription, speech synthesis, interruption, and wake
  words;
- cross-restart memory;
- heartbeat scheduling and proactive notices;
- confirmation gates and the final configuration/audit layer.

Deferring them is a correctness requirement, not an omission. Tier 1 proves the
brain independently before any other moving part is attached.

## Design self-review

- No placeholders or unresolved decisions remain.
- The selected architecture matches the text-first requirement.
- Provider-specific behavior is isolated to one adapter.
- Failure semantics prevent corrupted conversation history.
- All Tier 1 requirements have corresponding automated or live verification.
- Later tiers have clear extension seams but no premature implementation.
