# Zordon

Zordon is a personal AI assistant built one independently testable tier at a
time. Tier 1 is a streaming text conversation that remembers successful turns
until the process exits.

Licensed under the MIT License. See `LICENSE`.

## Requirements

- Windows
- PowerShell
- Python 3.12
- An API key from the official [OpenAI API keys page](https://platform.openai.com/api-keys)

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Create the local settings file in PowerShell if it does not already exist,
then open it in Notepad:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

In the text editor, replace the placeholder after `OPENAI_API_KEY=` with the
key from the official page, then save the file. Paste the key only into the
text editor, never into a PowerShell command, so it does not enter PowerShell
command history. Never commit `.env` or share its contents.

Zordon can also read `ZORDON_MODEL`, `ZORDON_REQUEST_TIMEOUT_SECONDS`,
`ZORDON_HISTORY_MESSAGE_LIMIT`, and `ZORDON_OUTPUT_TOKEN_LIMIT` from the
environment. History is bounded to complete conversation turns; the defaults
retain 40 messages and cap each model response at 2,048 output tokens.

For troubleshooting, set `ZORDON_DEBUG_LOG=logs/zordon-debug.jsonl`. This
opt-in log contains timestamps, event names, counts, and durations only. It
does not record prompts, responses, API keys, raw exceptions, or tracebacks.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
zordon
```

Type `/exit` to stop. The text interface remains part of Zordon after voice is
added because it is the dependable debugging and fallback path.

## Automated checks

```powershell
python -m pytest -q
python -m ruff check src tests
python -m pyright
python -m compileall -q src tests
python -c "from zordon.cli import main; print('Zordon import OK')"
```

## Tier 1 live verification

1. Tell Zordon a temporary fact.
2. Talk about something else for at least two turns.
3. Ask Zordon to recall the temporary fact.
4. Confirm the answer appeared incrementally.
5. Exit and restart Zordon.
6. Confirm the temporary fact is forgotten after restart.
7. Disconnect the network for one turn and confirm Zordon reports the problem
   without a traceback, then reconnect and continue.

Tier 2 does not begin until these checks pass and the user approves Tier 1.

The dated automated and live-verification status is recorded in
`docs/verification/tier-1-approval.md`.
