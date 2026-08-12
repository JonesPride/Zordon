# Zordon

Zordon is a personal AI assistant built one independently testable tier at a
time. Tier 2 adds safe local tools for date/time, decimal calculations,
approved documents, audio metadata, and managed playback while retaining the
streaming text conversation and process-local history from Tier 1.

Licensed under the MIT License. See `LICENSE`.

## Requirements

- Windows
- PowerShell
- Python 3.12
- An API key from the official [OpenAI API keys page](https://platform.openai.com/api-keys)
- 64-bit VLC 3.x for audio playback (other tools work without VLC)

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
environment. `ZORDON_REASONING_EFFORT` accepts `low`, `medium`, `high`, or
`xhigh`. History is bounded to complete conversation turns; the defaults retain
40 messages, cap each model response at 2,048 output tokens, and use medium
reasoning effort.

For troubleshooting, set `ZORDON_DEBUG_LOG=logs/zordon-debug.jsonl`. This
opt-in log contains timestamps, event names, counts, and durations only. It
does not record prompts, responses, API keys, raw exceptions, or tracebacks.

## Approve local folders

No local folder is accessible by default. Add a separate setting to `.env` for
each folder Zordon may search. The suffix becomes its case-insensitive folder
ID; Zordon exposes only that ID and a relative path, never the full root path.

```dotenv
ZORDON_FOLDER_NOTES=C:\Users\Quinton\Documents\Notes
ZORDON_FOLDER_PROJECTS=C:\Users\Quinton\Documents\Projects
ZORDON_FOLDER_MUSIC=D:\Music
```

Document extraction is local and read-only for `.txt`, `.md`, `.pdf`, and
`.docx`. Audio metadata is read locally from `.mp3`, `.m4a`, `.wav`, `.flac`,
and `.ogg`. Zordon does not upload, transcribe, edit, move, or delete these
files. File contents and metadata are treated as untrusted data, not commands.

Optional safety limits may also be placed in `.env`:

```dotenv
ZORDON_DOCUMENT_SCAN_LIMIT=5000
ZORDON_AUDIO_SCAN_LIMIT=20000
ZORDON_SEARCH_RESULT_LIMIT=10
ZORDON_DOCUMENT_READ_CHARS=12000
```

Allowed ranges are 100–50,000 document files, 100–100,000 audio files, 1–25
search results, and 1,000–20,000 characters per document read. Scans stop at
their configured limit and report partial results.

## VLC playback setup

Install the 64-bit VLC 3.x desktop application from VideoLAN and make sure its
architecture matches 64-bit Python. The Python package is installed with
Zordon, but VLC itself is a separate system prerequisite. VLC loads lazily:
when it is absent or incompatible, date/time, calculation, and document/audio
search remain available while playback returns a controlled
`playback_unavailable` result without a traceback.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
zordon
```

Type `/exit` to stop. The text interface remains part of Zordon after voice is
added because it is the dependable debugging and fallback path. Natural
requests can play, pause, resume, stop, or report the currently playing track.
When audio search has multiple matches, Zordon lists choices and requires a
selection on a later user turn; it cannot play an ambiguous result immediately.
Playback is stopped and released on exit, EOF, Ctrl+C, or terminal failure.

## Automated checks

```powershell
python -m pytest --cov=zordon --cov-branch --cov-report=term-missing --cov-report=xml
python -m ruff check src tests
python -m ruff format --check src tests
python -m pyright
python -m compileall -q src tests
python -c "from zordon.application import build_application; from zordon.tools.registry import ToolRegistry; print('Zordon imports OK')"
```

The test command prints line and branch coverage and writes `coverage.xml`.
Windows CI uploads that report as the `tier-1-coverage` artifact.

## Tier 1 live verification

1. Tell Zordon a temporary fact.
2. Talk about something else for at least two turns.
3. Ask Zordon to recall the temporary fact.
4. Confirm the answer appeared incrementally.
5. Exit and restart Zordon.
6. Confirm the temporary fact is forgotten after restart.
7. Disconnect the network for one turn and confirm Zordon reports the problem
   without a traceback, then reconnect and continue.

## Tier 2 live verification

Run these checks on the Windows laptop after automated verification passes:

1. Configure separate approved document and music roots in `.env`.
2. Ask for the current date/time and a multi-operator calculation.
3. Search and read known text, Markdown, PDF, and Word content.
4. Confirm an unapproved folder and a `..` traversal attempt are refused.
5. Search known audio by artist, title, album, and filename.
6. Confirm multiple matches are listed and nothing plays before selection.
7. Select one result on the next turn; play, pause, resume, check status, and stop.
8. Start playback again, enter `/exit`, and confirm the audio stops.
9. Restart Zordon and confirm previous audio IDs and caches do not persist.
10. Temporarily make VLC unavailable; confirm non-playback tools still work and playback reports `playback_unavailable` without a traceback.

Tier 2 is accepted only after the automated checks, whole-branch security
review, all ten Windows checks, and user approval pass.

The dated automated and live-verification status is recorded in
`docs/verification/tier-1-approval.md`.
