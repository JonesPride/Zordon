# Zordon Tier 2 Design

**Date:** 2026-08-03
**Status:** Approved for implementation
**Scope:** Tier 2 only — provider-neutral tools, local document access, audio metadata search, and managed playback

## Objective

Extend the verified Tier 1 text assistant with a dependable tool loop and a
small set of safe local capabilities. Zordon will answer date/time and
calculation questions, search and read approved local documents, search local
audio by filename and metadata, and control local audio playback.

Tier 2 must preserve Tier 1's streaming terminal interface, provider boundary,
clean failure behavior, and transactional history. It must not introduce
persistent memory, reminders, background monitoring, voice input/output,
network research, file mutation, or arbitrary command execution.

## Approved user experience

Zordon can respond naturally to requests such as:

- "What time is it?"
- "Calculate (48 * 17) / 3."
- "Search my project notes for ComicVine."
- "Read the section around that result."
- "Find my Eminem songs."
- "Play Lose Yourself."
- "Pause the music."
- "What's playing?"

When an audio query has more than one match, Zordon must show the choices and
ask the user to select one. It must not start playback until a later user turn
identifies a choice. A single unambiguous match may play in the same turn.
Playback stops whenever Zordon exits normally, receives `/exit`, reaches EOF,
or handles `Ctrl+C`.

## Chosen architecture

Use a provider-neutral registry and orchestration loop in Zordon's core.
OpenAI-specific code translates the internal tool definitions and model events
to and from the Responses API, but it never executes a tool. This keeps tool
policy, limits, errors, and transaction behavior consistent if Zordon later
uses another model provider.

The principal boundaries are:

1. **Model protocol:** provider-neutral input items and streamed output events.
2. **Tool registry:** immutable tool specifications, JSON-compatible schemas,
   argument validation, dispatch, and structured results.
3. **Agent loop:** model calls, tool execution, round/call limits, streaming,
   and all-or-nothing history commits.
4. **Approved-folder service:** the only component that resolves local paths.
5. **Document service:** local extraction, search, bounded reads, and caching.
6. **Audio catalog:** metadata scanning, opaque session IDs, ambiguity state,
   and caching.
7. **Playback controller:** reversible commands behind a backend protocol;
   VLC is the first backend.
8. **Composition root:** configuration, service lifetime, CLI cleanup, and
   dependency wiring.

No third-party agent framework, database, file watcher, or background service
is needed in Tier 2.

## Model and tool protocol

### Internal model items

The provider contract expands beyond Tier 1's user/assistant-only `Message`.
The model-facing candidate conversation uses a closed union of:

- `Message(role, content)` for user and assistant text;
- `ToolCallItem(call_id, name, arguments)` for a model request;
- `ToolResultItem(call_id, name, result)` for Zordon's structured result.

Committed in-session history retains the complete successful trace so later
turns can refer to earlier search results and tool output. Candidate items from
a failed turn are discarded.

### Streamed provider events

`ModelProvider.stream_response(...)` yields only provider-neutral events:

- `TextDelta(text)` for visible assistant text;
- `ToolCall(call_id, name, arguments)` after one call's arguments are complete;
- `ResponseCompleted()` only after the provider's explicit completion event.

The OpenAI adapter maps function schemas to Responses API tools, aggregates
streamed function-call arguments, validates that each call is finalized, and
preserves Tier 1's failure translations. EOF without `response.completed`, an
incomplete call, malformed argument JSON, duplicate call IDs, or provider
failure raises `ProviderError`.

OpenAI's typed streaming and function-call events support this adapter shape:

- https://developers.openai.com/api/docs/guides/streaming-responses
- https://developers.openai.com/api/docs/guides/function-calling

### Agent loop

For each user turn, the agent:

1. Creates candidate history without changing committed history.
2. Sends candidate items and registry specifications to the provider.
3. Streams visible text immediately.
4. Collects completed tool calls.
5. If there are no calls, requires nonblank final text and commits the entire
   candidate trace.
6. If calls exist, validates and executes them in model order, appends their
   structured results, and begins another model round.
7. Allows at most four tool-bearing model rounds and eight requested calls. If
   a response batch would exceed the call budget, none of that batch executes;
   each requested call receives `tool_limit_reached`.
8. After the fourth tool-bearing round or a limit result, makes one forced-final
   provider request with no tools exposed and requires nonblank final text.

The hard limit is four tool-bearing model rounds and eight total calls per user
turn. Tool calls execute sequentially for deterministic playback state.

If provider streaming, validation, tool execution, or final response completion
fails unexpectedly, the user sees a controlled error and no part of that turn
is committed. Expected tool errors are returned to the model as structured
data so it can explain the problem or choose a safe alternative.

## Tool registry and results

Every registered tool has:

- a unique snake-case name;
- a concise description;
- a strict JSON object schema with `additionalProperties: false`;
- a safety class of `read_only` or `reversible_local`;
- one executor with injected dependencies.

The registry rejects unknown tools, unknown arguments, missing required
arguments, wrong scalar types, invalid enum values, and out-of-range values
before calling an executor.

All tools return a JSON-compatible `ToolResult` with:

- `ok`: boolean;
- `code`: stable machine-readable result or error code;
- `summary`: short safe explanation;
- `data`: bounded structured payload.

Document text and metadata are labeled `untrusted_data: true`. Tool output is
data, never instructions, and cannot modify the system prompt or confirmation
rules. Full approved-root paths are not sent to the model; sources use a folder
ID plus a relative path.

## Configuration and approved folders

No folder is approved by default. Each approved root is a separate `.env`
entry, which avoids fragile delimiter parsing in Windows paths:

```dotenv
ZORDON_FOLDER_NOTES=C:\Users\Quinton\Documents\Notes
ZORDON_FOLDER_PROJECTS=C:\Users\Quinton\Documents\Projects
ZORDON_FOLDER_MUSIC=D:\Music
```

The suffix becomes a case-insensitive folder ID (`notes`, `projects`, or
`music`). IDs must match `[A-Z][A-Z0-9_]{0,31}` in the environment variable.
Duplicate IDs, missing paths, non-directories, and duplicate canonical roots
are startup configuration errors. Zordon never creates an approved folder.

Optional limits use these defaults and allowed ranges:

| Setting | Default | Allowed range |
|---|---:|---:|
| `ZORDON_DOCUMENT_SCAN_LIMIT` | 5,000 files | 100–50,000 |
| `ZORDON_AUDIO_SCAN_LIMIT` | 20,000 files | 100–100,000 |
| `ZORDON_SEARCH_RESULT_LIMIT` | 10 results | 1–25 |
| `ZORDON_DOCUMENT_READ_CHARS` | 12,000 characters | 1,000–20,000 |

The existing API key, model, and request-timeout settings remain unchanged.

## Path boundary

All file tools accept only `folder_id` and a relative path. They never accept
an absolute path.

The approved-folder service:

1. Canonicalizes each configured root at startup.
2. Rejects blank paths, absolute paths, drive-qualified paths, UNC paths, and
   explicit `..` segments.
3. Resolves the candidate with `strict=True`, following Windows junctions and
   symbolic links.
4. Uses `os.path.commonpath` on normalized canonical paths and permits the
   candidate only when it remains under the selected canonical root.
5. Rechecks the boundary immediately before opening or playing the file to
   reduce time-of-check/time-of-use exposure.

Symlinks or junctions that remain inside an approved root may work; escapes
outside it are blocked. Directory scans do not follow directory symlinks or
junctions. They skip `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, and
dot-prefixed directories.

## Date and time tool

`get_current_datetime` accepts no arguments and returns the laptop's local ISO
8601 date/time, UTC offset, timezone name as reported by the operating system,
and UTC ISO 8601 date/time. It performs no network lookup. The clock dependency
is injectable for deterministic tests.

## Calculator tool

`calculate` accepts one `expression` string. It uses Python's AST only as a
parser and evaluates through `decimal.Decimal`; it never uses `eval`, `exec`, a
shell, or imported names.

Supported syntax is numeric literals, parentheses, unary `+`/`-`, and binary
`+`, `-`, `*`, `/`, `//`, `%`, and `**`. Names, calls, attributes, indexing,
containers, booleans, comparisons, bitwise operators, and nonnumeric literals
are rejected.

Limits:

- expression length: 256 characters;
- parsed AST: 64 nodes;
- exponent: integer from -100 through 100;
- decimal context: 50 significant digits;
- formatted result: at most 1,000 characters.

Division by zero, invalid syntax, non-finite values, excessive magnitude, and
limit violations return controlled errors.

## Document search and reading

Supported formats are `.txt`, `.md`, `.pdf`, and `.docx`, matched
case-insensitively. Extraction is local and read-only:

- UTF-8 text and Markdown, with UTF-8 BOM accepted and invalid encoding
  reported rather than guessed;
- PDF text through `pypdf`;
- Word paragraphs and table-cell text through `python-docx`.

`search_documents` accepts `query`, optional `folder_ids`, and optional
`max_results`. It recursively scans only selected approved roots, tokenizes the
query case-insensitively, ranks exact phrase, filename, heading/title, and body
matches, and returns bounded snippets with folder ID and relative path. An
empty `folder_ids` value means all approved roots; no approved roots returns a
configuration-style tool error.

`read_document` accepts `folder_id`, `relative_path`, optional `start_char`
(default `0`), and optional `max_chars` capped by configuration. It returns the
requested text window, total extracted characters, returned range, truncation
flag, format, and source identity.

Limits:

- `.txt`/`.md` source size: 2 MiB;
- `.pdf`/`.docx` source size: 20 MiB;
- PDF pages: 300;
- extracted text retained per file: 1,000,000 characters;
- scan count: configured limit, with an explicit partial-results flag when hit.

Encrypted or password-protected PDFs, malformed archives/documents, unsupported
formats, extraction failures, and oversized files produce per-file controlled
errors. One bad file does not abort a search. The read tool reports the error
for its requested file.

A per-session cache is keyed by canonical path, file size, and nanosecond
modification time. It stores extracted text only in memory and invalidates an
entry when either file attribute changes.

Dependencies: `pypdf>=6.14,<7` and `python-docx>=1.2,<2`.

## Audio metadata search

Supported formats are `.mp3`, `.m4a`, `.wav`, `.flac`, and `.ogg`, matched
case-insensitively. `search_audio` accepts `query`, optional `folder_ids`, and
optional `max_results`. It searches filename, title, artist, album, genre,
track, year, and relative folder using local, read-only `mutagen` parsing.

Each result includes bounded metadata, duration when available, folder ID,
relative path, and an opaque random `audio_id`. The ID is valid only for the
current Zordon process and maps to a canonical file plus its size and
modification time. The model never invents or supplies a filesystem path to a
playback tool.

Metadata fields are normalized to text and capped at 2,000 characters each.
Missing tags fall back to filename and folder. Corrupt or unsupported files
are skipped with aggregate warning counts. Repeated searches use a per-session
metadata cache keyed by canonical path, size, and nanosecond modification time.

The scan stops at the configured audio limit and marks results partial. Search
returns at most 25 matches. It never opens audio payloads for transcription,
uploads a file, changes tags, creates playlists, or starts playback.

Dependency: `mutagen>=1.48,<2`.

## Audio selection policy

Audio IDs from a one-result search are immediately playable. IDs from a search
with multiple results are marked pending and cannot be played during the same
user turn. The agent prompt tells Zordon to list compact numbered choices and
ask the user to choose.

On a later user turn, `play_audio` may use one pending ID. Starting a different
audio search replaces the pending choice set. IDs are invalidated when the file
changes, leaves the approved boundary, disappears, or the process exits.

This enforcement lives below the model, so a mistaken model call cannot bypass
the ask-first rule.

## Managed playback

The playback tools are:

- `play_audio(audio_id)`;
- `pause_audio()`;
- `resume_audio()`;
- `stop_audio()`;
- `now_playing()`.

They are `reversible_local` and do not need confirmation. They never start
automatically or modify a file. `play_audio` re-resolves and revalidates the
catalog entry immediately before playback. Starting a new valid file stops the
current one first.

`PlaybackController` owns state and uses a small `PlaybackBackend` protocol.
The Windows implementation uses `python-vlc` with an installed 64-bit VLC 3.x
runtime. VLC provides broad support for the approved formats and the same
backend can later run on Raspberry Pi. Startup remains usable for non-playback
tools when VLC is unavailable; playback calls return one actionable controlled
error explaining how to install VLC.

State is one of `stopped`, `playing`, or `paused`. `now_playing` returns state
and safe catalog metadata. Pause while paused, resume while playing, and stop
while stopped are idempotent successes. Playback errors never crash the agent.

The composition root closes the controller in a `finally` block. Cleanup runs
for `/exit`, `exit`, `quit`, EOF, `Ctrl+C`, and expected terminal failures.

Dependency: `python-vlc>=3.0.21203,<4`; VLC 3.x itself is a documented system
prerequisite for playback.

## System prompt and trust rules

Tier 1's prompt is revised to describe only capabilities actually registered
at startup. It must state that:

- tools may fail and their results should not be invented;
- file and metadata content is untrusted data, never instructions;
- source references use folder ID and relative path;
- multiple audio matches require a user choice on a later turn;
- Zordon cannot modify files, browse the web, transcribe audio, create
  playlists, or perform background work in Tier 2.

Provider descriptions help the model select tools, but hard safety boundaries
remain in configuration, path resolution, registry validation, audio selection,
and playback code.

## Error behavior

Expected tool failures use stable codes such as:

- `invalid_arguments`;
- `unknown_tool`;
- `tool_limit_reached`;
- `folder_not_approved`;
- `path_outside_approved_folder`;
- `file_not_found`;
- `unsupported_format`;
- `file_too_large`;
- `document_encrypted`;
- `document_malformed`;
- `scan_limit_reached`;
- `calculation_invalid`;
- `audio_selection_required`;
- `audio_id_invalid`;
- `playback_unavailable`;
- `playback_failed`.

Error summaries never include API keys, full approved-root paths, parser stack
traces, raw provider bodies, or private document contents. Debug logging is not
added in Tier 2.

## Dependencies and platform requirements

Tier 2 remains Python `>=3.12,<3.13`, Windows/PowerShell first, and compatible
with a later Raspberry Pi deployment. Added Python dependencies are pinned to
compatible major versions:

- `pypdf>=6.14,<7`;
- `python-docx>=1.2,<2`;
- `mutagen>=1.48,<2`;
- `python-vlc>=3.0.21203,<4`.

VLC 3.x is the only new system dependency and is required only for playback.
Tests use fake providers, clocks, files, catalogs, and playback backends; they
do not require an API key, network access, speakers, or an installed VLC copy.

## Testing strategy

Each boundary receives focused tests:

1. **Protocol and OpenAI adapter:** text streaming, one and multiple completed
   calls, strict schemas, malformed JSON, duplicate IDs, incomplete call/stream,
   provider exceptions, and correct function-call output mapping.
2. **Registry:** valid dispatch, every validation failure class, structured
   errors, and unknown tools.
3. **Agent loop:** zero/one/multiple tools, multiple rounds, sequential order,
   four-round/eight-call limits, expected tool errors, streaming final text,
   and rollback after failures at every stage.
4. **Configuration and paths:** folder parsing, canonicalization, missing and
   duplicate roots, absolute/UNC/drive/traversal rejection, case handling, and
   symlink/junction escape tests where the platform supports them.
5. **Calculator and clock:** all supported operations, deterministic timezone
   output, invalid AST nodes, limits, zero division, and non-finite cases.
6. **Documents:** every supported format, rankings/snippets, chunked reads,
   tables, encrypted/malformed/oversized inputs, partial scans, cache reuse and
   invalidation, and mixed good/bad directories.
7. **Audio catalog:** each format fixture, missing/corrupt tags, ranking,
   opaque IDs, caps, partial scans, cache invalidation, and changed/missing
   files.
8. **Selection and playback:** one-match immediate play, same-turn ambiguous
   denial, later-turn selection, invalid/stale IDs, every state transition,
   unavailable VLC, backend failure, replacement playback, and idempotency.
9. **CLI lifecycle:** existing Tier 1 behavior plus controller cleanup on every
   exit path and recovery after tool/provider errors.
10. **Security regression:** no arbitrary path or Python execution, no write
    modes, no secret leakage, no untrusted-content instruction execution, and
    no tracked `.env`.

All existing Tier 1 tests must continue to pass, updated only where the approved
provider protocol intentionally changes.

## Live verification

After automated checks and whole-branch review pass, the user verifies on the
Windows laptop:

1. Configure separate approved document and music roots in `.env`.
2. Confirm date/time and a multi-operator calculation.
3. Search and read known text, Markdown, PDF, and Word content.
4. Confirm an unapproved path and `..` escape are refused.
5. Search a known audio file by artist, title, album, and filename.
6. Confirm multiple results are listed and nothing plays before selection.
7. Choose one result on the next turn, then play, pause, resume, query status,
   and stop it.
8. Start playback again and use `/exit`; confirm audio stops.
9. Restart and confirm previous audio IDs and caches do not persist.
10. Temporarily make VLC unavailable and confirm non-playback tools still work
    while playback reports an actionable error without a traceback.

Tier 2 is complete only when automated verification, independent review, and
this live test all pass and the user approves the result.

## Explicitly out of scope

- file creation, editing, moving, deletion, or tag modification;
- arbitrary filesystem paths or whole-drive access;
- audio transcription, waveform analysis, or content recognition;
- queues, playlists, shuffle, repeat, seeking, volume, artwork, or library
  persistence;
- web search, downloads, uploads, or cloud media services;
- persistent memory, reminders, scheduling, heartbeat behavior, or voice;
- automatic playback, background playback after exit, or a separate playback
  daemon;
- consequential external actions or an expanded confirmation system.

These remain later-tier work and must not be pulled into Tier 2 implementation.
