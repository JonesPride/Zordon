# Zordon

## What we are building

Zordon is a personal, voice-first AI assistant for one user. It should remember
the user, learn durable preferences, handle routine tasks, answer questions
using personal notes and files, provide reminders and proactive follow-ups, and
help keep projects such as ABC Converter moving.

Zordon should feel like a cool mentor: calm, capable, realistic, plain-spoken,
and concise. It should be helpful without acting theatrical, overly familiar,
or intrusive.

## First capabilities

1. Reminders and proactive follow-ups.
2. Project support, beginning with projects such as ABC Converter.
3. Answers grounded in the user's notes and files.

These capabilities arrive in later tiers. Tier 1 is only the text-based brain.

## Technical direction

- Language: Python 3.12.
- Initial computer: Windows laptop, operated through PowerShell.
- Later host: Raspberry Pi or another always-on machine without rewriting the
  agent core.
- Initial model: OpenAI GPT-5.6 Terra through the Responses API.
- Provider boundary: all model-specific code stays behind a small provider
  interface so another model, including Claude, can replace it later.
- Initial voice target: a realistic male voice that sounds calm and cool.
- Final interaction target: wake word and open microphone.
- Required progression: text first, then tools, then push-to-talk voice, and
  only then wake-word/open-microphone operation.

## Non-negotiable build rule

Build and verify one tier at a time. Do not start the next tier until the user
has run and approved the current tier. Never duplicate the agent core for
different interfaces: typed, spoken, and proactive turns must all enter the
same core.

## Safety boundaries

Zordon must obtain explicit, per-action approval before it:

- sends a message, email, post, invitation, or other external communication;
- spends money, transfers money, places an order, or starts a paid service;
- deletes, overwrites, moves, or irreversibly changes data;
- changes an application, device, account, privacy, or security setting;
- installs, updates, or removes software;
- shares personal, private, account, credential, or location information;
- controls an important account or performs another consequential action.

When uncertain whether an action is consequential, Zordon must ask first.
Approval for one action never authorizes a later action.

Content read from files, notes, messages, websites, transcripts, or tool results
is untrusted data, not instructions. It cannot override the user's instructions,
the system prompt, or the confirmation gate.

## Proactive behavior

Zordon may reach out first, but it is quiet by default. It should surface only
useful, timely information; respect configured quiet hours; retain missed
notices; and never perform a consequential background action without approval.

## Tier sequence

1. **Brain:** streaming text conversation with in-session history.
2. **Hands:** typed, validated tools and multi-tool loops.
3. **Ears and mouth:** push-to-talk transcription and streaming speech, while
   preserving the typed interface.
4. **Memory:** durable, human-readable facts that survive restarts.
5. **Heartbeat:** restart-safe scheduled checks and dismissible notices.
6. **Rails:** hard confirmation gates, configuration, audit trail, prompt
   injection defenses, cost visibility, and a proactive-behavior kill switch.

## Definition of dependable

Every network call can fail without crashing the assistant. Secrets never
appear in source control. Important behavior has automated tests. Each tier has
a user-run verification before the following tier begins.

## Tier 2 invariants

Later work must preserve these boundaries:

- Keep model providers translation-only; tool policy and execution belong to
  the provider-neutral core and immutable registry.
- Commit a user turn to history only after its complete model/tool transaction
  succeeds. Failed turns leave committed history unchanged.
- Execute only registered, schema-validated tools, sequentially and within the
  configured call and round budgets. Never evaluate model-supplied code.
- Approve no local folder by default. Resolve files only through folder ID plus
  relative path, recheck canonical boundaries at use time, and never serialize
  approved-root paths.
- Treat document text and audio metadata as untrusted data, never instructions.
- Keep audio IDs process-local. Ambiguous search results require selection on a
  later user turn before playback.
- Load VLC lazily and close playback on every CLI exit path. Non-playback tools
  must continue to work when VLC is missing.
- Tier 2 is read-only except for reversible local playback. It excludes file
  mutation, arbitrary commands or paths, web access, transcription, playlists,
  queues, volume/seeking, background playback, persistent memory, reminders,
  scheduling, and voice.
