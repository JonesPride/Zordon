# Tier 2 core integration verification

Recorded: 2026-10-08 (America/Los_Angeles).

Status: local core integration complete; Windows and live acceptance pending.
The owner approved committing this integration, pushing its feature branch,
and opening a draft pull request to run Windows CI on 2026-10-08.
No merge or change to main is authorized.

## Baselines and reconciliation

- Base main: `70db55d565cd011437bcef4b7c1ef97158dfd0fa`.
- Typed acceptance source: `a588479b25dcb66f06dbba2c500b36870adf34d1`.
- Integration branch: `work/tier-2-core-integration`.

The active package adopts the typed acceptance agent, provider, validated tool
registry, approved-folder services, playback lifecycle, and application factory.
These services are included because the application factory composes them; this
does not constitute live acceptance of the document or playback work packages.
The installed CLI now uses the same core and provider contract as its tests.
The migrated main implementation is preserved byte-for-byte under
`archive/migrated-main`, outside installed packages and active test discovery.
Its voice, memory, heartbeat, projects, writing, and image features remain
inactive references for later integration, not supported Tier 2 entry points.

Current main's setup-python v7 and repository attributes are retained. The
existing Tier 1 approval record is unchanged. Broad lint/type suppressions from
the hybrid runtime are removed from the active configuration.

## Fixes beyond the acceptance source

- Convert optional tool properties to required nullable properties in the
  OpenAI strict wire schema. Keep the provider-neutral schema unchanged.
- Treat null optional arguments as omitted before validated execution. Required
  and unknown arguments remain validated and rejected when invalid.
- Convert early application composition failures, including an approved folder
  disappearing after configuration, to a safe startup error.
- Test the installed console entry point with the real application, SDK adapter,
  calculator, structured results, failed response, clean history, recovery, and
  shutdown. Only SDK transport and terminal I/O are substituted.
- Update CI imports and coverage artifact name for the Tier 2 application.
- Explain that tool-returned document excerpts/audio metadata go to the model;
  local extraction does not mean all returned content stays local.

Strict-schema reference:
https://developers.openai.com/api/docs/guides/function-calling

## Local automated results

Environment: Linux, Python 3.12.14, isolated virtual environment.
Locked runtime/development dependencies installed successfully. Editable
installation succeeded using normal build isolation.

- [x] Initial Pytest: 268 passed; 93% combined line/branch coverage.
- [x] Rooted-path fix Pytest: 271 passed; 93% combined coverage.
- [x] Pyright: 0 errors, 0 warnings.
- [x] Source and active test compilation.
- [x] Ruff lint and format checks.
- [x] Application/registry/CLI imports and installed entry-point integration.
- [x] Archived baseline bytes match the pinned main source/tests/setup files.
- [x] Built wheel contains only active zordon modules, no archived code.

The first published run (37885842388) passed installation and 267 tests but
failed the rooted-path test on Windows. The validator now rejects Windows roots
without requiring a drive letter; regression tests also exercise Windows path
semantics on Linux. The follow-up Windows run is the acceptance gate.
Tests in the inactive archive are intentionally excluded; they describe the
retired runtime and are not evidence for the active application.

## Remaining acceptance gates

- [ ] Run all automated checks on Windows Python 3.12 at the proposed commit.
- [ ] Complete the five outstanding Tier 1 live checks in the original record.
- [ ] Review the entire reconciled branch for security and tier scope.
- [ ] Complete all ten Tier 2 live checks in README with a real API key/VLC.
- [ ] Verify a live document-search/read/calculation/final-answer tool chain.
- [ ] Verify network failure and recovery on the integrated application.
- [ ] Review Windows junction/race behavior and document extraction limits.
- [ ] Verify end-of-file/asynchronous VLC status behavior.
- [ ] Verify playback state if a following model round fails: history rollback
  does not undo local side effects, and retry must not replay them automatically.
- [ ] Obtain owner acceptance before Tier 3 begins.

No real model requests, user documents, Windows machine, microphone, or audio
hardware were used in local verification. The default model remains the
repository's configured baseline; its live availability is not claimed here.

## Review boundary

This is a reviewable implementation authorized for a draft pull request and
Windows CI. Merge remains pending explicit user approval. Existing PR #8 and
#9 and dependency upgrade PRs are outside this change. Windows CI is followed
by live acceptance, not activating voice or memory.
