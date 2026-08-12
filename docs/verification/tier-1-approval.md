# Tier 1 Verification and Approval

**Recorded:** 2026-08-11  
**Baseline:** `main` at `44fe771`  
**Target platform:** Windows, Python 3.12

## Automated baseline

The existing Tier 1 suite passed before hardening: 31 tests, source/test
compilation, and the `zordon.cli` import gate. The hardening pull request adds
a Windows GitHub Actions job that repeats the tests plus Ruff, Pyright,
compilation, and import checks using locked dependencies. Its completed run is
the authoritative Windows automation record.

## Live Windows checklist

These checks require a real API key and human observation, so CI does not
claim them:

- [ ] A temporary fact is remembered after at least two intervening turns.
- [ ] The response appears incrementally while streaming.
- [ ] The temporary fact is forgotten after process restart.
- [ ] A disconnected-network turn reports a safe error without a traceback.
- [ ] Conversation continues successfully after the network is restored.

## Approval

The repository owner directed that Tier 1 approval be recorded and Tier 2
begin behind a separate feature branch and pull request. This authorizes Tier
2 development, while the unchecked live items remain visible acceptance work
and must not be represented as completed.
