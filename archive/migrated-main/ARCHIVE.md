# Inactive migrated assistant reference

This directory preserves source, tests, packaging, and setup documentation from
main at `70db55d565cd011437bcef4b7c1ef97158dfd0fa`, before reconciliation.
It retains voice, memory, heartbeat, command routing, project support, drafting,
and image work for deliberate integration in later tiers.

This is a historical reference, not an alternate runnable assistant. Do not
install it, add it to PYTHONPATH, or import it from the active runtime. Its tests
are outside current Tier 2 validation. Original imports and the compatibility
core are retained unchanged for comparison. Only `src/zordon` is installed.

Active Tier 2 contracts come from feature/tier-2-acceptance at
`a588479b25dcb66f06dbba2c500b36870adf34d1`. Windows setup-python v7,
repository attributes, and the Tier 1 verification record from main are preserved.

When a later feature is approved, port it to the single active Agent and provider
contract with tests. Do not reactivate the archived compatibility agent or copy
its runtime monkey-patching into the active package.
