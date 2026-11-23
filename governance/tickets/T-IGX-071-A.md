# T-IGX-071-A — Cursor integrator correction (IGX-CURSOR-INTEGRATOR-CORR-001)

## Scope

Correct three defects in `scripts/cursor/igx_integrator.py` under OD-AF-INT A:

1. **INT-F01** — Documentation PR CI gate profile when GitHub has no required checks configured.
2. **INT-F02** — Deterministic UTF-8 decoding for subprocess output on Windows.
3. **INT-F03** — Idempotent `publish` when a matching open PR already exists (e.g. PR #756).

**Plan:** [`PLAN_T-IGX-071-A.md`](plans/PLAN_T-IGX-071-A.md)

## Plan digest

Plan digest: 09db804be84a502ea1cfb7ae2c973a86199fd824f92ca7a28612c95c0a5f7a2e

## Approved

_(awaiting operator-delegate approval)_

## Lifecycle state

plan_pending

## Test contract

```text
py -3.12 -m pytest tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_cursor_integration_guard.py -q
py -3.12 -m scripts.governance.igx_gov ticket
```

## Out of scope

Claude Code surface; SFX product implementation; canonical SFX R2 adoption; GitHub settings; hook/CI bypass.

## Implementer session

Implementer session: local_c8e4b912-7f3a-4d2e-91b5-6a0c3e8f4d21