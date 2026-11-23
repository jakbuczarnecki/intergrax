# T-IGX-072-A — Cursor integrator correction (IGX-DELIVERY-UNBLOCK-001)

## Scope

Correct three defects in `scripts/cursor/igx_integrator.py` under OD-AF-INT A:

1. **INT-F01** — Documentation PR CI gate profile when GitHub has no required checks configured.
2. **INT-F02** — Deterministic UTF-8 decoding for subprocess output on Windows.
3. **INT-F03** — Idempotent `publish` when a matching open PR already exists (e.g. PR #756).

**Plan:** [`PLAN_T-IGX-072-A.md`](plans/PLAN_T-IGX-072-A.md)

Collision resolution: this ticket replaces the integrator campaign's contested `T-IGX-071-A` identifier. EQO-WB02 retains a separate `T-IGX-071-A` scope on the main checkout.

## Plan digest

Plan digest: 00a49d58b53f26b2bc7d806125f311252d66af117102f1fa7628b17536d1b91f

## Approved

Approved: operator-delegate @ 2026-11-23T14:15:00+01:00
Delegation: docs/project/maintainers/qualification/cursor_integration/operator_decisions/DELEGATION-T-IGX-072-A-INT.md@a6cd65b7843e018c9af471dae5bb145c4ad3c785
Delegate session: local_5d8e91a4-7f2c-4b6e-a3d0-1e9f8c7b6a54

## Lifecycle state

plan_approved

## Test contract

```text
uv run pytest tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_claude_surface_unchanged.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Claude Code surface; SFX product implementation; canonical SFX R2 adoption; GitHub settings; hook/CI bypass; IIAC/ARCH-FREEZE evidence repairs.

## Implementer session

Implementer session: local_c8e4b912-7f3a-4d2e-91b5-6a0c3e8f4d21