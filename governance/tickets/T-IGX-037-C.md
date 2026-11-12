# T-IGX-037-C — SCENARIO-FACTORY-X P2: gap analysis, proposed architecture, ADRs and contract designs (docs only)

## Scope

Publish ADR-SFX-001..006 (all Proposed), the SFX contract schema drafts, the canonicalization and metric specifications, and the consolidated gap analysis (master plan R1, units SFX-P2-01..02, wave W1).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-C.md`.

## Plan digest

Plan digest: d52a04a1f8023063007dec4c3a814d2e699a20c2955772b91021bcd5053f7c47

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-C.md`](plans/PLAN_T-IGX-037-C.md).

## Approved

Approved: operator-delegate @ 2026-11-12T13:53:13+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

## Test contract

```text
uv run python scripts/maintenance/check_harness_adr.py
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- ADR acceptance (OD-SFX-06); implementing schemas as code (SFX-P5+).
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

plan_approved

## Lease holder

Worktree `_wt/intergrax/sfx-w1-arch`, branch `sfx-w1-arch` (stacked on `sfx-p0-registration`).
