# T-IGX-062-B — ENT-AUDIT-X sandbox `uv` executable resolution (EQO Wave 04 / P04)

## Scope

Resolve the `uv` executable before sandbox subprocess launch so Windows hosts where `uv` is not on PATH (but available via `shutil.which` or explicit `INTERGRAX_UV_EXECUTABLE`) can run semantic ENT-AUDIT-X checks. No shell, no machine-specific hardcoded paths.

## Plan digest

Plan digest: 856033ca734c46dcca222886fa89b1611a8e34cd23db447a1b705f8c3ac8bc9f

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-062-B.md`](plans/PLAN_T-IGX-062-B.md).

## Approved

Approved: operator @ 2026-11-22T19:55:00+01:00

Approval source: operator turn **EQO WAVE 04 — AUTHORIZATION UNLOCK & IMPLEMENTATION** (this session), authorizing formal ticket publication and implementation of T-IGX-062-A and T-IGX-062-B. Timestamp is the recorded entry time.

## Write set

```text
intergrax/platform/enterprise_audit/sandbox.py
tests/unit/platform/enterprise_audit/test_sandbox_uv_resolution.py
governance/tickets/T-IGX-062-B.md
governance/tickets/plans/PLAN_T-IGX-062-B.md
```

## Test contract

```text
uv run pytest tests/unit/platform/enterprise_audit/test_sandbox_uv_resolution.py tests/qualification/ent_audit_x/unit/test_sandbox_isolation.py -q --tb=line
uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_dup_01.py -q --tb=line
```

## Lifecycle state

`approved`