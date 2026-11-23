# T-IGX-062-A — TRACE-X P6 discovery: `redeliver` word-boundary (EQO Wave 04 / P01)

## Scope

Fix false-positive restart/resume discovery when audit comments contain the substring `redelivery` (e.g. `intergrax/platform/enterprise_audit/impact_map.py`). Qualification-only; no change to FRZ-TRC semantics or classification allowlists.

## Plan digest

Plan digest: 047cacbe3d9545a1ba05f6ac6ca7d27e474060b4d1d82169f051ad6f8474754c

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-062-A.md`](plans/PLAN_T-IGX-062-A.md).

## Approved

Approved: operator @ 2026-11-22T19:55:00+01:00

Approval source: operator turn **EQO WAVE 04 — AUTHORIZATION UNLOCK & IMPLEMENTATION** (this session), authorizing formal ticket publication and implementation of T-IGX-062-A and T-IGX-062-B. Timestamp is the recorded entry time.

## Write set

```text
tests/qualification/trace_x/_trace_x_p6_discovery.py
tests/unit/qualification/trace_x/test_trace_x_p6_discovery_redeliver_boundary.py
governance/tickets/T-IGX-062-A.md
governance/tickets/plans/PLAN_T-IGX-062-A.md
```

## Test contract

```text
uv run pytest tests/unit/qualification/trace_x/test_trace_x_p6_discovery_redeliver_boundary.py tests/qualification/trace_x/test_trace_x_p6_closed_world_gates.py -q --tb=line
```

With `TRACE_X_P6_PASS1=1`, include the eight P6-A..H matrix nodeids per Wave 02 evidence.

## Lifecycle state

in_progress