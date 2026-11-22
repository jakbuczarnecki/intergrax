# PLAN T-IGX-062-A — TRACE-X P6 `redeliver` discovery word-boundary

## Problem

Substring marker `redeliver` matches comment text `redelivery`, falsely registering `intergrax/platform/enterprise_audit/impact_map.py` as a restart/resume surface (EQO WB10 / A2-019).

## Change

1. In `_trace_x_p6_discovery.py`, match `redeliver` with `\bredeliver\b` only; other markers remain literal substring checks.
2. Add unit test: comment-only `redelivery` must not register; real `redeliver` token must register.

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

P6 pass1 matrix (8 nodeids) with `TRACE_X_P6_PASS1=1` as in Wave 02 evidence.