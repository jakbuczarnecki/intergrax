# T-IGX-056-A — EQO A4-006: rebind RuntimeEvent tenant propagation gate emitters

## Approved

Approved: operator @ 2026-11-23T13:30:00+01:00

## Scope

Repair `EQOA-A4-006` / `EDER-F-0115`: the maintenance gate `scripts/maintenance/check_runtime_event_tenant_propagation.py` still listed the removed `intergrax/agents/uaep.py` path and failed closed on a missing file. Rebind the UAEP emitter to `intergrax/runtime/nexus/uaep/uaep_executor.py`; keep `intergrax/runtime/middleware/trace_middleware.py`. Add a focused unit test and EQO packet evidence for WAVE-01 (`EQO-PKG-A4-006`). Package reference: approved plan `eqo_completion_orchestration_67c47aad.plan.md` @ `a0598ba3e4f4f903eb4da3db8ea012d5cbf71225`.

Renumbered from `T-IGX-055-A` during merge with `origin/development` (remote `T-IGX-055-A` is CURSOR-ORCH-CORRECTIVE).

## Write set

```text
scripts/maintenance/check_runtime_event_tenant_propagation.py
tests/unit/scripts/test_check_runtime_event_tenant_propagation.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/orchestration/packets/EQO-PKG-A4-006/
governance/tickets/T-IGX-056-A.md
```

## Test contract

```text
uv run pytest tests/unit/scripts/test_check_runtime_event_tenant_propagation.py -q --basetemp=D:/Projekty/_wt/intergrax/_scratch/eqo-w1-a4006/pytest
```

## Out of scope

FRZ checklist, roadmap, `IIAC_POLICY_ACTIVATION_RECORD.json`, `EQO_MASTER_CLOSURE_MATRIX.md` status promotions, `EQO_CLOSED`, any other `RuntimeEvent` call site census, host factories, `applications/`, EDER `register.jsonl`, obligation view JSON, `eqo_closure/orchestration/*.json`, `HANDOFF.md`.

## Lifecycle state

in_progress