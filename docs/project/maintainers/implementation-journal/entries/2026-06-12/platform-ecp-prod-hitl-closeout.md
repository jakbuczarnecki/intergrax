---
id: IJ-2026-06-12-028
date: 2026-06-12
tiers:
  - tier-1
  - tier-3
scope: ELASTIC_CAPACITY_AND_SCALING
plan_ref:
  - ECP-PROD.6
  - ECP-PROD.1
  - AUDIT-IDEAL-30.4
status: completed
commit: pending
adr: none — extends ECP-PROD within ADR-SCALE-001 tier boundaries
---

# ECP-PROD closeout — HITL approval queue and production adapters

## Operator request

Iteratively close remaining ECP-PROD gaps per Layer Completion Mode instruction.

## Summary

Shipped `CapacityApprovalQueue` with SCALE_REQUESTED/APPROVED/DENIED events, scheduler
enqueue/drain flow, queue depth provider from `task_index`, URL-gated K8s production
adapter resolution, and Prometheus query bridge when `INTERGRAX_PROMETHEUS_URL` is set.

## Project impact

ECP production elasticity reaches L3 when operators enable `ScalingProfile`; HITL scale-up
is no longer status-only; production hosts can use live K8s REST scale outside CI.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md` §22 |
| Plan | `docs/project/maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md` Phase ECP-PROD |
| ADR | `docs/project/technical/adr/entries/2026-06-08/ADR-SCALE-001.md` |

## Changed artifacts

- `intergrax/runtime/capacity/approval_queue.py` — HITL queue
- `intergrax/runtime/capacity/governance.py` — approve/deny API
- `intergrax/runtime/capacity/production_adapters.py` — URL-gated K8s backend
- `intergrax/runtime/events/runtime_event.py` — SCALE_REQUESTED/APPROVED/DENIED

## Verification

- `uv run pytest tests/unit/runtime/capacity/ tests/integration/runtime/test_ecp_backpressure_scale.py -q` — 19 passed
- `uv run python scripts/maintenance/check_production_capacity_adapters.py` — OK

## Risks and follow-ups

- `ScalingProfile.policy.enabled` remains opt-in on lab defaults
- Live K8s scale requires runtime `INTERGRAX_KUBERNETES_URL` — not exercised in CI
- ECP complements native HPA/Celery autoscale; operators still configure both
