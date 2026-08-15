---
id: IJ-2026-06-12-027
date: 2026-06-12
tiers:
  - tier-1
  - tier-3
scope: ELASTIC_CAPACITY_AND_SCALING
plan_ref:
  - ECP-PROD.1
  - ECP-PROD.2
  - ECP-PROD.3
  - ECP-PROD.4
  - ECP-PROD.5
  - ECP-PROD.7
  - AUDIT-IDEAL-30.1
status: completed
commit: pending
adr: none — documentation honesty + incremental ECP-PROD; ADR-SCALE-001 unchanged
---

# ECP-PROD — honest maturity and production elasticity hardening

## Operator request

Close the Elastic Capacity layer with honest production positioning: ECP architecture
is sound but must not be marketed as finished autoscaling; reconcile docs with code,
then implement the highest-priority ECP-PROD gaps.

## Summary

Reclassified ECP-DEPTH as scaffold (L2) vs production fleet autoscaling. Added
`CapacityEventBridge` for `GRAPH_BACKPRESSURE`, HITL-safe `CapacityScheduler`,
`KubernetesDeploymentScaleClient` (REST scale when `INTERGRAX_KUBERNETES_URL` set),
Celery/ceiling provisioner backends, and integration test for backpressure → K8s scale.

## Project impact

Harness operators get an honest maturity model and a test-proven closed-loop path;
production fleet autoscaling still requires enabling `ScalingProfile` and live
cluster/broker configuration (ECP-PROD.6 HITL queue and AUDIT-IDEAL-30.4 live cluster remain partial).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md` §1.1, §22 |
| Plan | `docs/project/maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md` Phase ECP-PROD |
| ADR | `docs/project/technical/adr/entries/2026-06-08/ADR-SCALE-001.md` |
| Audit / gap | AUDIT-IDEAL-30.1 Done; AUDIT-IDEAL-30.4 Partial |

## Changed artifacts

- `intergrax/runtime/capacity/event_bridge.py` — live backpressure subscription
- `intergrax/runtime/capacity/scheduler.py` — skip non-planned evaluation status
- `intergrax/integrations/providers/cloud_platform/kubernetes/rest_client.py` — K8s scale API
- `intergrax/runtime/capacity/provisioner.py` — Celery + ceiling backends
- `tests/integration/runtime/test_ecp_backpressure_scale.py` — E2E gate

## Verification

- `uv run pytest tests/unit/runtime/capacity/ tests/unit/integrations/providers/test_kubernetes_scale_client.py tests/integration/runtime/test_ecp_backpressure_scale.py -m gate -q` — 18 passed
- `uv run python scripts/maintenance/check_production_capacity_adapters.py` — OK

## Risks and follow-ups

- ECP-PROD.6 — durable HITL approval queue for scale-up not yet shipped
- AUDIT-IDEAL-30.4 — live cluster/broker adapters outside in-memory CI probe
- `ScalingProfile.policy.enabled` remains false by default; operators must opt in with runbooks
