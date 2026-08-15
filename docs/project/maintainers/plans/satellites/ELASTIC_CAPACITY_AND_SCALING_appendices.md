# ELASTIC_CAPACITY_AND_SCALING — appendices

**Parent hub:** [`ELASTIC_CAPACITY_AND_SCALING.md`](../ELASTIC_CAPACITY_AND_SCALING.md)

## Appendix A — Elastic capacity traceability (Phase ECP-DEPTH)

| Architecture § | Topic | Task IDs |
|----------------|--------|----------|
| §5 ECP layers | Control plane | ECP-0.*, ECP-OBS.2 |
| §10 Signals | Collector | ECP-2.* |
| §11 Policies | Evaluator | ECP-3.* |
| §12 Actions | Provisioner | ECP-4.*, ECP-5.*, ECP-6.* |
| §16 ScalingProfile | Tier-3 profile | ECP-1.* |
| §17 Governance | HITL + policy | ECP-7.* |
| §18 Observability | Metrics + events | ECP-OBS.*, ECP-2.4, ECP-3.4 |
| §15 AHI | Bridge | ECP-8.* |

### Historical as-built (pre-ECP domain)

| Artifact | Status | ECP architecture § |
|----------|--------|-------------------|
| `max_inflight_nodes` / `GRAPH_BACKPRESSURE` | **Done** (FLOW-13) | §13 |
| `queueing` workers | **Done** | §14 |
| W-OPS.4 SLO catalog | **Done** | §10, §18 |
| W-OPS.12 Celery modality | **Done** | §14 |
| `kubernetes` integration beta | **Done** | §12.3 |
| K8s HPA in Tier-3 Helm | Operator-owned | §8, §16 |

---

## Appendix B — FAUDIT-32 §30 extension scorecard

| Audit question | Pre-ECP | Post ECP-DOC | Post ECP-DEPTH scaffold | Post ECP-PROD target |
|----------------|---------|--------------|-------------------------|---------------------|
| SLOs defined? | Yes (W-OPS) | Yes | Yes | Maintain |
| SLIs → capacity action? | No | Documented §10 | Gate tests only | ECP-PROD.1 live |
| Closed-loop scale? | No | Canon §5 | Mock/stub path | ECP-PROD.7 E2E |
| Runbooks for scale failure? | Partial | §19 taxonomy | §19 taxonomy | ECP-7 + runbook |
| **Ops excellence (capacity)** | **L1** | **L1** | **L2** (honest) | **L3+** |

---

## Appendix C — Operator reading order

1. [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../architecture/ELASTIC_CAPACITY_AND_SCALING.md) — ECP canon
2. [`adr/entries/2026-06-08/ADR-SCALE-001.md`](../adr/entries/2026-06-08/ADR-SCALE-001.md) — decision vs K8s HPA
3. This plan — ECP-DEPTH when implementing
4. [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §49 — queueing (not duplicate)
5. [`guides/HARNESS_ENVIRONMENT.md`](../guides/HARNESS_ENVIRONMENT.md) — SLO + Celery env

---

### ECP-DEPTH — Paydown log

| Date | ECP ID | Summary |
|------|--------|---------|
| 2026-06-09 | ECP-0.*–ECP-OBS.* | Phase ECP-DEPTH **28/28 scaffold Done** (ECP-6.2 Cancelled) |
| 2026-06-12 | AUDIT-IDEAL-30.1 | Honest maturity: ECP-DEPTH ≠ production autoscaling; §22 gap register |
| 2026-06-12 | ECP-PROD.* | Phase ECP-PROD closed — HITL queue, K8s URL-gated adapters, E2E |

---
