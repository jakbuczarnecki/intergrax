# Elastic Capacity and Scaling — Implementation Plan

**Architecture (1:1):** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../architecture/ELASTIC_CAPACITY_AND_SCALING.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**ADR:** [`adr/ADR-SCALE-001.md`](../adr/ADR-SCALE-001.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §24.7 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-24.3 | §24 Cost | CPU/memory/concurrency quotas with tenant fairness (shared UAEP) | P2 | **Done** |
| AUDIT-IDEAL-30.1 | §30 Ops | Sync `architecture/ELASTIC_CAPACITY_AND_SCALING.md` §22 after ECP-DEPTH | **P0** | Planned |
| AUDIT-IDEAL-30.4 | §30 Ops | Celery/K8s production-scale adapters (beyond stub/beta) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

(Global)

1. **Contract** — Pydantic / Protocol public API for signals, policies, actions
2. **Trace** — capacity transitions emit `RuntimeEvent` (`ops:capacity`, `ops:backpressure`)
3. **Test** — unit + integration, deterministic; mock integrations (no live K8s in gate)
4. **Documentation** — update this plan + architecture pair when contracts change
5. **No regression** — `pytest -m gate` green
6. **Reuse Tier-0** — extend `integrations/`, `queueing/`; no parallel cloud SDK stacks in Nexus
7. **Async control plane** — ECP MUST NOT block `NexusLoop` hot path
8. **Tier discipline** — provision via Integration Library; deploy YAML stays Tier-3
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2

---

## Phase ECP-DOC — Domain pair establishment (Band 2an)

**Status:** **Done** (2026-06-08) — architecture + plan pair + ADR-SCALE-001; hub + audit routing updated  
**Prerequisites:** Phase W-OPS **Done** · Phase ORCH/FLOW backpressure **Done** · `kubernetes` integration beta  
**Goal:** Establish **19th domain pair** as canonical source for Harness Elastic Capacity Plane (ECP) — consolidate scattered scaling docs without runtime controller  
**Priority ladder:** **Band 2an** (§4.0 PLATFORM_FOUNDATION) — **closed** on doc merge  
**Execution order:** [§6.2an](#62an-phase-ecp-doc-execution-order-band-2an--closed) · queue: [§6.1an](#61an-harness-implementation-queue--elastic-capacity-domain-pair-closed)

**Delivery rule:** ECP-DOC.* = docs + ADR only; runtime work routes to ECP-DEPTH.*

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| ECP-DOC.1 | **`architecture/ELASTIC_CAPACITY_AND_SCALING.md`** — full ECP canon | **Done** | **Critical** | `docs/architecture/` | Hub links; audit §30 extension |
| ECP-DOC.2 | **`plan/ELASTIC_CAPACITY_AND_SCALING.md`** — this file; ECP-DEPTH register | **Done** | **Critical** | `docs/plan/` | 1:1 pair check green |
| ECP-DOC.3 | **`docs/adr/ADR-SCALE-001.md`** — ECP vs K8s HPA; tier separation | **Done** | High | `docs/adr/` | Linked from architecture + adr README |
| ECP-DOC.4 | **Hub update** — 19 domain pairs; audit routing for capacity | **Done** | High | `intergrax_runtime_architecture.md` | `check_docs_domain_pairs.py` OK |
| ECP-DOC.5 | **Cross-ref sync** — ORCHESTRATION §49, OBS §9.3, INTEGRATIONS k8s, AGENTS.md, audit map §30 | **Done** | High | `docs/*` | No orphan scaling narrative |
| ECP-DOC.6 | **Gate script** — `python scripts/check_docs_domain_pairs.py` | **Done** | Medium | CI scripts | 19 pairs reported |

---

## Phase ECP-DEPTH — Elastic capacity runtime (Band 2ao — closed)

**Status:** **Done** (2026-06-09) — **28/28 Done** (ECP-6.2 **Cancelled**) · register: [ECP-DEPTH — Master deliverables register](#ecp-depth--master-deliverables-register-all-28-tasks)  
**Prerequisites:** Phase ECP-DOC **Done**  
**Goal:** Raise ECP from **L1 → L3+** — closed-loop Observe → Evaluate → Govern → Provision  
**Priority ladder:** **Band 2ao** (§4.0) — **closed**; default queue = §6.1 maintenance  
**Traceability:** [Appendix A](#appendix-a--elastic-capacity-traceability-phase-ecp-depth)

**Delivery rule:** One **ECP-* ID per PR** → update master table + architecture §22 → `pytest -m gate` green.

**Principle:** **complement, not replace** K8s HPA · async scheduler · policy-first scale-up · idempotent actions · hysteresis on scale-down.

**Out of scope:** K.1/K.2 product hosts · training/inference cluster autoscaling (MLOps) · replacing Tier-3 Helm ownership · agent registry mutation as “scaling”.

### ECP-DEPTH — Maturity targets

| Area | Current (post ECP-DOC) | Target | Primary IDs |
|------|------------------------|--------|-------------|
| ScalingProfile | L0 | L3 | ECP-1.* |
| Signal collection | L1 partial | L3 | ECP-2.* |
| Rules evaluator | L0 | L3 | ECP-3.* |
| K8s scale API | L1 beta | L3 | ECP-4.* |
| Celery worker scale | L2 manual | L3 | ECP-5.* |
| nginx / ingress | L0 | L2 | ECP-6.* |
| Policy + HITL gates | L0 | L3 | ECP-7.* |
| AHI ↔ ECP bridge | L0 | L2 optional | ECP-8.* |
| Capacity observability | L1 | L3 | ECP-OBS.* |

**Success gate:** P0 + P1 **Done**; integration tests with mock K8s; `GRAPH_BACKPRESSURE` → scale action in lab profile; FAUDIT §30 extension **L3+**.

```text
Wave ECP0 — Package scaffold (4 tasks)
Wave ECP1 — Contracts + ScalingProfile P0 (5 tasks)
Wave ECP2 — Signal collector P0 (4 tasks)
Wave ECP3 — Evaluator + rules P0 (4 tasks)
Wave ECP4 — K8s provisioner P1 (4 tasks)
Wave ECP5 — Celery / queue worker scale P1 (3 tasks)
Wave ECP6 — nginx / ingress RFC + slug P2 (2 tasks)
Wave ECP7 — Policy, HITL, anti-flap P1 (3 tasks)
Wave ECP8 — AHI bridge optional P2 (1 task)
Wave ECP-OBS — Metrics + trace events P1 (2 tasks)
Total ECP-DEPTH: 28 (excluding ECP-DOC)
```

---

### 6.2an Phase ECP-DOC execution order (Band 2an — closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | ECP-DOC.3 | ADR-SCALE-001 | High |
| 2 | ECP-DOC.1 | Architecture canon | Critical |
| 3 | ECP-DOC.2 | Plan register | Critical |
| 4 | ECP-DOC.4 | Hub 19-pair index | High |
| 5 | ECP-DOC.5 | Cross-ref sync | High |
| 6 | ECP-DOC.6 | Domain pair gate | Medium |

### 6.1an Harness implementation queue — Elastic capacity domain pair (closed)

**Status:** **Closed** (2026-06-08)  
**Band:** 2an  
**Outcome:** 19th domain pair live; ECP canon + ADR accepted.

---

### 6.2ao Phase ECP-DEPTH execution order (Band 2ao — planned)

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| ECP0 | ECP-0.1–ECP-0.4 | 4 | Package scaffold + import gate |
| ECP1 | ECP-1.1–ECP-1.5 | 5 | **P0** — contracts + ScalingProfile |
| ECP2 | ECP-2.1–ECP-2.4 | 4 | **P0** — CapacitySignalCollector |
| ECP3 | ECP-3.1–ECP-3.4 | 4 | **P0** — ScalingEvaluator |
| ECP4 | ECP-4.1–ECP-4.4 | 4 | **P1** — K8s provisioner depth |
| ECP5 | ECP-5.1–ECP-5.3 | 3 | **P1** — Celery worker autoscale |
| ECP6 | ECP-6.1–ECP-6.2 | 2 | **P2** — nginx / ingress slug |
| ECP7 | ECP-7.1–ECP-7.3 | 3 | **P1** — Policy + HITL + flap guard |
| ECP8 | ECP-8.1 | 1 | **P2** — AHI approved proposal bridge |
| ECPOBS | ECP-OBS.1–ECP-OBS.2 | 2 | **P1** — Trace + metrics |
| **Total** | | **28** | |

---

## ECP-DEPTH — Master deliverables register (all 28 tasks)

### Wave ECP0 — Package scaffold

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| ECP-0.1 | **`intergrax/runtime/capacity/`** package — `contracts.py`, `__init__.py` | **Done** | **Critical** | Importable; no side effects |
| ECP-0.2 | **Gate import test** — `tests/unit/runtime/capacity/test_ecp_depth_gate.py` | **Done** | Medium | `-m gate` green |
| ECP-0.3 | **Architecture ↔ plan sync** — paydown log row | **Done** | Low | §22 updated |
| ECP-0.4 | **Extend `runtime/architecture/__init__.py`** re-exports if needed | **Done** | Low | `capacity` package import gate |

### Wave ECP1 — Contracts and ScalingProfile (P0)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-1.1 | **`ScalingProfile`** on `ApplicationEnvironmentProfile` | **Done** | **Critical** | `environment_profile.py` | Round-trip on lab defaults |
| ECP-1.2 | **`CapacitySignal`**, **`ScalingPolicy`**, **`ScalingAction`** Pydantic models | **Done** | **Critical** | `capacity/contracts.py` | `test_ecp_depth_gate.py` |
| ECP-1.3 | **`ScalingTarget`** enum — `NEXUS_HOST`, `CELERY_POOL`, `MODALITY_POOL`, `ORCHESTRATION_CEILING` | **Done** | High | same | Exhaustive match |
| ECP-1.4 | **`scaling_wiring.py`** — host bootstrap hook (no-op when disabled) | **Done** | High | `applications/_shared/scaling_wiring.py` | Lab host lifespan when enabled |
| ECP-1.5 | **Reference YAML** — lab scaling policy stub in docs only | **Done** | Low | `HARNESS_ENVIRONMENT.md` | Example policy JSON |

### Wave ECP2 — Signal collector (P0)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-2.1 | **`CapacitySignalCollector`** — aggregate `GRAPH_BACKPRESSURE` rate | **Done** | **Critical** | `capacity/collector.py` | `test_ecp_depth_gate.py` |
| ECP-2.2 | **Queue depth signal** — from `task_index` | **Done** | High | same | `queue_depth_provider` hook |
| ECP-2.3 | **Prometheus SLI bridge** (optional profile) | **Done** | Medium | `capacity/prometheus_bridge.py` | Stub PromQL bridge |
| ECP-2.4 | **Emit `CAPACITY_SIGNAL_COLLECTED`** events | **Done** | High | `capacity/events.py`, collector | `RuntimeEventType.CAPACITY_SIGNAL_COLLECTED` |

### Wave ECP3 — Evaluator (P0)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-3.1 | **`ScalingEvaluator`** — rule matching + cooldown | **Done** | **Critical** | `capacity/evaluator.py` | `test_ecp_depth_gate.py` |
| ECP-3.2 | **Hysteresis** — separate up/down thresholds | **Done** | High | same | Flap scenario test |
| ECP-3.3 | **`ScalingActionPlan`** output — ordered actions | **Done** | High | same | Immutable plan |
| ECP-3.4 | **Emit `SCALE_EVALUATED`** | **Done** | Medium | evaluator | `RuntimeEventType.SCALE_EVALUATED` |

### Wave ECP4 — Kubernetes provisioner (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-4.1 | **Extend `kubernetes` contract** — `scale_workload`, `get_replicas` | **Done** | **Critical** | `integrations/_shared/p5/clients.py` | `test_ecp_depth_gate.py` |
| ECP-4.2 | **`ScalingProvisioner`** — K8s backend | **Done** | **Critical** | `capacity/provisioner.py` | Integration with mock |
| ECP-4.3 | **Emit `SCALE_APPLIED` / `SCALE_FAILED`** | **Done** | High | provisioner | Dedicated runtime event types |
| ECP-4.4 | **INTEGRATIONS plan row** — cross-ref ECP-4 | **Done** | Low | `plan/INTEGRATIONS.md` M-P4.20 | Link resolves |

### Wave ECP5 — Celery / queue worker scale (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-5.1 | **Celery worker scale action** — document + stub executor | **Done** | High | `capacity/provisioner.py` | Stub pass-through action |
| ECP-5.2 | **Generalize W-OPS.12 pattern** — beyond modality only | **Done** | High | `scaling_wiring.py` | Lab host wiring |
| ECP-5.3 | **Queue depth → worker scale rule** — reference policy | **Done** | Medium | `HARNESS_ENVIRONMENT.md`, `test_capacity_events_gate.py` | Reference JSON policy |

### Wave ECP6 — nginx / ingress (P2)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-6.1 | **RFC: nginx vs ingress_controller slug** | **Done** | Medium | ADR-SCALE-002 | Defer slug; K8s deployment path canonical |
| ECP-6.2 | **Integration scaffold** (if accepted) | **Cancelled** | Low | — | Superseded by ADR-SCALE-002 deferral |

### Wave ECP7 — Policy and HITL (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-7.1 | **`BEFORE_CAPACITY_ACTION` hook** | **Done** | High | `capacity/action_gate.py`, provisioner | `test_capacity_events_gate.py` deny path |
| ECP-7.2 | **HITL gate for scale-up** when `require_hitl_for_scale_up` | **Done** | High | `capacity/governance.py`, evaluator | `hitl_required` plan status |
| ECP-7.3 | **Anti-flapping guard** — max actions/hour | **Done** | High | evaluator | `max_actions_per_hour` |

### Wave ECP8 — AHI bridge (P2, optional)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-8.1 | **Consume approved AHI proposal** → ceiling raise action | **Done** | Low | `capacity/ahi_bridge.py` | `test_ecp_depth_gate.py` |

### Wave ECP-OBS — Observability (P1)

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ECP-OBS.1 | **Capacity metrics** — `harness_scale_actions_total`, replica gauge | **Done** | High | `capacity/metrics.py` | `test_ecp_depth_gate.py` |
| ECP-OBS.2 | **`CapacityScheduler`** — async cron driver | **Done** | **Critical** | `capacity/scheduler.py` | Async lifespan on lab host when enabled |

---

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
| `queueing/` workers | **Done** | §14 |
| W-OPS.4 SLO catalog | **Done** | §10, §18 |
| W-OPS.12 Celery modality | **Done** | §14 |
| `kubernetes` integration beta | **Done** | §12.3 |
| K8s HPA in Tier-3 Helm | Operator-owned | §8, §16 |

---

## Appendix B — FAUDIT-32 §30 extension scorecard

| Audit question | Pre-ECP | Post ECP-DOC | Post ECP-DEPTH target |
|----------------|---------|--------------|----------------------|
| SLOs defined? | Yes (W-OPS) | Yes | Maintain |
| SLIs → capacity action? | No | Documented §10 | ECP-2.* Done |
| Closed-loop scale? | No | Canon §5 | ECP-3.*–4.* Done |
| Runbooks for scale failure? | Partial | §19 taxonomy | ECP-7 + runbook |
| **Ops excellence (capacity)** | **L1** | **L1** (plan accurate) | **L3** (ECP-DEPTH **Done**) |

---

## Appendix C — Operator reading order

1. [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../architecture/ELASTIC_CAPACITY_AND_SCALING.md) — ECP canon
2. [`adr/ADR-SCALE-001.md`](../adr/ADR-SCALE-001.md) — decision vs K8s HPA
3. This plan — ECP-DEPTH when implementing
4. [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §49 — queueing (not duplicate)
5. [`guides/HARNESS_ENVIRONMENT.md`](../guides/HARNESS_ENVIRONMENT.md) — SLO + Celery env

---

### ECP-DEPTH — Paydown log

| Date | ECP ID | Summary |
|------|--------|---------|
| 2026-06-09 | ECP-0.*–ECP-OBS.* | Phase ECP-DEPTH **28/28 Done** (ECP-6.2 Cancelled) |

---

*End of Elastic Capacity and Scaling Implementation Plan.*
