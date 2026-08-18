# Elastic Capacity and Scaling — Implementation Plan

**Architecture (1:1):** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**ADR:** [`adr/entries/2026-06-08/ADR-SCALE-001.md`](../../technical/adr/entries/2026-06-08/ADR-SCALE-001.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Last updated:** 2026-06-20 — **P2-ARCH-11** ECP production boundary.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ELASTIC_CAPACITY_AND_SCALING plan).

- **Implement / audit default:** ECP phase registers · open P0/P1 capacity rows · skip closed scaling history unless cited
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/ELASTIC_CAPACITY_AND_SCALING_appendices.md`](plan/satellites/ELASTIC_CAPACITY_AND_SCALING_appendices.md) | appendices |
| [`plan/satellites/ELASTIC_CAPACITY_AND_SCALING_audit_history.md`](plan/satellites/ELASTIC_CAPACITY_AND_SCALING_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §24.7 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-24.3 | §24 Cost | CPU/memory/concurrency quotas with tenant fairness (shared UAEP) | P2 | **Done** |
| AUDIT-IDEAL-30.1 | §30 Ops | Honest §22 maturity — ECP is architecture, not production autoscaling | **P0** | **Done** (2026-06-12) |
| AUDIT-IDEAL-30.4 | §30 Ops | Celery/K8s production-scale adapters (beyond stub/beta) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

(Global)

1. **Contract** — Pydantic / Protocol public API for signals, policies, actions
2. **Trace** — capacity transitions emit `RuntimeEvent` (`ops:capacity`, `ops:backpressure`)
3. **Test** — unit + integration, deterministic; mock integrations (no live K8s in gate)
4. **Documentation** — update this plan + architecture pair when contracts change
5. **No regression** — `pytest -m gate` green
6. **Reuse Tier-0** — extend `integrations`, `queueing`; no parallel cloud SDK stacks in Nexus
7. **Async control plane** — ECP MUST NOT block `NexusLoop` hot path
8. **Tier discipline** — provision via Integration Library; deploy YAML stays Tier-3
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2

---

## ECP-DEPTH — Master deliverables register (all 28 tasks)

### Wave ECP0 — Package scaffold

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| ECP-0.1 | **`intergrax/runtime/capacity`** package — `contracts.py`, `__init__.py` | **Done** | **Critical** | Importable; no side effects |
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

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-11** | Clarify ECP production boundary and scaling governance | **Done** (2026-06-20) |

---
