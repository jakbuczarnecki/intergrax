# Elastic Capacity and Scaling

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](../plan/ELASTIC_CAPACITY_AND_SCALING.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §0.3, §3.8, §12  
**Audit layers:** 30 (Operational Excellence) · cross-ref 9 (orchestration backpressure), 21 (observability SLIs)  
**Audit instruction:** [`audit/ELASTIC_CAPACITY_AND_SCALING.md`](../audit/ELASTIC_CAPACITY_AND_SCALING.md)  
**ADR:** [ADR-SCALE-001](../adr/entries/2026-06-08/ADR-SCALE-001.md)  
**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates ECP-PROD); honest maturity **Done**

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ELASTIC_CAPACITY_AND_SCALING canon).

- **Implement / audit default:** capacity adapter contracts. Skip scaling history unless ECP task.
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](../plan/ELASTIC_CAPACITY_AND_SCALING.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md`](../guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/ELASTIC_CAPACITY_AND_SCALING_scenario_catalog.md`](arch/ELASTIC_CAPACITY_AND_SCALING_scenario_catalog.md) | scenario catalog |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


## Table of contents

1. [Purpose](#1-purpose)
2. [Problem statement](#2-problem-statement)
3. [Terminology](#3-terminology)
4. [Design principles](#4-design-principles)
5. [Harness Elastic Capacity Plane (ECP)](#5-harness-elastic-capacity-plane-ecp)
6. [Two scaling dimensions](#6-two-scaling-dimensions)
7. [Ideal architecture alignment](#7-ideal-architecture-alignment)
8. [Tier placement and responsibility matrix](#8-tier-placement-and-responsibility-matrix)
9. [Domain boundaries](#9-domain-boundaries)
10. [Signal model](#10-signal-model)
11. [ScalingPolicy and rules engine](#11-scalingpolicy-and-rules-engine)
12. [Scaling actions and integration surface](#12-scaling-actions-and-integration-surface)
13. [Relationship to orchestration backpressure](#13-relationship-to-orchestration-backpressure)
14. [Relationship to queueing and workers](#14-relationship-to-queueing-and-workers)
15. [Relationship to Adaptive Harness Intelligence](#15-relationship-to-adaptive-harness-intelligence)
16. [ScalingProfile (Tier-3)](#16-scalingprofile-tier-3)
17. [Governance, policy, and HITL](#17-governance-policy-and-hitl)
18. [Observability and trace contracts](#18-observability-and-trace-contracts)
19. [Failure taxonomy and anti-flapping](#19-failure-taxonomy-and-anti-flapping)
20. [End-to-end capacity loop](#20-end-to-end-capacity-loop)
21. [As-built vs target](#21-as-built-vs-target)
22. [Maturity scorecard and gap register](#22-maturity-scorecard-and-gap-register)
23. [Related documents](#23-related-documents)
24. [Appendix A — Code map (as-built + target)](#appendix-a--code-map-as-built--target)
25. [Appendix B — Integration catalog (scaling-relevant)](#appendix-b--integration-catalog-scaling-relevant)
26. [Appendix C — Audit and ideal traceability](#appendix-c--audit-and-ideal-traceability)

---

## 1. Purpose

Define the **Harness Elastic Capacity Plane (ECP)** — the subsystem that answers:

> **When load grows, how does the platform add execution capacity — runners, workers, replicas — in a governed, observable way?**

ECP completes the **Observe → Evaluate → Provision** loop for **compute and worker capacity**. It **does not** decide agent topology (which agents run) — that belongs to [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md). It **does not** schedule graph batches inside fixed capacity — that belongs to [`ORCHESTRATION.md`](ORCHESTRATION.md).

**Strategic positioning:** The Harness owns **how** capacity scales (signals, rules, actions, audit); infrastructure vendors (Kubernetes, nginx, Celery, cloud APIs) remain **replaceable integrations**; Tier-3 applications own **deployment manifests and profile defaults**.

**Core invariant:** Capacity mutations MUST flow through **typed `ScalingAction` contracts** and **Integration Library / ToolRuntime** — never ad-hoc SDK calls from `NexusLoop` or agents.

### 1.1 Production positioning (honest)

| Claim | Accurate today? |
|-------|-----------------|
| ECP is the **canonical architecture** for Harness elastic capacity | **Yes** |
| ECP delivers **production autoscaling** comparable to K8s HPA + Celery autoscale | **No** |
| Backpressure (`GRAPH_BACKPRESSURE`) **throttles** within fixed capacity | **Yes** — primary as-built behavior |
| Closed-loop **Observe → Evaluate → Govern → Provision** runs in production | **Yes** when `ScalingProfile.policy.enabled=true` + operator config |

**Phase split:**

| Phase | Delivers | Maturity |
|-------|----------|----------|
| **ECP-DOC** | Domain pair, ADR-SCALE-001, tier boundaries | **Done** |
| **ECP-DEPTH** | Contracts, `runtime/capacity/` scaffold, gate tests, disabled-by-default wiring | **Done** (harness **L2** — architecture + scaffold) |
| **ECP-PROD** | Live signal bridge, HITL queue, K8s REST/Celery adapters, E2E gate | **Done** (target **L3** with operator enablement) |

Do **not** market ECP-DEPTH as a finished production autoscaling system. Operators should continue to rely on **K8s HPA**, **Celery autoscale**, and manual runbooks until **ECP-PROD** closes the gap register in §22.

---

## 2. Problem statement

Intergrax today scales **within** fixed capacity and scales **data planes** — but lacks a unified Harness layer for **elastic compute**:

| Gap | Impact |
|-----|--------|
| `max_inflight_nodes` / `GRAPH_BACKPRESSURE` throttle only | Saturation visible; no automatic capacity response |
| `queueing/` + message_bus workers exist | Fixed worker count; no Harness policy to add workers |
| `kubernetes` integration (beta) | Health/deploy facade; no scale workload API in canon |
| W-OPS SLO catalog (Done) | SLIs measured; not wired to provisioning actions |
| AHI `ExecutionStrategyEngine` | Proposes profile deltas; does not provision replicas |
| No nginx / ingress integration slug | Load-balancer scaling outside catalog |
| `ScalingProfile` exists but **disabled by default**; collector not wired to live `GRAPH_BACKPRESSURE` stream | Rules declarative; loop not closed in production |
| Documentation scattered across ORCH, OBS, MODALITY, INTEGRATIONS | **Resolved** by ECP-DOC domain pair |
| Plan claimed **L3 Done** while §21 still listed components as Missing | **Resolved** (2026-06-12) — honest maturity in §22 |

ECP-DOC closed the **canon** gap; ECP-DEPTH closed the **contract + scaffold** gap; **ECP-PROD** closes **production elasticity**.

---

## 3. Terminology

| Term | Meaning in Intergrax |
|------|----------------------|
| **Capacity** | Available execution slots: Nexus host replicas, async workers, modality pools, tenant concurrency budget |
| **Elastic scaling** | Automated add/remove capacity based on signals and policy |
| **ECP** | Elastic Capacity Plane — this domain |
| **Scaling signal** | Normalized load metric (queue depth, backpressure rate, latency SLI, cost pressure) |
| **ScalingPolicy** | Rule set: triggers, conditions, actions, cooldowns |
| **ScalingAction** | Typed intent: `SCALE_DEPLOYMENT`, `ADD_WORKERS`, `RAISE_CEILING`, `NOTIFY_ONLY` |
| **Backpressure** | In-process throttle when inflight cap hit (`GRAPH_BACKPRESSURE`) — **not** provisioning |
| **Ceiling raise** | Increase `max_inflight_nodes` / `max_parallel_nodes` within existing replicas |
| **Provisioner** | Component that executes `ScalingAction` via integrations |
| **Flapping** | Rapid scale-up/down oscillation — must be prevented by cooldown + hysteresis |
| **HPA (Kubernetes)** | Native Horizontal Pod Autoscaler — **complementary**, not replaced by ECP |

**Not ECP:** Agent roster expansion in `NexusPlan`, RAG index sharding, trace store Cassandra migration (see [`OBSERVABILITY.md`](OBSERVABILITY.md) §9.3).

---

## 4. Design principles

| Principle | Meaning in Intergrax |
|-----------|---------------------|
| **Async control plane** | Capacity controller runs **outside** Nexus hot path (like AHI scheduler) |
| **Integrations not SDKs** | K8s, Celery, nginx actions via `intergrax/integrations/` and tools |
| **Policy before provision** | `PolicyEngine` + optional HITL on scale-out/up in production |
| **Idempotent actions** | Same signal burst → one effective scale step within cooldown window |
| **Hysteresis** | Separate scale-up and scale-down thresholds |
| **Observe first** | No blind scale — minimum signal window + confidence |
| **Tier-3 profiles** | `ScalingProfile` selects rules per application host |
| **Complement native autoscalers** | ECP may **coordinate with** K8s HPA, not duplicate CPU/memory logic |
| **Fail safe** | Provisioner error → alert + `NOTIFY_ONLY`; never silent infinite scale |
| **Trace everything** | `SCALE_SIGNAL`, `SCALE_EVALUATED`, `SCALE_REQUESTED`, `SCALE_APPLIED` events |

---

## 5. Harness Elastic Capacity Plane (ECP)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER A — Signals (observe)                                             │
│  queue depth · GRAPH_BACKPRESSURE rate · task latency · SLO · cost      │
│  Sources: RuntimeEvent store, Prometheus, queue metrics, AHI signals     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│  LAYER B — Evaluate (Tier-1 ECP)                                         │
│  CapacitySignalCollector → ScalingEvaluator → ScalingActionPlan          │
│  Applies ScalingPolicy rules + cooldown + tenant isolation               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│  LAYER C — Govern (policy + HITL)                                        │
│  PolicyEngine · cost budget · risk class · approval for scale-up         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│  LAYER D — Provision (integrate)                                         │
│  ScalingProvisioner → kubernetes · celery · nginx (future) · cloud API   │
└─────────────────────────────────────────────────────────────────────────┘
```

| Layer | Phase | Status (as-built 2026-06-12) |
|-------|-------|------------------------------|
| **A — Signals** | ECP-DEPTH scaffold | `CapacitySignalCollector` in gate tests; **not** subscribed to live `GRAPH_BACKPRESSURE` / `task_index` on hosts |
| **B — Evaluate** | ECP-DEPTH scaffold | `ScalingEvaluator` + rules — unit/gate only |
| **C — Govern** | ECP-DEPTH scaffold | `CapacityActionGate`, HITL status — **scheduler may apply actions when `hitl_required`** (ECP-PROD.2) |
| **D — Provision** | ECP-DEPTH scaffold | K8s path works with **injected mock client**; default factory is health-only; Celery action is **no-op stub** |

---

## 6. Two scaling dimensions

Do **not** conflate:

| Dimension | Question | Owner domain | Example |
|-----------|----------|--------------|---------|
| **A — Execution capacity** | How many runners/workers/replicas? | **ECP (this doc)** | 3 → 5 Nexus host pods; +2 Celery workers |
| **B — Agent topology** | Which agents/steps in the plan? | [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | Multi-agent plan with 4 steps |

ECP handles **A**. When load grows, ECP adds **capacity to run more agent instances in parallel**; reasoning/orchestration decides **what** those instances execute.

```text
Load increase
    → ECP: more replicas/workers (dimension A)
    → ORCHESTRATION: schedule graph nodes within new capacity
    → REASONING: may propose wider decomposition (dimension B) — separate loop
```

---

## 7. Ideal architecture alignment

[`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md):

| Ideal area | ECP mapping |
|------------|-------------|
| §0.3 modular under scale | ECP as explicit capacity plane |
| §3.8 Reliability & Runtime | Idempotency, recovery — ECP respects checkpointed long runs |
| §12 Observability & Operations | SLI-driven scaling; SLO breach as signal |
| L2 → L3 “Scalable Harness” (audit map) | ECP closes operational scaling gap |

Ideal does **not** yet name ECP explicitly — this domain pair adds that name to canon.

---

## 8. Tier placement and responsibility matrix

| Concern | Tier-0 | Tier-1 ECP | Tier-2 Agent | Tier-3 Application |
|---------|--------|------------|--------------|-------------------|
| Integration adapters (K8s, Celery bus) | **defines** | consumes | — | selects in profile |
| `ScalingPolicy` / `ScalingAction` contracts | defines | evaluates + orchestrates | — | configures |
| Signal collection | metrics backends | `CapacitySignalCollector` | — | webhook thresholds |
| Provision execution | tool handlers | `ScalingProvisioner` | — | — |
| In-process limits | — | may **raise ceiling** via profile patch | — | `OrchestrationProfile` caps |
| Deploy manifests (Helm, HPA YAML) | — | — | — | **owns** `applications/*/docker/` |
| Domain load patterns | — | — | may inform signals | product SLOs |

### 8.1 What ECP MUST NOT do

- Embed Kubernetes/nginx YAML generation in Nexus
- Scale inside `GraphExecutor` synchronous path
- Replace K8s HPA for raw CPU/memory without Harness-specific signals
- Add agents to `AgentRegistry` dynamically as “scaling”
- Bypass `PolicyEngine` for production scale-up

### 8.2 What applications MUST NOT do

- Call cloud APIs directly from Tier-3 host code for routine scaling
- Fork parallel capacity controllers outside ECP
- Set infinite `max_inflight_nodes` instead of provisioning replicas

---

## 9. Domain boundaries

```text
ELASTIC_CAPACITY_AND_SCALING  →  how much capacity (replicas, workers, ceilings)
ORCHESTRATION                 →  when/order/retry within capacity (graph, scheduler)
REASONING_AND_COGNITION       →  what agents/steps (plan topology)
INTEGRATIONS                  →  how to talk to K8s, queues, nginx (adapters)
OBSERVABILITY                 →  SLI storage + alert; feeds ECP signals
ADAPTIVE_HARNESS_INTELLIGENCE →  profile tuning recommendations (optional input)
TIER3_APPLICATION_ENVIRONMENT →  deploy package, ScalingProfile host wiring
```

---

## 10. Signal model

### 10.1 Primary signals (v1 target)

| Signal ID | Source (as-built) | Normalized field | Scale implication |
|-----------|-------------------|------------------|-----------------|
| `SIG_BACKPRESSURE` | `RuntimeEventType.GRAPH_BACKPRESSURE` | events/min per tenant | Raise ceiling or add replicas |
| `SIG_QUEUE_DEPTH` | `intergrax/queueing/task_index.py` | pending task count | Add async workers |
| `SIG_TASK_LATENCY` | trace / Prometheus SLI | p95 task duration | Add replicas |
| `SIG_SLO_BREACH` | W-OPS SLO catalog | boolean + budget burn | Scale + page |
| `SIG_COST_PRESSURE` | `cost_budget.py` | normalized spend rate | Scale-down or block scale-up |
| `SIG_MODALITY_QUEUE` | Celery modality executor | pending modality jobs | Modality worker pool |

### 10.2 HarnessOutcomeSignal bridge (optional)

[`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md) `HarnessOutcomeSignal` may include step/retry/parallel efficiency — ECP MAY consume as **secondary** signal; AHI does not provision infrastructure by default.

### 10.3 Signal contract

**As-built** (`intergrax/runtime/capacity/contracts.py` — ECP-1.2):

```python
class CapacitySignal(BaseModel):
    signal_id: str
    target: ScalingTarget
    metric_name: str
    value: float
    collected_at: datetime
```

**Target enrichment (ECP-PROD.1):** `tenant_id`, `source`, `unit`, `window_seconds`, `metadata` for multi-tenant fairness and PromQL windows.

---

## 11. ScalingPolicy and rules engine

### 11.1 Policy structure (target)

```text
ScalingPolicy:
    policy_id: str
    enabled: bool
    targets: list[ScalingTarget]      # deployment, worker_pool, orchestration_ceiling
    rules: list[ScalingRule]
    cooldown_seconds: int
    max_scale_up_step: int
    max_scale_down_step: int

ScalingRule:
    rule_id: str
    trigger: SignalTrigger | ScheduleTrigger | EventTrigger
    conditions: list[ScalingCondition]   # tenant, risk, time window, min replicas
    actions: list[ScalingAction]
    priority: int
```

### 11.2 Trigger types

| Trigger | Example |
|---------|---------|
| **Signal threshold** | `SIG_BACKPRESSURE > 10/min for 5m` |
| **Schedule** | Scale up 08:00 UTC weekdays |
| **Event** | `SLO_BREACH` runtime event |
| **Composite** | queue depth AND latency |

### 11.3 Condition examples

| Condition | Purpose |
|-----------|---------|
| `tenant_id in allowlist` | Noisy neighbor isolation |
| `replicas < max_replicas` | Hard cap |
| `cost_budget_remaining > 0` | FinOps gate |
| `application_profile == production` | Stricter HITL |

---
