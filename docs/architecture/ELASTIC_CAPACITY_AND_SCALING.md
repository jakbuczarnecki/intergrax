# Elastic Capacity and Scaling

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](../plan/ELASTIC_CAPACITY_AND_SCALING.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §0.3, §3.8, §12  
**Audit layers:** 30 (Operational Excellence) · cross-ref 9 (orchestration backpressure), 21 (observability SLIs)  
**Audit instruction:** [`audit/ELASTIC_CAPACITY_AND_SCALING.md`](../audit/ELASTIC_CAPACITY_AND_SCALING.md)  
**ADR:** [ADR-SCALE-001](../adr/entries/2026-06-08/ADR-SCALE-001.md)  
**Last updated:** 2026-06-20 — **P2-ARCH-11** ECP production boundary; honest maturity **Done**

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ELASTIC_CAPACITY_AND_SCALING canon).

- **Implement / audit default:** capacity adapter contracts (§1–§7). Extended §8+: [`arch/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md`](arch/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md).
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
| [`arch/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md`](arch/ELASTIC_CAPACITY_AND_SCALING_extended_depth.md) | extended depth |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


## Table of contents

1. [Purpose](#1-purpose)
   - [Production Boundary](#production-boundary)
   - [ECP responsibility boundary](#ecp-responsibility-boundary)
   - [Allowed ECP actions](#allowed-ecp-actions)
   - [Disallowed ECP actions](#disallowed-ecp-actions)
   - [Production readiness statement](#production-readiness-statement)
   - [Scaling action governance](#scaling-action-governance)
   - [Cursor review checklist](#cursor-review-checklist)
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

## Production Boundary

Elastic Capacity Plane manages **infrastructure capacity signals and scaling actions**. It **MUST NOT** decide agent topology, domain execution strategy, graph semantics, HITL policy, tool permissions or product workflow behavior.

ECP is the Harness **capacity architecture and governed scaling scaffold** — not an agent planner, graph scheduler, orchestration brain, business strategy layer, or drop-in replacement for Kubernetes HPA, Celery autoscaling or cloud-native autoscaling unless explicitly implemented, tested and documented as such.

**Cross-refs:** [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) §9 · [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md) · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §12.2 (S8) · [`ORCHESTRATION.md`](ORCHESTRATION.md) · [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary) · [`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine) · [`INTEGRATIONS.md`](INTEGRATIONS.md#integration-layer-contract) · [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md)

### Honest maturity snapshot

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

Do **not** market ECP-DEPTH as a finished production autoscaling system. Operators should continue to rely on **K8s HPA**, **Celery autoscale**, and manual runbooks until maturity and evidence statements justify otherwise ([Production readiness statement](#production-readiness-statement)).

---

## ECP responsibility boundary

| Concern | Owner |
|---|---|
| Capacity signal observation | ECP / observability / metrics |
| Worker/runner capacity recommendation | ECP |
| ScalingAction contract | ECP |
| Actual infrastructure mutation | Integration / platform deployment backend |
| Graph topology | Nexus / orchestration |
| Agent selection | Nexus / routing policy |
| Agent local step behavior | AgentEngine / Tier-2 agent |
| Tool side-effect execution | ToolRuntime |
| Product workflow behavior | Tier-3 application + Tier-2 agents |
| Policy/HITL boundaries | Runtime policy + Nexus/HITL |
| Cost/latency evidence | Observability / metrics / AHI where applicable |

---

## Allowed ECP actions

ECP **MAY**:

- observe capacity-related metrics,
- observe queue depth / runner saturation / worker availability if exposed,
- recommend scaling up or down,
- emit typed `ScalingAction` proposals,
- apply scaling actions only through approved integration/backend mechanisms when explicitly enabled,
- support backpressure decisions where architecture allows,
- report capacity limits,
- report cost/latency tradeoffs,
- integrate with observability and deployment backends,
- provide evidence for manual or governed scaling decisions.

---

## Disallowed ECP actions

ECP **MUST NOT**:

- decide which agent should handle a task,
- decide graph topology,
- modify Nexus plans,
- modify agent contracts,
- change ToolProfiles or tool permissions,
- alter HITL boundaries,
- bypass runtime policy,
- bypass observability,
- directly call cloud/vendor SDKs outside approved integrations,
- silently mutate production deployment capacity without explicit configuration/governance,
- claim production autoscaling readiness without maturity/evidence statement,
- replace Kubernetes HPA, Celery autoscaling or cloud-native autoscaling unless explicitly implemented, tested and documented as such.

---

## Production readiness statement

ECP should be treated as **capacity architecture and governed scaling scaffold** unless implementation, tests and deployment evidence prove otherwise.

For production deployments:

- use existing proven infrastructure autoscaling mechanisms where available,
- use ECP as observability/recommendation/governance layer unless explicitly enabled,
- do not advertise ECP as production autoscaler unless Production readiness is **P4** or higher and Evidence maturity is **E4** or higher,
- external enterprise claims require **P5/E5** according to [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md).

---

## Scaling action governance

- Scaling actions must be typed and traceable.
- Scaling actions must preserve correlation to capacity signals and triggering evidence.
- Production scaling mutations require explicit configuration or governance approval.
- Cost-impacting scaling should include budget/cost guardrails where available.
- Scale-down decisions must consider in-flight work and graceful drain semantics.
- Scaling failures must be visible through `RuntimeEvent` / observability spine.
- Any automatic production scaling mode must define rollback or emergency disable behavior.

---

## Cursor review checklist

Before adding or modifying ECP behavior, Cursor must verify:

- Is this capacity management, not orchestration?
- Does this change agent topology, routing or graph semantics? If yes, it does not belong to ECP.
- Is the scaling action typed?
- Is the triggering evidence traceable?
- Is production auto-scaling explicitly enabled?
- Are infrastructure mutations routed through approved integrations/deployment backends?
- Are cost/budget implications considered?
- Are scale-down and in-flight work risks considered?
- Is maturity stated using [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md)?
- Does the change avoid claiming HPA/Celery/cloud autoscaler equivalence without evidence?

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
