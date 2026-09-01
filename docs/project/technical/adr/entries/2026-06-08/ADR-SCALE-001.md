# ADR-SCALE-001: Harness Elastic Capacity Plane - complement K8s HPA, tier-separated provisioning

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-08 |
| **Deciders** | Harness platform architecture |
| **Related** | [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md) · Phase ECP-DOC · Phase ECP-DEPTH |

## Context

Intergrax must operate at organizational scale: more concurrent Nexus tasks, async workers, and agent host replicas as load grows. An architecture review (2026-06-08) found:

- **In-process** scaling controls exist (`max_inflight_nodes`, `GRAPH_BACKPRESSURE`, graph batch caps).
- **Async** queue workers exist (`intergrax/queueing`, Celery/Kafka/RabbitMQ) but with **fixed** worker counts unless operators intervene manually.
- **Observability** SLO catalog exists (Phase W-OPS Done) without closed-loop capacity provisioning.
- **`kubernetes` integration** (beta) provides health/deploy facade - not Harness-native scale rules.
- **Adaptive Harness Intelligence** proposes profile tuning (`max_parallel_nodes`) but does not provision replicas.
- **No** unified canon for elastic capacity; scaling concerns scattered across ORCHESTRATION, OBSERVABILITY, MODALITY, INTEGRATIONS.

Operators may already use **Kubernetes HPA**, **nginx** upstreams, or **Celery autoscale** outside Intergrax. The question is whether Intergrax needs its **own** capacity control plane.

Alternatives considered:

1. **Infrastructure-only scaling** - all elasticity delegated to K8s HPA / cloud autoscaler; Harness stays unaware.
2. **Fat Nexus** - `NexusLoop` calls K8s API when `GRAPH_BACKPRESSURE` fires.
3. **Tier-3 only** - each application documents Helm HPA; no Harness contracts.
4. **Harness Elastic Capacity Plane (ECP) (chosen)** - async Tier-1 controller; signals from runtime; actions via Integration Library; `ScalingProfile` on Tier-3.

## Decision

Adopt the **Harness Elastic Capacity Plane (ECP)** as a **Tier-1 async control plane** with **strict tier separation**:

| Tier | Owns |
|------|------|
| **Tier-0** | Integration adapters (`kubernetes`, `celery`, future `nginx`/`ingress`); optional read-only capacity tools |
| **Tier-1** | `CapacitySignalCollector`, `ScalingEvaluator`, `ScalingProvisioner`, `CapacityScheduler`; trace events |
| **Tier-2** | - (agents do not provision infrastructure) |
| **Tier-3** | `ScalingProfile`, deploy manifests (Helm/HPA YAML), min/max replicas, HITL policy for scale-up |

**ECP complements - does not replace - native autoscalers:**

- **K8s HPA** remains valid for CPU/memory/resource metrics on pods.
- **ECP** adds **Harness-aware signals**: sustained `GRAPH_BACKPRESSURE`, queue depth, task latency SLI, SLO breach, cost budget - and **unified audit** (`SCALE_*` events).

**Rejected:**

- **Infrastructure-only** - Harness operators cannot correlate agent load with capacity actions in one trace spine.
- **Fat Nexus** - violates domain-agnostic Nexus; blocks hot path; untestable at gate without network.
- **Tier-3 only** - duplicates policy per product; no reusable Harness capability.

**Core rules:**

1. Capacity mutations flow through **typed `ScalingAction`** + **integrations** - no vendor SDKs in `NexusLoop`.
2. ECP runs **async** (scheduler/worker) - never synchronously inside graph execution.
3. Scale-up in production defaults to **policy + optional HITL**; scale-down uses **hysteresis** and cooldown.
4. **Two dimensions stay separate:** ECP scales **execution capacity** (replicas/workers); [`REASONING_AND_COGNITION.md`](../../architecture/REASONING_AND_COGNITION.md) scales **agent topology** in plans.

## Consequences

### Positive

- Single canon domain pair (`ELASTIC_CAPACITY_AND_SCALING`) for authors and audits.
- Reuses W-OPS SLIs, runtime events, queueing, AHI signals, Integration Library.
- Enables governed “scale agents with load” without agent-specific Nexus branches.
- Clear boundary: ORCHESTRATION throttles; ECP provisions.

### Negative

- New Tier-1 module surface (`runtime/capacity`) - Phase ECP-DEPTH required for runtime value.
- Operators must configure **both** K8s HPA (optional) and Harness `ScalingProfile` - documentation must explain division.
- nginx integration not yet in catalog - ECP-6 backlog.
- Risk of flapping if signals and cooldowns misconfigured - mitigated by ECP-7 anti-flap guards.

## Compliance

- Tier boundaries preserved - integrations only in Tier-0; deploy YAML in Tier-3.
- Nexus remains domain-agnostic.
- Linked from architecture canon, plan Phase ECP-DOC, hub domain table.

## Implementation notes

- Architecture: [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md)
- Plan: Phase ECP-DOC (Done) · Phase ECP-DEPTH (Band 2ao, planned)
- Verification (when runtime lands): `tests/unit/runtime/capacity`; mock K8s in gate; no live cluster in CI
