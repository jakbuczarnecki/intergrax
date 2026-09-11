# Enterprise Scale & Resilience — W5-A Observability Backpressure Inventory

**Task:** W5-A — Observability Backpressure Inventory & Event Pipeline Ownership  
**Status:** INVENTORY COMPLETE · delivery contracts + bounded sink implemented (transport only)  
**Production runtime changed:** YES (`intergrax/contracts/event_delivery.py`, `intergrax/runtime/observability/event_delivery/`)

Companion: [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md`](../architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md) · ADR [`ADR_ENTERPRISE_OBSERVABILITY_EVENT_PIPELINE_SCALING.md`](../architecture/ADR_ENTERPRISE_OBSERVABILITY_EVENT_PIPELINE_SCALING.md).

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `9a532d778538c212c8cf36d31c8eea49d9e2e5fa` |
| Branch | `development` |

## ETAP 1 — Event producer matrix

| Producer | Event type | Consumer | Persistence | Blocking on producer? |
|----------|------------|----------|-------------|------------------------|
| **ExecutionRuntime** / Nexus orchestration | `RuntimeEvent` (lifecycle, graph, ops) | `RuntimeEventBus` subscribers; export fan-out | `RuntimeEventPersistence` when catalog mandates; `EvidencePersistenceRequirement` gates fail-closed | **Sync `record`**: durable commit before handlers; async `publish` awaits handlers |
| **GraphExecutor** | `GRAPH_BACKPRESSURE`, node/task events | Bus + `capacity.event_bridge` metrics | Bus history / optional store | Async publish on graph path (not root admission) |
| **ObservabilityEmitter** | Trace + bus diagnostics | `RunTraceWriter`, in-memory trace, `RuntimeEventBus` | Trace persistence (Nexus tracing); bus evidence path | `record` on bus — same as bus |
| **Decision engine / host** | Decision lifecycle, finalization signals | Host callbacks + optional bus | **DecisionFinalizationPersistence** (terminal); **DecisionCheckpointPersistence** (snapshot) — not bus-owned | Host-defined; durable commit on finalization path |
| **External operation** (tool invoker) | Terminal / cancellation state | `tool_operation_termination` + execution boundary | External-operation CAS / worker terminal semantics (W2/W4) | Detached worker; caller not blocked on slow telemetry |
| **Lineage** | `ExecutionLineage*` records | Lineage stores / admission codecs | Dedicated lineage persistence (`execution/lineage/`) | Append on segment boundaries — not via observability bus |
| **Failure evidence** | `ExecutionFailureEvidenceRequest` → runtime events | `runtime_event_recorder` → bus | Mandatory vs best-effort via `evidence_durability` | Sync `bus.record` on failure path |
| **Audit** (`agent_governance/audit.py`) | Governance audit events | Audit sink `record` | Audit store (governance plane) | Sink-local |
| **Telemetry / export** | OTLP, JSONL, problem signals | `ObservabilityExporter` implementations | Export transport only — not execution truth | Export failures isolated per exporter health |
| **Causal evidence** | Causal evidence records | `CausalEvidencePersistence` adapters | Document/memory causal stores | Query/export separate from execution hot path |
| **Capacity plane** | Backpressure metric bridge | `CapacityCollector` | In-memory counters | Non-blocking handler on bus |

**Ownership rule (W5-A frozen):** checkpoint CAS, lineage CAS, external-operation terminal CAS, and decision finalization commits **remain outside** observability transport. W5-A adds **transport-only** `EventSinkPort` + `BoundedEventSink` — not a replacement for durable evidence stores.

## ETAP 2 — Event class separation

| Class | Examples | Semantics |
|-------|----------|-----------|
| **Critical** | `DECISION_FINALIZED`, `SECURITY_EVENT`, `EXTERNAL_OPERATION_TERMINAL`, `CHECKPOINT_COMMITTED`, `RECOVERY_STATE_CHANGE` | accept **or** fail-closed (`CriticalEventDeliveryError`) |
| **Important** | `metrics.*`, `timing.*`, performance / resource statistics | accept · bounded wait · defer (`EventDeliveryDisposition.DEFERRED`) |
| **Best effort** | debug trace, verbose telemetry, diagnostic breadcrumbs | accept · drop |

Catalog helpers: `CriticalEventKind`, `classify_kind_string`, `priority_for_critical_kind` in `intergrax/contracts/event_delivery.py`. Runtime bus retention (`RetentionClass`, `evidence_persistence_requirement`) remains authoritative for **durable** runtime events — delivery priority is an **additional** transport boundary.

## ETAP 3–4 — Contracts & backpressure

| Artifact | Location |
|----------|----------|
| `EventSinkPort`, `EventDeliveryPolicy`, `EventPriority`, `EventDeliveryResult` | `intergrax/contracts/event_delivery.py` |
| `BoundedEventSink` (bounded `queue.Queue`, overflow by priority) | `intergrax/runtime/observability/event_delivery/bounded_event_sink.py` |
| `InMemoryEventSink` (qualification / wiring) | `intergrax/runtime/observability/event_delivery/in_memory_sink.py` |

| Priority | Overflow |
|----------|----------|
| Critical | reject producer / `CriticalEventDeliveryError` |
| Important | bounded wait → defer |
| Best effort | drop |

**Forbidden (W5 guardrails):** `TelemetryManager`, `EventManager`, `ObservabilityManager`, unbounded `queue.put()` without `maxsize`.

## ETAP 5 — Execution isolation verdict

| Question | Answer |
|----------|--------|
| Can a slow telemetry consumer stop ExecutionRuntime / Recovery / Checkpoint / Cancellation / ExternalOperation? | **NO** — when producers use `BoundedEventSink`, best-effort publish returns without waiting on downstream; critical saturation fails closed without silent loss; drain runs on a dedicated worker thread. |
| Shared execution + telemetry queue? | **Forbidden** — W5-A buffer is observability-transport-only. |

**Gap (pre-integration):** `RuntimeEventBus` still synchronously commits durable evidence and dispatches handlers on the caller thread — W5-A qualifies the **replacement transport boundary**; bus integration is a follow-on wiring task.

## ETAP 6 — Durable evidence boundary

```text
event transport (EventSinkPort / BoundedEventSink)
        ≠
evidence persistence (RuntimeEventPersistence, checkpoint CAS, lineage CAS, external-operation terminal)
```

No changes to checkpoint CAS, lineage CAS, or external-operation CAS in W5-A.

## ETAP 8 — Qualification tests

| Case | Module |
|------|--------|
| Best-effort overflow | `test_enterprise_scale_resilience_w5_a_observability_backpressure.py` |
| Critical overflow | same |
| Slow consumer isolation | same |
| Critical ordering A→B→C | same |
| Cancellation during publish | same |

## Quality gates (W5-A)

| Gate | Status |
|------|--------|
| NEW_MANAGERS | NO |
| NEW_SCHEDULERS | NO |
| GLOBAL_SINGLETON_STATE | NO |
| ANY_ADDED | NO |
| DICT_STR_ANY | NO |
| TYPE_IGNORE | NO |
| GETATTR_SETATTR | NO |
