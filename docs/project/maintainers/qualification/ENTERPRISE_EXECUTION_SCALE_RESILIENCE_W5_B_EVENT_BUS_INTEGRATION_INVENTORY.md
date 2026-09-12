# Enterprise Scale & Resilience — W5-B Runtime Event Bus Integration Inventory

**Task:** W5-B — Runtime Event Bus → Bounded Event Sink Integration & Delivery Telemetry  
**Baseline:** W5-A `EventSinkPort` + `BoundedEventSink` on `development`.

## Component ownership

| Element | Owner | Current responsibility |
|---------|--------|----------------------|
| **RuntimeEventBus** | `intergrax/runtime/events/event_bus.py` | Pub/sub routing, durable evidence commit, optional `event_sink` transport (W5-B) |
| **Event producer** | Nexus publishers, `ObservabilityEmitter`, failure recorders, graph emitters | `RuntimeEvent` creation; no direct sink/backpressure |
| **Persistence** | `RuntimeEventPersistence` adapters + evidence durability rules | Mandatory/best-effort durable evidence (orthogonal to W5 transport) |
| **Delivery** | `EventSinkPort` implementations (`BoundedEventSink`, future brokers) | Bounded buffering, priority overflow, background drain |
| **NexusRuntimeEventPublisher** | `nexus/orchestration/task_events.py` | Scoped publish onto bus (identity, trace injection) |
| **RuntimeEventExecutionFailureEvidenceRecorder** | `execution/failure_evidence/runtime_event_recorder.py` | Failure evidence via `bus.record` |
| **FunctionalEvidenceRecorder** | `observability/functional_evidence_recorder.py` | Domain functional evidence emission |
| **ExecutionLineage** | `ExecutionLineagePersistence` contracts + stores | Lineage CAS — not observability sink |
| **Decision evidence** | Decision append/checkpoint planes (W3-C) | Append-only decision stream — not `RuntimeEventBus` sink |

## Target pipeline (W5-B)

```text
Producer → RuntimeEventBus → EventSinkPort → BoundedEventSink → Consumer
```

Producers **must not** call SQLite/Kafka/OTLP directly for observability transport. Backpressure, retry, and buffer ownership sit on the sink — not on producers.

## Pre-integration gap (closed in W5-B)

W5-A qualified transport-only contracts without wiring every bus emission through `EventSinkPort`. W5-B adds injectable `event_sink` on `RuntimeEventBus`, explicit `RuntimeEventType` priority table, internal `InternalDeliveryMetrics` (no recursive bus publish), and `close()` for sink lifecycle.

## Quality gates (W5-B)

| Check | Evidence |
|-------|----------|
| Bus uses sink on publish/record | `test_enterprise_scale_resilience_w5_b_event_bus_integration.py` Test 1 |
| Critical fail-closed on saturation | Test 2 |
| Best-effort drop | Test 3 |
| Execution isolation | Test 4 |
| Shutdown / no orphan worker | Test 5 |
| Metrics not on bus | Test 6 |
