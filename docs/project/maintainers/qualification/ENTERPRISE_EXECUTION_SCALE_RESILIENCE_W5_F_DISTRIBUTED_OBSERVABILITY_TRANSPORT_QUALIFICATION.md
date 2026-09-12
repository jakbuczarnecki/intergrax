# Enterprise Scale & Resilience — W5-F Distributed Observability Transport Qualification

**Task:** W5-F — Distributed observability transport boundary  
**Baseline:** `development` @ `fff58c41d087b55f338de719c14e987729e15299`

## Ownership table (ETAP 1)

| Obszar | Owner |
|--------|--------|
| Event generation | runtime producers |
| Routing | `RuntimeEventBus` |
| Backpressure | `BoundedEventSink` |
| Export lifecycle | `RuntimeEventExportSink` |
| Transport | `OtlpTransportPort` |
| External delivery | W5-F `CollectorTransport` adapter |

## Pipeline (frozen)

```text
Execution Plane
      |
      v
RuntimeEventBus
      |
      v
BoundedEventSink
      |
      v
RuntimeEventExportSink
      |
      v
EventExportSinkPort (OtlpEventExportSink)
      |
      v
OtlpTransportPort
      |
      +----------------------+
      |                      |
      v                      v
OtlpTransport (local)   CollectorTransport (distributed)
                               |
                               v
                     Collector / broker boundary
```

## Lifecycle audit

| Question | Finding |
|----------|---------|
| Who creates transport? | `resolve_application_runtime_event_delivery_wiring` → `_create_export_transport` per `ExporterKind` (`OTLP` → `OtlpTransport`, `DISTRIBUTED_OTLP` → `CollectorTransport`). |
| Who owns lifecycle? | `ApplicationRuntimeEventDeliveryWiring.otlp_transport` (transport port reference); export sink closes transport on shutdown. |
| Who closes resources? | `RuntimeEventExportSink.close()` → `OtlpEventExportSink.close()` → `transport.close()`; `CollectorTransport` delegates flush-before-close to inner adapter. |
| Instance isolation? | Each `wire_application_environment()` / wiring resolution constructs a new transport; no global registry or shared mutable transport. |

## Delivery semantics (ETAP 8)

| Term | Meaning |
|------|---------|
| **Accepted** | Event accepted by local `BoundedEventSink` buffer (not external delivery). |
| **Exported** | Transport accepted the export call (OTLP SDK handoff at adapter boundary). |
| **Failed** | Transport rejected delivery (`DistributedTransportError` / `OtlpTransportError`); recorded on `InternalDeliveryMetrics.export_failed_total`. |

## Failure & backpressure

- Collector outage: `CollectorTransport.export()` raises `DistributedTransportError`; execution plane continues; metrics increment.
- Backpressure remains solely in `BoundedEventSink`; no `DistributedQueueManager` or broker-side pressure coupling.

## Quality audit (ETAP 12)

| Gate | Expected |
|------|----------|
| NEW_MANAGERS | NO |
| NEW_SCHEDULERS | NO |
| GLOBAL_SINGLETON | NO |
| Separate `DistributedEventTransportPort` | Not added — `OtlpTransportPort` reused |
