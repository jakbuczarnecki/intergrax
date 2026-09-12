# Enterprise Scale & Resilience — W5-D OTLP Exporter Composition Inventory

**Task:** W5-D — OTLP exporter plugin composition & distributed observability readiness  
**Baseline:** `development` @ `f719a1bbaeb81308d9f1c977e57550a88cd445aa`

## Ownership table (ETAP 1)

| Element | Owner (package) |
|---------|-----------------|
| `RuntimeEventBus` | `intergrax/runtime/events` |
| `BoundedEventSink` | `intergrax/runtime/observability/event_delivery` |
| `EventExportSinkPort` | `intergrax/contracts/event_delivery.py` |
| `RuntimeEventExportSink` | `intergrax/runtime/observability/event_delivery` (delivery bridge) |
| Composition wiring | `intergrax/applications/_shared/runtime_event_delivery_wiring.py` via `ApplicationEnvironmentWiring` |

## Pipeline (pre W5-D)

```text
Execution plane producers
      |
      v
RuntimeEventBus._deliver_through_event_sink
      |
      v
BoundedEventSink (queue + drain worker — sole backpressure)
      |
      v
RuntimeEventExportSink (EventSinkPort bridge)
      |
      v
EventExportSinkPort.export(RuntimeEvent)
      |
      +-- NoopEventExportSink (hard-coded in composition pre W5-D)
```

## Lifecycle audit (pre W5-D)

| Question | Finding |
|----------|---------|
| Who creates the exporter? | `resolve_application_runtime_event_delivery_wiring` instantiates `NoopEventExportSink()` directly — no factory boundary. |
| Who closes the exporter? | `HarnessHostRuntime.close` → `close_application_runtime_event_delivery` → `RuntimeEventBus.close()` → `BoundedEventSink.close()` → drain worker → `RuntimeEventExportSink.close()` → `flush` then exporter `close`. |
| Singleton? | No global exporter singleton; each `ApplicationRuntimeEventDeliveryWiring` owns instances per environment wire. |
| Per runtime instance swap? | Yes — distinct `wire_application_environment` calls yield distinct sinks; exporter kind was not configurable (always noop when bounded delivery on). |

## Gaps addressed in W5-D

- `ObservabilityExportProfile` + `EventExportSinkFactoryPort` at contracts boundary.
- `ObservabilityExportSinkFactory` in runtime (non-singleton).
- `OtlpTransportPort` seam (no direct OpenTelemetry SDK in runtime).
- `observability_exporter_kind` on `ObservabilityProfile`.
- Extended `InternalDeliveryMetrics` export counters + `exporter_kind` label.

## Target shutdown order (frozen)

1. Runtime shutdown (host)
2. `RuntimeEventBus.close()`
3. `BoundedEventSink.close()` (drain queue)
4. `RuntimeEventExportSink.flush_sync()` (exporter flush)
5. Exporter `close()`

**Invariant:** exporter never closes before bounded drain completes.
