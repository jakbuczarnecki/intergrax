# Enterprise Scale & Resilience — W5-C Downstream Event Export Inventory

**Task:** W5-C — Pluggable downstream event export sink pipeline  
**Baseline:** `development` @ `f9011e161f9667b4fdd051f10d49c6e43df40f07`

## Ownership table (ETAP 1)

| Element | Aktualny owner | Docelowy owner |
|---------|----------------|----------------|
| Event creation | Producer | Producer |
| Routing | `RuntimeEventBus` | `RuntimeEventBus` |
| Buffer/backpressure | `BoundedEventSink` | `BoundedEventSink` |
| Export | `AcceptingObservabilityEventSink` (`EventSinkPort` terminal) | `EventExportSinkPort` (+ concrete exporters) |
| Lifecycle | Composition (`ApplicationRuntimeEventDeliveryWiring`, `HarnessHostRuntime.close`) | Composition |
| Metrics | `InternalDeliveryMetrics` (bus publish path) | `InternalDeliveryMetrics` (+ export counters, diagnostic snapshot only) |

## Current pipeline (pre W5-C)

```text
RuntimeEventBus._deliver_through_event_sink
      |
      v
BoundedEventSink (queue + drain worker)
      |
      v
AcceptingObservabilityEventSink.publish(DeliverableEvent)
```

- **RuntimeEventBus** (`intergrax/runtime/events/event_bus.py`): optional `event_sink: EventSinkPort`; classifies `RuntimeEvent` → `DeliverableEvent` + `EventPriority`; never knows transport.
- **BoundedEventSink** (`bounded_event_sink.py`): sole backpressure owner; downstream `EventSinkPort`.
- **AcceptingObservabilityEventSink** (`accepting_event_sink.py`): sync terminal; acceptance counter only.
- **InternalDeliveryMetrics** (`delivery_metrics.py`): updated on bus publish result; not re-published on bus.
- **ApplicationEnvironmentWiring** / **`runtime_event_delivery_wiring.py`**: creates `AcceptingObservabilityEventSink` → `BoundedEventSink` → `RuntimeEventBus` when `bounded_event_delivery_enabled`.
- **HarnessHostRuntime.close** → `close_application_runtime_event_delivery` → `RuntimeEventBus.close()` → `BoundedEventSink.close()` → downstream `close()`.

## Target pipeline (W5-C)

```text
RuntimeEventBus
      |
      v
BoundedEventSink (RuntimeEvent + priority in queue)
      |
      v
RuntimeEventExportSink (EventSinkPort bridge)
      |
      v
EventExportSinkPort.export(RuntimeEvent)
      |
      +-- NoopEventExportSink
      +-- RecordingEventExportSink (tests)
      +-- OtlpEventExportSink (adapter seam)
```

## Shutdown order (target)

1. Stop accepting (`RuntimeEventBus.close` / bounded closed)
2. Drain bounded queue (worker)
3. `flush` exporter
4. `close` exporter
5. Runtime host teardown (exporter cannot outlive runtime bus close path)
