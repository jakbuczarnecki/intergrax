# Enterprise Scale & Resilience — W5-E OTLP Transport Integration Inventory

**Task:** W5-E — OpenTelemetry transport adapter integration & production observability activation  
**Baseline:** `development` @ `601454af605bd01556f5df7fc8bd43e8de3b032f`

## Ownership table (ETAP 1)

| Element | Owner (package) |
|---------|-----------------|
| `EventExportSinkPort` | `intergrax/contracts/event_delivery.py` |
| `OtlpEventExportSink` | `intergrax/runtime/observability/event_delivery` |
| `OtlpTransportPort` | `intergrax/contracts/observability_export.py` |
| OpenTelemetry adapter (`OtlpTransport`) | `intergrax/runtime/observability/exporters/otlp` |
| Composition | `ApplicationEnvironmentWiring` via `runtime_event_delivery_wiring.py` |

## Pipeline (frozen)

```text
Execution Runtime
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
EventExportSinkPort
      |
      v
OtlpEventExportSink
      |
      v
OtlpTransportPort
      |
      v
OtlpTransport (OpenTelemetry SDK exporter)
```

## Production profile & lifecycle audit

| Question | Finding |
|----------|---------|
| Where is production profile defined? | `GovernanceBundle.production_slo()` → `ObservabilityProfile` with `bounded_event_delivery_enabled=True`, `observability_exporter_kind=ExporterKind.OTLP` (`bundles.py`). |
| Who creates the exporter? | `resolve_application_runtime_event_delivery_wiring` → `ObservabilityExportSinkFactory.create` → `OtlpEventExportSink` when OTLP kind; `OtlpTransport` created when OTLP endpoint resolves from profile or host settings. |
| Who owns lifecycle? | `ApplicationRuntimeEventDeliveryWiring` holds `otlp_transport`, export sink, bridge, bounded sink; `HarnessHostRuntime.close` → `close_application_runtime_event_delivery` → bus → bounded drain → `RuntimeEventExportSink.close` → exporter → transport. |
| Who closes SDK resources? | `OtlpTransport.close()` shuts down `LoggerProvider` after `flush()`; no global singleton exporter. |

## Profile matrix

| Profile | `bounded_event_delivery` | `observability_exporter_kind` |
|---------|--------------------------|-------------------------------|
| `production_slo` | `True` | `OTLP` |
| `lab` | default `False` | `NOOP` |
| Tests / qualification | explicit override | `RECORDING` or `NOOP` |

## Invariants

- Backpressure, retry, and queueing remain solely in `BoundedEventSink`.
- OTLP transport maps events and invokes SDK export only; failures become metrics, not execution errors.
- No OpenTelemetry imports in execution plane, event bus, or bounded sink modules.
