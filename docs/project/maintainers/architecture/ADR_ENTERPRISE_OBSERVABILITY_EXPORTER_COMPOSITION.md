# ADR-ENTERPRISE-OBSERVABILITY-EXPORTER-COMPOSITION: Observability Exporter Composition

| Field | Value |
|-------|-------|
| **Status** | **Accepted** — W5-D freezes exporter DI boundary |
| **Date** | 2026-09-12 |
| **Baseline** | W5-D on `development` (`f719a1bbaeb81308d9f1c977e57550a88cd445aa`) |
| **Related** | W5-C event export · W5-B2 composition wiring · [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_D_OTLP_EXPORTER_COMPOSITION_INVENTORY.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_D_OTLP_EXPORTER_COMPOSITION_INVENTORY.md) |

---

## 1. Context

W5-C introduced `EventExportSinkPort` and `RuntimeEventExportSink` but composition still constructed `NoopEventExportSink` inline. Enterprise deployments need OTLP (and future Kafka/NATS/Redis transports) without coupling exporters to execution runtime, retry, backpressure, or global lifecycle.

---

## 2. Decision

| Topic | Decision |
|-------|----------|
| Exporter selection | **Dependency injection boundary** — `EventExportSinkFactoryPort` maps `ObservabilityExportProfile` → `EventExportSinkPort`. |
| Owner | **`ApplicationEnvironmentWiring`** / `runtime_event_delivery_wiring.py` creates profile, factory, exporter, bridge, bounded sink, bus. |
| Not owners | `RuntimeEventBus`, `BoundedEventSink`, `RuntimeEventExportSink` do **not** choose or register exporter implementations. |
| OTLP SDK | Behind `OtlpTransportPort`; runtime ships adapter sink only. |
| Forbidden types | `ExporterManager`, `ExporterRegistry`, `GlobalExporterFactory`, `ObservabilityManager`, global singleton factories. |

---

## 3. Pipeline (frozen)

```text
RuntimeEventBus → BoundedEventSink → RuntimeEventExportSink → EventExportSinkPort → Exporter plugin
```

Export failures are recorded on `InternalDeliveryMetrics` only; they do not propagate to execution plane handlers.

---

## 4. Consequences

- Production can set `observability_exporter_kind=OTLP` with injected transport at composition edge (future step).
- Tests use `RECORDING` without OTLP SDK.
- Local / legacy paths keep `bounded_event_delivery_enabled=false` and noop export profile.
