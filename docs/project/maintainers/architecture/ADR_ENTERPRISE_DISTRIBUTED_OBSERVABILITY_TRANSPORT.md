# ADR-ENTERPRISE-DISTRIBUTED-OBSERVABILITY-TRANSPORT: Distributed OTLP Transport Plugin

| Field | Value |
|-------|-------|
| **Status** | **Accepted** — W5-F freezes distributed collector export behind `OtlpTransportPort` |
| **Date** | 2026-09-12 |
| **Baseline** | W5-F on `development` (`fff58c41d087b55f338de719c14e987729e15299`) |
| **Related** | W5-E OTLP adapter · [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_F_DISTRIBUTED_OBSERVABILITY_TRANSPORT_QUALIFICATION.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_F_DISTRIBUTED_OBSERVABILITY_TRANSPORT_QUALIFICATION.md) |

---

## 1. Context

Enterprise deployments export runtime telemetry to a shared collector or broker. The execution plane must not import Kafka, OpenTelemetry clients, or broker SDKs directly.

---

## 2. Decision

Distributed transport is an **infrastructure plugin** at the composition root.

```text
Contract (OtlpTransportPort)
      |
      v
Adapter (CollectorTransport)
      |
      v
External system (OTLP collector)
```

**Not allowed:**

```text
Runtime → Kafka / OpenTelemetry client
```

| Layer | Role |
|-------|------|
| `intergrax/contracts/observability_export.py` | `ExporterKind.DISTRIBUTED_OTLP`, `OtlpTransportPort` |
| `intergrax/runtime/observability/exporters/distributed/` | `CollectorTransport`, `DistributedTransportConfiguration` |
| `runtime_event_delivery_wiring.py` | Selects local vs distributed OTLP per profile; one transport instance per environment wiring |

**Forbidden:** `ObservabilityManager`, global exporter registry, transport singletons, retry/backpressure in the distributed adapter, mixing distributed transport with execution admission.

---

## 3. Consequences

- `ExporterKind.OTLP` — process-local OTLP endpoint (`OtlpTransport`).
- `ExporterKind.DISTRIBUTED_OTLP` — collector boundary with required `observability_export_service_name`.
- OpenTelemetry SDK remains confined to the existing OTLP adapter; distributed adapter delegates serialization and SDK export to that layer.
