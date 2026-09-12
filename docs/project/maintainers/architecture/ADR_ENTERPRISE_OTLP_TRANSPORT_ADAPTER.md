# ADR-ENTERPRISE-OTLP-TRANSPORT-ADAPTER: OpenTelemetry Transport Adapter

| Field | Value |
|-------|-------|
| **Status** | **Accepted** — W5-E freezes OTLP SDK adapter behind `OtlpTransportPort` |
| **Date** | 2026-09-12 |
| **Baseline** | W5-E on `development` (`601454af605bd01556f5df7fc8bd43e8de3b032f`) |
| **Related** | W5-D exporter composition · [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_E_OTLP_TRANSPORT_INVENTORY.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_E_OTLP_TRANSPORT_INVENTORY.md) |

---

## 1. Context

W5-D introduced `OtlpTransportPort` and `OtlpEventExportSink` without a concrete OpenTelemetry SDK adapter. Production observability must export runtime events over OTLP while keeping the execution plane free of vendor SDK coupling.

---

## 2. Decision

OpenTelemetry is an **infrastructure adapter**, not a core runtime dependency.

| Layer | Role |
|-------|------|
| `intergrax/contracts/observability_export.py` | `OtlpTransportPort`, `OtlpExportConfiguration`, `OtlpProtocol` |
| `intergrax/runtime/observability/exporters/otlp/` | `OtlpTransport` — `RuntimeEvent` → OTLP representation → SDK exporter |
| `ApplicationEnvironmentWiring` | Selects OTLP when profile + endpoint resolve; injects transport into `ObservabilityExportSinkFactory` |

```text
contracts (OtlpTransportPort)
      |
      v
runtime adapter (OtlpTransport)
      |
      v
OpenTelemetry SDK (optional infrastructure plugin)
```

**Not allowed:** `TelemetryManager`, `ObservabilityManager`, `GlobalTracer`, singleton exporters, retry/backpressure in the OTLP adapter.

---

## 3. Failure semantics

`OtlpTransport.export()` raises `OtlpTransportError`. `RuntimeEventExportSink` records `export_failed` on metrics; execution outcomes remain unchanged.

---

## 4. Consequences

- Production SLO profile can activate OTLP export when composition supplies endpoint configuration.
- Lab and tests default to `NOOP` or `RECORDING` without SDK.
- SDK import surface is confined to `exporters/otlp/otlp_transport.py` (HARDEN-3E allowlist).
