# Enterprise Scale & Resilience — W5-H Final Observability Qualification & Deployment Readiness

**Task:** W5-H — Enterprise observability final qualification  
**Branch:** `development`

## ETAP 1 — Pipeline inventory (frozen)

```text
Event producer (RuntimeEvent publishers)
      |
      v
RuntimeEventBus          ownership: runtime/events — routing, subscribers, optional history
      |
      v
BoundedEventSink         ownership: event_delivery — queue, drain worker, backpressure
      |
      v
RuntimeEventExportSink   ownership: event_delivery — export bridge, failure isolation, flush/close
      |
      v
EventExportSinkPort      ownership: contracts + adapters (noop / recording / OTLP)
      |
      v
ObservabilityExportSinkFactory (EventExportSinkFactoryPort)
      |
      v
OtlpTransportPort        ownership: OTLP + Collector adapters (sync seam)
      |
      v
OTLP SDK / CollectorTransport → external collector
```

| Element | Owner | Lifecycle | Failure domain | Retry | Backpressure | Metrics |
|---------|--------|-----------|----------------|-------|--------------|---------|
| Producers | Execution / graph / host | N/A | Execution plane | N/A | Via sink disposition | Bus-local only |
| `RuntimeEventBus` | `intergrax/runtime/events` | `close()` → bounded sink | Handler errors isolated per subscriber | None on bus | Delegates to `event_sink` | Optional `InternalDeliveryMetrics` ref |
| `BoundedEventSink` | `event_delivery` | Worker drain → downstream `close()` | `CriticalEventDeliveryError` on critical reject | None | **Owner** — bounded `queue.Queue` | Via downstream bridge metrics |
| `RuntimeEventExportSink` | `event_delivery` | `flush_sync` → `export_sink.close()` | Swallows export errors | **None** | None (serial export on drain thread) | **Owner** — `export_*` counters |
| `EventExportSinkPort` | Adapters | `flush` / `close` | Transport errors → metrics | None in adapters | N/A | Bridge records |
| `ObservabilityExportSinkFactory` | Composition | Per wiring resolution | `ConfigurationError` at wire time | None | N/A | N/A |
| `OtlpTransportPort` | OTLP / Collector adapters | `flush` / `close` | `OtlpTransportError` / `DistributedTransportError` | **None** (explicit non-goal W5-E/F) | N/A | N/A |
| Composition root | `applications/_shared/runtime_event_delivery_wiring.py` | `close_application_runtime_event_delivery` | Misconfig fail-fast | N/A | Policy from env profile | One `InternalDeliveryMetrics` per stack |

**Composition root** constructs a new stack per `ApplicationEnvironmentProfile` resolution — no global registry, singleton transport, or background exporter scheduler.

## Profile qualification matrix

| Profile | Transport | Delivery | Expected |
|---------|-----------|----------|----------|
| `lab` | `NOOP` | best effort (bounded off) | PASS |
| `recording` | `RECORDING` | deterministic in-process | PASS |
| `production_slo` | `OTLP` (`OtlpTransport`) | bounded | PASS |
| `enterprise_cluster` | `DISTRIBUTED_OTLP` (`CollectorTransport`) | bounded → collector | PASS |

Qualification evidence: `tests/unit/runtime/observability/test_enterprise_scale_resilience_w5_h_final_qualification.py` and W5-A…G suites.

## Failure scenarios

### Collector unavailable

```text
Runtime
   |
   |  publish / handlers
   v
[ OK ]
   |
   X  (export)
Collector
```

| Expectation | Mechanism |
|-------------|-----------|
| Execution continues | Bus handlers run independently of export path |
| Event delivery isolated | `RuntimeEventExportSink` catches export failures |
| Metric increment | `InternalDeliveryMetrics.export_failed_total` |
| No exception leakage | Errors do not propagate to `publish` callers |

### Slow exporter (latency > producer rate)

| Check | Owner |
|-------|--------|
| Bounded queue | `BoundedEventSink` `max_capacity` |
| Backpressure | CRITICAL fail-closed · IMPORTANT defer · BEST_EFFORT drop |
| No unbounded memory | Fixed-size queue only |
| Execution plane not blocked | Async `publish` does not wait on OTLP |

### Shutdown ordering

```text
close()
   |
   v
bus.close()  →  bounded drain worker joins
   |
   v
RuntimeEventExportSink.close()
   |
   +-- flush_sync()  →  EventExportSinkPort.flush()
   |
   +-- export_sink.close()  →  transport flush/close (OTLP adapters)
```

## Contract stability (W5-H gate)

| Contract | Location | Verdict |
|----------|----------|---------|
| `EventExportSinkPort` | `intergrax/contracts/event_delivery.py` | Minimal async export / flush / close |
| `EventExportSinkFactoryPort` | `intergrax/contracts/observability_export.py` | Profile → sink only |
| `OtlpTransportPort` | `intergrax/contracts/observability_export.py` | Sync transport seam — SDK behind adapters |

No new manager/coordinator/registry abstractions in W5-H.

## Quality gates

| Gate | Command / scope |
|------|------------------|
| Ruff | `uv run ruff check` (changed paths) |
| Pyright | project check on runtime + tests |
| W5-A…H pytest | enterprise scale resilience + OTLP/distributed exporter tests |

## Cursor implementation audit (post W5-H)

| Section | Expected |
|---------|----------|
| STATUS | Qualification doc + test suite only |
| ARCHITECTURE | Ports → adapters → composition root unchanged |
| CONTRACTS | No contract expansion |
| REGRESSIONS | `production_slo` remains `OTLP`, not `DISTRIBUTED_OTLP` |
| TESTS | Six qualification tests in W5-H module |
| QUALITY | 0 failures, 0 new globals |
| RISKS | Long-running backpressure test uses intentional slow export |
