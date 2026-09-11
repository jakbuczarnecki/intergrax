# ADR-ENTERPRISE-OBSERVABILITY-EVENT-PIPELINE-SCALING: Observability Event Pipeline Scaling

| Field | Value |
|-------|-------|
| **Status** | **Accepted** — W5-A freezes local bounded transport contracts; distributed broker **not** implemented |
| **Date** | 2026-09-11 |
| **Baseline** | W5-A on `development` (`9a532d778538c212c8cf36d31c8eea49d9e2e5fa`) |
| **Related** | [`ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_A_OBSERVABILITY_BACKPRESSURE_INVENTORY.md`](../qualification/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_W5_A_OBSERVABILITY_BACKPRESSURE_INVENTORY.md) · NPSC-5F evidence plane · W1–W4 execution isolation |

---

## 1. Context

Observability emissions today fan out through `RuntimeEventBus`, `ObservabilityEmitter`, trace writers, and export adapters. Without an explicit delivery boundary, slow exporters or handlers can couple to execution threads via synchronous `record` / handler dispatch. W5-A introduces a **transport-only** contract (`EventSinkPort`) that is storage- and vendor-agnostic.

Durable truth remains in existing stores (checkpoint, lineage, finalization, mandatory runtime event persistence). This ADR covers **scaling the pipe**, not re-homing evidence ownership.

---

## 2. Local mode (implemented in W5-A)

| Decision | Choice |
|----------|--------|
| Buffer | Process-local `queue.Queue(maxsize=policy.max_capacity)` |
| Drainer | Single background worker thread per `BoundedEventSink` |
| Critical ordering | FIFO per sink instance (single consumer) |
| Critical saturation | Fail-closed (`CriticalEventDeliveryError`) |
| Important saturation | Bounded wait → `DEFERRED` |
| Best effort saturation | `DROPPED` |
| Shutdown | `close()` sends sentinel; worker joins — no orphan drain thread |

**Non-goals (W5-A):** wiring every bus producer through `BoundedEventSink`; OpenTelemetry/Kafka adapters.

---

## 3. Future distributed mode (design only)

| Option | Ordering | Partitioning | Retention | Replay | Failure semantics |
|--------|----------|--------------|-----------|--------|-------------------|
| **Kafka** | Per-partition total order | `tenant_id` / `run_id` key | Topic retention + compaction for critical kinds | Consumer offsets + optional compacted topic | Producer acks=all for critical; idempotent producer |
| **NATS JetStream** | Stream subject order | Subject hierarchy per execution | Max age / max bytes per stream | Durable consumers | Explicit ack / nak redelivery |
| **Redis Streams** | Stream ID order | Hash tag per run | `MAXLEN ~` + critical side stream | `XREADGROUP` pending | `NOACK` forbidden for critical |
| **Durable queue (cloud)** | Vendor-specific | Per-tenant shard | Policy per event class | DLQ + replay job | Critical → DLQ alerts, no silent drop |

### Cross-cutting decisions (frozen for follow-on)

1. **Ordering:** Critical kinds require per-`execution_id` (or `run_id`) single-writer partition; best-effort may cross partitions.
2. **Partitioning:** Never mix critical and best-effort in one unbounded shared partition without priority queues at the broker edge.
3. **Retention:** Critical ≥ evidence retention policy; best-effort ≤ 24h default.
4. **Replay:** Replay is a **downstream consumer** concern; replay ≠ checkpoint restore (W3 ADR).
5. **Failure semantics:** Critical publish failure surfaces to producer (fail-closed); important may shed load; best-effort drop with metrics.

---

## 4. Consequences

- Producers gain a swappable `EventSinkPort` implementation (in-memory, bounded local, future broker).
- Execution isolation tests qualify the bounded local sink; bus integration remains incremental.
- No new manager types; no global singleton event scheduler.

---

## 5. Compliance

| Check | W5-A |
|-------|------|
| Evidence ownership unchanged | YES |
| Unbounded queues in new code | NO |
| Distributed broker implemented | NO (ADR only) |
