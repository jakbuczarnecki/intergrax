# DIST-SCALE-X — Capacity envelope (P7)

| Field | Value |
| --- | --- |
| **Mode** | `ENVELOPE_ONLY` / `NOT_LOCKED` (OD-04) |
| **SAC** | `SAC-DIST-SCALE-X-v1` |
| **AUDIT_BASELINE_SHA** | `IMPLEMENTATION_SHA` @ P7 publication (see [DIST_SCALE_X_P7_READINESS_RECONCILIATION.md](DIST_SCALE_X_P7_READINESS_RECONCILIATION.md)) |

## Tested topology (qual)

| Component | Configuration |
| --- | --- |
| PostgreSQL | `postgres:16.6`, host `:5434` |
| RabbitMQ | `rabbitmq:3.13-management` (**M**) |
| LKW workers | **2** baseline (P2/P6); **4** for WL-3-class |
| Tenants | **2–3** (WL-2/3/5 slices) |
| HTTP qual | loopback WL-4 idempotent POST |

## Observed points (not product SLO)

| Workload / slice | Duration / load | Notes |
| --- | --- | --- |
| P5 I PG pause | ~30 writes @ 0.1s | Errors during pause; recovery post-unpause |
| P5 I Rabbit | 25 s outage | Broker restore + readiness |
| P5 L admission | `max_concurrent_calls=1` | REJECT on second acquire |
| P6 WL-7 harness | **120 s** default PG write loop | Full **14400 s** envelope **pending** operator wall clock |
| P6 chaos N | 4 seeds | Delegates P4 G/F + P5 I |

Numeric caps remain **TO BE LOCKED** per [DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md](DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md) unless OD-THRESHOLD-PRODUCT.

## Shape properties (PASS without numeric lock)

- `CorrectnessInvariantViolation` empty on accepted physical runs.
- Admission overload → REJECT or bounded WAIT (L).
- No marketing throughput claim from qual envelopes alone.
