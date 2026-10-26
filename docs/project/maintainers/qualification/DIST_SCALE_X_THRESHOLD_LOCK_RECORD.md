# DIST-SCALE-X — Threshold Lock Record (P2)

| Field | Value |
| --- | --- |
| **Record ID** | `DIST-SCALE-X-THRESHOLD-LOCK-P2-ENVELOPE` |
| **Operator approval** | `OPERATOR APPROVED — DIST-SCALE-X THRESHOLD LOCK RECORD PUBLICATION — P2 ENVELOPE_ONLY` |
| **SAC contract** | `SAC-DIST-SCALE-X-v1` (OD-03) |
| **Lock mode** | `ENVELOPE_ONLY` (OD-04) |
| **Runtime catalog enum** | `DIST_SCALE_X_THRESHOLD_LOCK_STATE = NOT_LOCKED` (numeric performance lock **not** activated @ P2) |
| **Publication** | See Git commit that adds this file (`THRESHOLD_LOCK_PUBLICATION_SHA` in P2 evidence) |
| **Product SLO** | **Not asserted** |
| **Performance guards** | **Disabled** (`performance_guards_enabled: false`) |

## Broker and host (OD-01, OD-02)

| Role | Backend |
| --- | --- |
| Mandatory **M** | RabbitMQ (`BK-RABBIT`, `RabbitMQTaskQueue`) |
| Representative **R** | Kafka (`BK-KAFKA`) — **excluded** from P2 mandatory physical runs |
| Representative host | LKW — multi-process worker topology via platform `DocumentStoreTaskWorker` / Rabbit worker processes (qual harness) |

## Infrastructure (OD-INFRA-P2)

| Service | Image (compose pin) | Port |
| --- | --- | --- |
| PostgreSQL | `postgres:16.6` | host **5434→5432** when using existing `intergrax-postgresql` container |
| RabbitMQ | `rabbitmq:3.13-management` | 5672 / 15672 |

**Start (explicit services only):** `docker compose --profile core up -d postgresql rabbitmq` from `infra/integration/`.

**DSN (qual default @ this host):** `postgresql://intergrax:intergrax@localhost:5434/intergrax` (mapped port for `intergrax-postgresql`)

## Workload parameters — WL-1 (P2)

| Parameter | Value |
| --- | --- |
| Worker measurement points | **N=1** and **N=2** mandatory; **N=4** optional |
| In-flight per worker | **2** |
| Tenants | **2** |
| Duration per measured point | **180** seconds |
| Task mix | 30% read-status / 70% write-admit (WL-1 spec) |
| Payload | **1024** bytes |
| Fault injection | **0%** |

## Workload parameters — WL-2 (P2 slice)

| Parameter | Value |
| --- | --- |
| Tenants | **≥3** (1 hot, ≥2 cold) |
| Hot traffic share | **80%** |
| Warm-up | **60** seconds |
| Measurement | **180** seconds |

## Dimension C (P2)

| Parameter | Value |
| --- | --- |
| Overload steps | **1** bounded step |
| Offered load | **1.5×** measured sustainable submit rate at **N=2** |
| Step duration | **60** seconds |

## Metrics (envelope — record, do not treat as SLO)

- Useful **total** throughput (`M-WL1-THR`)
- Worker count **N**
- In-flight and offered load
- p95 latency (`M-WL1-LAT-P95`)
- Error rate (non-transient @ 0% fault baseline)
- Max queue depth (`M-WL1-QD`)
- Scaling efficiency `throughput(N) / (N × throughput(1))` (`M-SCALE-EFF`) — **document only**; not a pass/fail numeric guard
- Tenant isolation / cold progress (WL-2 / dim K)
- `CorrectnessInvariantViolation` set (must be empty for PASS)

## Correctness (always active)

Automatic FAIL on any SAC §3.1 invariant. No `performance_threshold_metric_ids` on PASS records while enum remains `NOT_LOCKED`.

## Manifest binding

P2 runs must set:

- `qualification_phase`: `P2`
- `environment_definition` and `topology_description` matching this record and the executed compose/worker topology
- `catalog_integrity_sha256` at execution `code_sha`
