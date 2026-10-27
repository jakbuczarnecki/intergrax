# DIST-SCALE-X — P3 envelope record

| Field | Value |
| --- | --- |
| **Record ID** | `DIST-SCALE-X-ENVELOPE-P3-v1` |
| **Operator authorization** | `OPERATOR APPROVED — DIST-SCALE-X P3 ENVELOPE RECORD v1 PUBLICATION` |
| **SAC** | `SAC-DIST-SCALE-X-v1` (OD-03) |
| **P2 lock reference** | [DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md](DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md) @ `53aca60e554e906e8efc5747a6053aad9d322181` (meaning unchanged) |
| **Lock mode** | `ENVELOPE_ONLY` |
| **Runtime catalog enum** | `DIST_SCALE_X_THRESHOLD_LOCK_STATE = NOT_LOCKED` |
| **Performance guards** | disabled — no `performance_threshold_metric_ids` on PASS |

## Infrastructure (OD-INFRA-P3)

| Service | Version |
| --- | --- |
| PostgreSQL | `postgres:16.6` (host 5434→5432) |
| RabbitMQ | `rabbitmq:3.13-management` |
| HTTP | local loopback only (qual server) |

Broker **M** = RabbitMQ; Kafka **R** excluded; Redis not authorized.

## WL-2 (P3 continuation)

| Parameter | Value |
| --- | --- |
| Tenants | ≥3 (1 hot + ≥2 cold) |
| Hot share | **85%** (17/20 rational schedule; measured from ledgers) |
| Warm-up | 60 s |
| Measurement | 120 s |

## WL-3

| Parameter | Value |
| --- | --- |
| Workers | 4 OS processes |
| In-flight per worker (driver) | max 2 |
| Warm-up | 30 s |
| Measurement | 120 s |
| Backends | PostgreSQL + DocumentStore TaskQueue |

## WL-4

| Parameter | Value |
| --- | --- |
| Duration per scenario class | 180 s |
| Fault mix (offered) | 10% HTTP 503; 5% slow > tool deadline; 2% lost-response-after-effect class |
| Tenants | ≥2 |
| Scope | strict-host effective `RuntimeToolInvoker` + registered HTTP qual tool + loopback |

## WL-6

| Parameter | Value |
| --- | --- |
| Transient fault mix | 20% |
| Fault window | 120 s |
| Drain cap | 600 s |
| Tenants | ≥2 |
| Broker | RabbitMQ M where broker path used |

## Manifest binding

All P3 L3/L4 runs: `qualification_phase: P3`; F02 env/topology equality; `catalog_integrity_sha256` @ execution SHA.
