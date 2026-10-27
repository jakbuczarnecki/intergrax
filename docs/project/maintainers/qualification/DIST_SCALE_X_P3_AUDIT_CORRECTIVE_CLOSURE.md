# DIST-SCALE-X P3 — Independent audit corrective closure

| Field | Value |
| --- | --- |
| **START_HEAD** | `3088c0efc093f67cc9653f7acf787473c97a3000` |
| **AUDIT_IMPLEMENTATION_SHA** | `4a790e8f70700257028aa2f7d344514cc6c6d080` |
| **AUDIT_EXECUTION_SHA** (WL-6 Celery + WL-4 strict-host slice) | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` |

## B01 — WL-6 retry ownership

| Item | Resolution |
| --- | --- |
| Finding | Test-loop `_redrive_failed()` on `create_rabbitmq_worker` + `auto_ack` is not canonical bounded retry. |
| Canonical owner | `intergrax.queueing.worker.dispatcher` + `RetryPolicy` via `create_celery_worker_app` (composition root in `queueing/worker_bootstrap.py`). |
| Broker | RabbitMQ M as **AMQP** (`amqp://…@localhost:5672`). |
| Qual harness | `testing_support/dist_scale_x_p3_wl6_celery_app.py` (outside P1 authority scan). |
| Authoritative run | `wl06-rabbit-df8123a9c607` @ `e6020efc3` — 60 logical submissions; 74 physical attempts; 14 dispatcher retries; amp **1.23**; semantic effects **60**. |
| Historical (superseded) | `wl06-rabbit-61f709aa5dd6` @ `fc5960a15` — test redrive; retain as diagnostic only. |
| `create_rabbitmq_worker` poll | Documented **non-retrying** (`RabbitMQMessageConsumer` `auto_ack=True`; terminal FAILED on handler error). |

## B02 — WL-3 exactly-one effect

| Item | Resolution |
| --- | --- |
| Code | `p3_multiprocess_worker.py` — `semantic_effects` table (one row per idempotency key). |
| Assert | `effects == 1` in `wl_03_hot_key_contention.py`. |
| Ingress | DocumentStore queue collapses duplicate `idempotency_key` to one `task_id`; multi-process contention exercises worker claim path. |

## B03 — WL-4 strict-host

| Item | Resolution |
| --- | --- |
| Code | `p3_strict_host_invoker.py` — `production_mode=True`, strict durable env, HTTP capability governance, MSE boundary. |
| Negative | `test_p3_wl04_strict_host_mse_denies_without_authorization` — `ToolGovernanceDeniedError` (test-mse). |
| Lost-ack | Separate evidence `wl04-lost-ack-*` (not mixed-only). |

## E01 — WL-2 p95 attribution

| Item | Resolution |
| --- | --- |
| Code | `completion_latencies_by_tenant` in `p2_task_load.py`; WL-2 uses per-tenant lists for p95 hot/cold. |
| Physical regen | Required @ `e6020efc3` when PostgreSQL host path healthy — **pending** if `127.0.0.1:5434` handshake fails. |

## Infra recovery (2026-10-27)

| Check | Result |
| --- | --- |
| Docker | 27.5.1; `intergrax-postgresql` healthy (`postgres:16.6`, `0.0.0.0:5434→5432`) |
| In-container | `pg_isready` + `SELECT 1` OK |
| Host `127.0.0.1:5434` | TCP OK via `wslrelay`, **PostgreSQL handshake timeout** (stale `CLOSE_WAIT`) |
| Host `172.26.96.1:5434` | **psycopg OK** (hits `com.docker.backend` publish) |
| Qual run | `INTERGRAX_POSTGRESQL_DSN=postgresql://intergrax:intergrax@172.26.96.1:5434/intergrax` |
| RabbitMQ | container ping OK; host `pika` AMQP **not ready** — **not required** for WL-2/3 |

`pg_port_open()` now uses psycopg handshake (qual helper).

## WL-2 / WL-3 @ `8656fb7d8`

| run_id | Result |
| --- | --- |
| `wl02-p3-1a787bdee1a0` | hot share **0.851**; p95 hot **656 ms**; p95 cold **688 ms** |
| `wl03-hot-47ba51e7ad6a` | `logical_effects` **1**; workers **4**; submissions **3185** |

Log: `.tmp/session/dist-scale-x-p3-infra-recovery/wl02-wl03-physical.log`

## B01 — AMQP vs TaskQueue (broker M)

| Path | Role |
| --- | --- |
| **RabbitMQ as AMQP** + `create_celery_worker_app` | Approved WL-6 retry owner (`RetryPolicy` / dispatcher); evidence `wl06-rabbit-df8123a9c607` @ `e6020efc3` |
| **`create_rabbitmq_worker` + `RabbitMQMessageConsumer` (`auto_ack`)** | P2 throughput / KV terminal path; **no** broker redelivery; not a retry storm owner |

## Validation log

`.tmp/session/dist-scale-x-p3-audit-corrective/` · `.tmp/session/dist-scale-x-p3-infra-recovery/`
