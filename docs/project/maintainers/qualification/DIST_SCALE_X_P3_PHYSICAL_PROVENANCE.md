# DIST-SCALE-X P3 — Physical run provenance

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `fefa874c412519086caebbd13f1a53e47a9586ed` |
| **P2_RECONCILIATION + P3_ENVELOPE_PUBLICATION** | `e97b228f4f883b90bd562d9624e390ab53df7b79` |
| **IMPLEMENTATION_SHA** (Windows PG throttle) | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **EXECUTION_SHA** (WL-2/3/4 + HTTP WL-6) | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **IMPLEMENTATION_SHA** (WL-6 RabbitMQ M) | `fc5960a15b02c98e46abbce0534b63ac1322864a` |
| **AUDIT_CORRECTIVE_IMPLEMENTATION** | `4a790e8f70700257028aa2f7d344514cc6c6d080` |
| **AUDIT_CORRECTIVE_EXECUTION_SHA** | `e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c` |

## Preflight (structural)

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py -q` | 28 passed |

P2 physical batch **not** used as P3 preflight.

## Physical acceptance @ EXECUTION_SHA

| Field | Value |
| --- | --- |
| **Command** | Sequential `uv run pytest` on `DIST_SCALE_X_P3_SELECTION_BATCH` modules (WL-4 → WL-6 → WL-2 → WL-3; 45s cooldown between PG-heavy modules) |
| **Wall time** | ~24 min — `.tmp/session/dist-scale-x-p3-envelope/p3-physical-full.log` |
| **Infra** | `intergrax-postgresql` (16.6, 5434→5432), `intergrax-rabbitmq` (3.13-management) |
| **Envelope** | [DIST_SCALE_X_P3_ENVELOPE_RECORD.md](DIST_SCALE_X_P3_ENVELOPE_RECORD.md) |

| run_id | Module | Disposition |
| --- | --- | --- |
| `wl04-ok-540cfdfaef29` | WL-4 ok | PASS |
| `wl04-http-503-5a8a1f22bf5e` | WL-4 503 | PASS |
| `wl04-slow-deadline-4c2150a5ac05` | WL-4 slow | PASS |
| `wl04-fault-mix-0da741ea0740` | WL-4 mix | PASS |
| `wl04-lost-ack-045b1c18abb9` | WL-4 lost_ack | PASS |
| `wl06-storm-dda074987eb6` | WL-6 HTTP retry storm (supplementary @ `5bc896e27`) | PASS |
| `wl06-rabbit-df8123a9c607` | WL-6 Celery RetryPolicy @ Rabbit AMQP @ `e6020efc3` | PASS |
| `wl06-rabbit-61f709aa5dd6` | superseded (test redrive) @ `fc5960a15` | diagnostic |
| `wl02-p3-3c61beaf2929` | WL-2 85% measured | PASS |
| `wl03-hot-38a284878286` | WL-3 hot key | PASS |

`catalog_integrity_sha256`: `118155c03887d12d04184526018bf8020f7a124ceb513b39ec2c025e0c826268` (`NOT_LOCKED`).

## WL-2 measured hot share

**0.856** completed-work share (321 hot / 375 total); target 0.85 @ 60s warmup + 120s measure; **2** OS workers (Windows PG port mitigation; WL-3 retains 4 workers).

## WL-3 idempotency

3124 logical submissions → **1** `SUCCEEDED` task; `logical_effects` ≥ 1; workers=4, in_flight=2.

## Implementation notes

- Hot share: `p3_task_load.py` 17/20 schedule.
- WL-4: `p3_strict_host_invoker.py` + `p3_strict_http_stack.py` (`production_mode=True`). WL-6: `testing_support/dist_scale_x_p3_wl6_celery_app.py` + `wl_06_rabbit_broker_storm.py`. WL-2/3: `p3_task_load.py`, `p3_multiprocess_worker.py`, `require_p3_pg_infra`.
- Audit corrective record: [DIST_SCALE_X_P3_AUDIT_CORRECTIVE_CLOSURE.md](DIST_SCALE_X_P3_AUDIT_CORRECTIVE_CLOSURE.md).
- WL-6 Rabbit validation log: `.tmp/session/dist-scale-x-p3-wl6-rabbit/validation.log` (L1/L2 + physical).
- Selection: `DIST_SCALE_X_P3_SELECTION_BATCH` in `selection.py`; wrapper `test_dist_scale_x_p3_qualification_batch.py` (subprocess; do not nest inside full physical rerun in agent sessions).

## P2 evidence

Historical P2 manifests **retained**; paths unchanged unless code invalidation (none claimed for P2 queue workers).
