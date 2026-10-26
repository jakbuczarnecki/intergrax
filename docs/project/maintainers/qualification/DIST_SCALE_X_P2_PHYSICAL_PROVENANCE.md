# DIST-SCALE-X P2 — Physical run provenance

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** (corrective batch) | `cc2d52245c1daf8c27bc1cfa1040f8885d3d7ee5` |
| **IMPLEMENTATION_SHA** (B01–B04/E01 core) | `16ad35b7b1cc651763a6320f4e7b86bcbb1f0748` |
| **EXECUTION_SHA** / **EVIDENCE_SHA** | `ee5a8371e2bf2203d420ea89c6b697ca2ee78d4c` |
| **THRESHOLD_LOCK_PUBLICATION_SHA** | `53aca60e554e906e8efc5747a6053aad9d322181` |

Prior acceptance @ `3cbd8bea0` / `cc2d52245` — **superseded** for B01–B04 findings (enqueue-only accounting); retained as diagnostic history only.

## Diagnostic runs (8s — not acceptance)

Short env overrides (`DIST_SCALE_X_P2_POINT_DURATION_SECONDS=8`, etc.) — smoke only; see `.tmp/session/dist-scale-x-p2-corrective/smoke-pytest.log`.

## Corrective acceptance @ EXECUTION_SHA `ee5a8371e`

| Field | Value |
| --- | --- |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p2_l3_l4_qualification.py -q --tb=line` |
| **Parameters** | `POINT=180`, `WL2_WARMUP=60`, `OVERLOAD=60` (defaults) |
| **Wall time** | ~1052s — `.tmp/session/dist-scale-x-p2-corrective-180s/full-qual-pytest.log` |
| **Infra** | `intergrax-postgresql` (5434→5432), `intergrax-rabbitmq` (5672) |
| **Workers** | OS subprocesses: `p2_multiprocess_worker` (`DocumentStoreTaskWorker`), `p2_rabbit_multiprocess_worker` (`create_rabbitmq_worker`) |
| **F02** | Four manifests below → `load_and_validate_manifest_file` PASS |

| run_id | Workload / dim | Manifest |
| --- | --- | --- |
| `wl01-scale-738d211df7ea` | WL-1 N=1+N=2 completed-work @ 180s | `.tmp/proof/dist-scale-x/wl01-scale-738d211df7ea/` |
| `wl01-rabbit-f416144afd42` | Rabbit M @ 180s, handler completion accounting | `.tmp/proof/dist-scale-x/wl01-rabbit-f416144afd42/` |
| `wl02-hot-3719cd260bf3` | WL-2 60+180s, cold progress + cross-tenant fail-closed | `.tmp/proof/dist-scale-x/wl02-hot-3719cd260bf3/` |
| `dim-c-3e7c5e570951` | dim C: N=2 baseline 180s + 1.5× overload 60s | `.tmp/proof/dist-scale-x/dim-c-3e7c5e570951/` |

`catalog_integrity_sha256`: `118155c03887d12d04184526018bf8020f7a124ceb513b39ec2c025e0c826268` (`NOT_LOCKED`).

## Corrective mechanisms (B01–B04)

| ID | Fix |
| --- | --- |
| **B01** | `p2_task_load.py` — completed-work throughput, in-flight=2, 30/70 mix @ ≥30 samples, drain/reconcile, p95 on completion latency |
| **B02** | Rabbit: 2 OS workers, per-PID completion logs, submitted == handler invocations, drain before PASS |
| **B03** | dim C: sustainable rate from **N=2 @ 180s**, 1.5× overload, depth samples + post-overload drain |
| **B04** | WL-2: per-tenant completions, `assert_cross_tenant_queue_fail_closed` on `DocumentStoreTaskQueue` |
| **E01** | `test_dist_scale_x_p2_e01_lkw_trace.py` — LOCK OD-02 harness; LKW `background_worker_main` = Kafka **R** (P2 excluded); PG/Rabbit M = platform consumer stack |

## Architecture path (as exercised)

| Stage | Implementation |
| --- | --- |
| Submit | `TaskRequest` → `DocumentStoreTaskQueue.enqueue` / Rabbit `create_rabbitmq_integration` |
| Admission | `wire_background_execution_admission_dependencies` + `admit_background_execution_handler` (workers) |
| Worker | `DocumentStoreTaskWorker.drain_once` / `create_rabbitmq_worker.process_message` |
| Durable state | PostgreSQL per-run schema |
| Tenant | `tenant_id` on requests; cross-tenant `get_status`/`get_result` fail-closed proof |
| OS processes | `subprocess.Popen` × N |

**E01 note:** Full LKW `background_worker_main` + `HostTaskExecutionPort` Kafka ingest is **not** replayed @ P2 (LOCK excludes Kafka R). Contract trace tests document equivalence boundary.
