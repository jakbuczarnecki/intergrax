# DIST-SCALE-X P2 — Physical run provenance

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `db592f413c1f4de2744ffabbe3f2ff58189a353c` |
| **IMPLEMENTATION_SHA** | `53aca60e554e906e8efc5747a6053aad9d322181` |
| **EXECUTION_SHA** (full-duration acceptance) | `3cbd8bea0864a785c10cd39260345834a6e11dca` |
| **EVIDENCE_SHA** (manifest `code_sha`) | `3cbd8bea0864a785c10cd39260345834a6e11dca` |
| **THRESHOLD_LOCK_PUBLICATION_SHA** | `53aca60e554e906e8efc5747a6053aad9d322181` |
| **PUBLICATION_HEAD** | `3cbd8bea0864a785c10cd39260345834a6e11dca` |

## Diagnostic runs (8s — not acceptance)

| Executed tree | P2 committed? | Parameters | Limitation |
| --- | --- | --- | --- |
| `db592f413` + **uncommitted** P2 sources | No | `DIST_SCALE_X_P2_POINT_DURATION_SECONDS=8` | Manifests with `code_sha=db592f413` are smoke/diagnostic only |

## Full-duration acceptance @ EXECUTION_SHA `3cbd8bea0`

| Field | Value |
| --- | --- |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p2_l3_l4_qualification.py -q --tb=short` |
| **Parameters** | `POINT=180`, `WL2_WARMUP=60`, `OVERLOAD=60` |
| **Wall time** | ~861s (pytest log: `.tmp/session/dist-scale-x-p2-180s/full-qual-pytest.log`) |
| **Infra** | `intergrax-postgresql` (5434→5432), `intergrax-rabbitmq` (5672); images per LOCK |
| **Workers** | 1–2 OS subprocesses per test (`p2_multiprocess_worker`, `p2_rabbit_multiprocess_worker`) |
| **F02 validation** | All four manifests `load_and_validate_manifest_file` → empty violations |

| run_id | Workload / dim | duration_s (record) | Manifest path |
| --- | --- | --- | --- |
| `wl01-scale-6501c3f25785` | WL-1 N=1+N=2, A/B | 360 | `.tmp/proof/dist-scale-x/wl01-scale-6501c3f25785/` |
| `wl01-rabbit-1ab20b1dbbf0` | WL-1 Rabbit M, B | 180 | `.tmp/proof/dist-scale-x/wl01-rabbit-1ab20b1dbbf0/` |
| `wl02-hot-d95a93037d25` | WL-2, A/K | 240 (60+180) | `.tmp/proof/dist-scale-x/wl02-hot-d95a93037d25/` |
| `dim-c-5d269dfb599a` | WL-1 dim C | 60 overload step | `.tmp/proof/dist-scale-x/dim-c-5d269dfb599a/` |

`catalog_integrity_sha256` @ execution: `118155c03887d12d04184526018bf8020f7a124ceb513b39ec2c025e0c826268` (enum `NOT_LOCKED`).

## Architecture path (as exercised)

| Stage | Implementation |
| --- | --- |
| Submit | `TaskRequest` → `DocumentStoreTaskQueue.enqueue` (PG `create_postgresql_document_store`) or Rabbit `create_rabbitmq_integration` → `message_bus.enqueue` |
| Queue M | `DocumentStoreTaskQueue` + **RabbitMQ** `create_rabbitmq_worker` / consumer subprocesses |
| Worker | Subprocess `DocumentStoreTaskWorker.drain_once` → `execute_logical_task` → `TaskExecutionRegistry` handler |
| Durable state | PostgreSQL document-store schemas (per-run isolated schema, dropped after test) |
| Tenant | `tenant_id` on `TaskRequest`; WL-2 hot/cold mix 80% |
| OS processes | `subprocess.Popen` × N (independent workers) |

**Scoped deviation (documented):** qual harness uses `DocumentStoreTaskWorker` subprocess entrypoints, not full `local_workspace_application.host.background_worker_main` + `HostTaskExecutionPort` LKW binary. Same platform consumer stack (`colocated_worker` semantics). Independent audit may require LKW binary replay; not expanded in this batch.

## Integrity

- No `performance_threshold_metric_ids` on PASS.
- No `correctness_violations`.
- Record↔manifest binding validated (F02).
