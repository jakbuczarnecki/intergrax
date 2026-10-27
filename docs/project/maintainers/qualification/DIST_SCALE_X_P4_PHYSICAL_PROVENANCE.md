# DIST-SCALE-X P4-A — Physical run provenance

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `50838f077a70d35abb527fc4a499416f5971f1b0` |
| **IMPLEMENTATION_SHA** | *(see git commit containing this file)* |
| **EXECUTION_SHA** | Same as IMPLEMENTATION_SHA for consolidated batch below |
| **PUBLICATION_SHA** | Same commit as evidence publication |

## Preflight (structural)

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py -q` | 28 passed |
| `uv run pytest tests/unit/runtime/long_running/test_postgresql_task_checkpoint_store.py tests/unit/runtime/execution/test_postgresql_decision_checkpoint_persistence.py -q` | PG-gated unit negatives |

## Physical acceptance (consolidated batch)

| Field | Value |
| --- | --- |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p4_qualification_batch.py -q` |
| **Envelope override (batch driver)** | `DIST_SCALE_X_P4_WL5_WARMUP_SECONDS=2`, `DIST_SCALE_X_P4_WL5_MEASURE_SECONDS=3` (smoke — **not** full 60s/120s lock) |
| **Infra** | Local PostgreSQL 16.6 qual container; DSN via `INTERGRAX_POSTGRESQL_DSN` |
| **Families** | SX-F01 PG, SX-F03 PG, SX-F07/F09 via PG DocumentStore in `p4_persistence_op` |
| **catalog_integrity_sha256** | `118155c03887d12d04184526018bf8020f7a124ceb513b39ec2c025e0c826268` |

| Module | Disposition | Gap |
| --- | --- | --- |
| `wl_05_persistence_heavy.py` | Smoke PASS @ shortened envelope | Full 60s+120s @ 40 writes/s aggregate **pending** |
| `p4_physical_restart.py` | G/H partial PASS | F matrix / G kill / full H reload **open** |

## Evidence documents

- [DIST_SCALE_X_F_EVIDENCE.md](DIST_SCALE_X_F_EVIDENCE.md) — **incomplete**
- [DIST_SCALE_X_G_EVIDENCE.md](DIST_SCALE_X_G_EVIDENCE.md) — **partial**
- [DIST_SCALE_X_H_EVIDENCE.md](DIST_SCALE_X_H_EVIDENCE.md) — **partial**

## P4 / parent status

- **P4-A:** implementation delivered; physical qualification **partial** — not P4 scoped accepted.
- **DIST-SCALE-X parent:** **OPEN**
