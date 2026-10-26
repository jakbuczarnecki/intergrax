# DIST-SCALE-X P2 — Physical run provenance

## SHA lineage

| Label | SHA | Role |
| --- | --- | --- |
| **START_HEAD** | `db592f413c1f4de2744ffabbe3f2ff58189a353c` | Pre-P2 implementation (F01/F02 only) |
| **IMPLEMENTATION_SHA** | `53aca60e554e906e8efc5747a6053aad9d322181` | P2 qual code + LOCK record + evidence docs |
| **THRESHOLD_LOCK_PUBLICATION_SHA** | `53aca60e554e906e8efc5747a6053aad9d322181` | `DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md` |
| **PUBLICATION_HEAD** | `53aca60e554e906e8efc5747a6053aad9d322181` | Pushed to `origin/development` |

## Diagnostic runs (not full-duration acceptance)

| When | Executed tree | P2 files in tree? | Duration env | Notes |
| --- | --- | --- | --- | --- |
| Agent session @ `db592f413` | `db592f413` commit, **uncommitted** P2 working tree | **Yes — uncommitted** | `DIST_SCALE_X_P2_POINT_DURATION_SECONDS=8` | Smoke only; manifests under `.tmp/proof/dist-scale-x/` with `code_sha=db592f413` **do not** prove committed P2 @ full LOCK windows |

**Limitation:** Those manifests bind to `db592f413` while P2 sources were local-only; they are **diagnostic**, not L9 acceptance @ 180s.

## Full-duration acceptance runs

Recorded in section below after execution @ **EXECUTION_SHA** `53aca60e554e906e8efc5747a6053aad9d322181`.

**Command template:**

```powershell
$env:DIST_SCALE_X_P2_POINT_DURATION_SECONDS="180"
$env:DIST_SCALE_X_P2_WL2_WARMUP_SECONDS="60"
$env:DIST_SCALE_X_P2_OVERLOAD_SECONDS="60"
uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p2_l3_l4_qualification.py -q --tb=short
```

**Infra:** `intergrax-postgresql` (host 5434), `intergrax-rabbitmq` (5672); DSN per LOCK record.
