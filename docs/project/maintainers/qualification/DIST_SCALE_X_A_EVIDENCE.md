# DIST-SCALE-X — Dimension A Evidence (P2)

| Field | Value |
| --- | --- |
| **Phase** | P2 |
| **Workload** | WL-1, WL-2 (slice) |
| **Execution SHA** | `db592f413c1f4de2744ffabbe3f2ff58189a353c` |
| **Threshold lock** | [DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md](DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md) — `ENVELOPE_ONLY`, enum `NOT_LOCKED` |
| **Agent verification** | `DIST_SCALE_X_P2_POINT_DURATION_SECONDS=8` (accelerated); LOCK canonical **180s** per point for L9 replay |

## Runs

| run_id | Path | Notes |
| --- | --- | --- |
| `wl01-scale-cea67e78df1f` | `.tmp/proof/dist-scale-x/wl01-scale-cea67e78df1f/` | WL-1 throughput envelope N=1 vs N=2, DocumentStore+PG workers |
| `wl02-hot-438181f1b54b` | `.tmp/proof/dist-scale-x/wl02-hot-438181f1b54b/` | Hot-tenant slice (dim A load) |
| `wl01-rabbit-b21aed41fc5e` | `.tmp/proof/dist-scale-x/wl01-rabbit-b21aed41fc5e/` | Rabbit M concurrent submit path |

**Correctness:** no `correctness_violations` in manifests; no `performance_threshold_metric_ids` on PASS.
