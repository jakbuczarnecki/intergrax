# DIST-SCALE-X — Dimension A Evidence (P2)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `ee5a8371e2bf2203d420ea89c6b697ca2ee78d4c` |
| **Acceptance window** | 180s per point; log `.tmp/session/dist-scale-x-p2-corrective-180s/full-qual-pytest.log` |

## Authoritative runs (corrective)

| run_id | Workload |
| --- | --- |
| `wl01-scale-738d211df7ea` | WL-1 **completed** throughput N=1/N=2 @ 180s (356/586 completed; eff ≈ 0.82) |
| `wl02-hot-3719cd260bf3` | WL-2 hot 80% + 60s warmup + 180s measure (535 completed; cold 70) |

Provenance: [DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md)
