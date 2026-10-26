# DIST-SCALE-X — Dimension B Evidence (P2)

| Field | Value |
| --- | --- |
| **Phase** | P2 |
| **Workload** | WL-1 |
| **Execution SHA** | `db592f413c1f4de2744ffabbe3f2ff58189a353c` |

## Horizontal scale proof

| run_id | Workers | scaling_efficiency (envelope) | dup FX |
| --- | --- | --- | --- |
| `wl01-scale-cea67e78df1f` | N=1 → N=2 OS processes (`p2_multiprocess_worker.py`) | ~0.54 @ 8s window (documented, not a numeric guard) | none observed |
| `wl01-rabbit-b21aed41fc5e` | 2 Rabbit consumer OS processes | throughput recorded | none observed |

Manifest: `.tmp/proof/dist-scale-x/wl01-scale-cea67e78df1f/dist_scale_x_evidence_manifest.json`
