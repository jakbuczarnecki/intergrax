# DIST-SCALE-X — Dimension C Evidence (P2)

| Field | Value |
| --- | --- |
| **Phase** | P2 |
| **Workload** | WL-1 overload step |
| **Execution SHA** | `db592f413c1f4de2744ffabbe3f2ff58189a353c` |

## Backpressure step

| run_id | max_queue_depth observed | Result |
| --- | --- | --- |
| `dim-c-b7677adc6eb7` | bounded (&lt; 10000 guard) | PASS |

Manifest: `.tmp/proof/dist-scale-x/dim-c-b7677adc6eb7/dist_scale_x_evidence_manifest.json`

Overload: 1.5× sustainable submit @ N=2 worker topology (see LOCK record).
