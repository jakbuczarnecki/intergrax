# DIST-SCALE-X — Dimension A Evidence (P2)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `3cbd8bea0864a785c10cd39260345834a6e11dca` |
| **Acceptance window** | 180s per point (LOCK); log `.tmp/session/dist-scale-x-p2-180s/full-qual-pytest.log` |

## Authoritative runs

| run_id | Workload |
| --- | --- |
| `wl01-scale-6501c3f25785` | WL-1 throughput N=1/N=2 @ 180s each |
| `wl02-hot-d95a93037d25` | WL-2 hot 80% + 60s warmup + 180s measure |
| `wl01-rabbit-1ab20b1dbbf0` | Rabbit M submit load @ 180s |

Provenance: [DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md)
