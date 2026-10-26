# DIST-SCALE-X — Dimension K Evidence (P2 + P3)

## P2 (retained)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `ee5a8371e2bf2203d420ea89c6b697ca2ee78d4c` |

| run_id | Notes |
| --- | --- |
| `wl02-hot-3719cd260bf3` | ≥3 tenants, 80% hot; cold completed work; cross-tenant queue read fail-closed |

## P3 continuation (WL-2 @ 85%)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `5bc896e2797708f9b71efd1b6611bc910168e4be` |

| run_id | Notes |
| --- | --- |
| `wl02-p3-3c61beaf2929` | 85% measured hot share; cold tenant completions; tenant isolation negatives in module |

Manifest: `.tmp/proof/dist-scale-x/wl02-p3-3c61beaf2929/dist_scale_x_evidence_manifest.json`

Provenance: [DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md) · P2: [DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md)
