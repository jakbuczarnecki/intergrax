# DIST-SCALE-X — Dimension D Evidence (P3)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `5bc896e2797708f9b71efd1b6611bc910168e4be` |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P3-v1` |
| **Lock** | `ENVELOPE_ONLY` / `NOT_LOCKED` |

## Authoritative runs

| run_id | Workload | Notes |
| --- | --- | --- |
| `wl02-p3-3c61beaf2929` | WL-2 | Measured hot share **0.856** (target 0.85); p95 hot/cold; cold completed work |
| `wl03-hot-38a284878286` | WL-3 | 3124 submissions, **1** terminal SUCCEEDED; single logical effect |

Manifests: `.tmp/proof/dist-scale-x/<run_id>/dist_scale_x_evidence_manifest.json`

Provenance: [DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md)

## Invariants exercised

- Hot-partition scheduling via 17/20 rational mix (not P2 `tick % 10`).
- Idempotency claim / terminal replay under 4 OS workers × in-flight 2.
- No qualification-owned distributed locks or shadow idempotency stores.
