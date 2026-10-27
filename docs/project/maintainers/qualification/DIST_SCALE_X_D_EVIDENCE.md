# DIST-SCALE-X — Dimension D Evidence (P3)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** | `8656fb7d8c42e24e1448c4b7e97f18a15c1d765c` |
| **Prior EXECUTION_SHA** | `5bc896e2797708f9b71efd1b6611bc910168e4be` (historical) |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P3-v1` |
| **Lock** | `ENVELOPE_ONLY` / `NOT_LOCKED` |

## Authoritative runs

| run_id | Workload | Notes |
| --- | --- | --- |
| `wl02-p3-1a787bdee1a0` | WL-2 | hot share **0.851**; p95 hot/cold per-tenant (E01) |
| `wl03-hot-47ba51e7ad6a` | WL-3 | **1** semantic effect; ingress idempotency + 4×2 workers |

Manifests: `.tmp/proof/dist-scale-x/<run_id>/dist_scale_x_evidence_manifest.json`

Provenance: [DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P3_PHYSICAL_PROVENANCE.md)

## Invariants exercised

- Hot-partition scheduling via 17/20 rational mix (not P2 `tick % 10`).
- Idempotency claim / terminal replay under 4 OS workers × in-flight 2.
- No qualification-owned distributed locks or shadow idempotency stores.
