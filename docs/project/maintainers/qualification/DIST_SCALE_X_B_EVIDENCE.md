# DIST-SCALE-X — Dimension B Evidence (P2)

| Field | Value |
| --- | --- |
| **EXECUTION_SHA** (Rabbit M — authoritative) | `2f7b74300845ac31eeaae4bd676f75c0f6b690ff` |
| **EXECUTION_SHA** (DocumentStore scaling — unchanged) | `ee5a8371e2bf2203d420ea89c6b697ca2ee78d4c` |

## Authoritative Rabbit M run (final B02)

| run_id | Evidence |
| --- | --- |
| `wl01-rabbit-2b54945c3c0f` | 180s @ 2 OS workers; shared `SqliteDistributedKVStore`; `broker_terminal_succeeded` = `submitted` = 1407; `process_message` → KV `SUCCEEDED` + `get_result` |

Manifest: `.tmp/proof/dist-scale-x/wl01-rabbit-2b54945c3c0f/dist_scale_x_evidence_manifest.json`

## Historical (superseded for B02 terminal proof)

| run_id | Limitation |
| --- | --- |
| `wl01-rabbit-f416144afd42` @ `ee5a8371e` | Handler log only — not broker-terminal evidence |

DocumentStore scaling: `wl01-scale-738d211df7ea` @ `ee5a8371e` (unchanged).

Provenance: [DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P2_PHYSICAL_PROVENANCE.md)
