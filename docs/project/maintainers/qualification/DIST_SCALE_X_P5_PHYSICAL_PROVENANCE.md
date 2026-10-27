# DIST-SCALE-X P5 — Physical run provenance

## Disposition

**P5 I/J/L — READY FOR INDEPENDENT AUDIT (scoped)** — parent DIST-SCALE-X **OPEN**; not CLOSED; no FRZ PASS.

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **EXECUTION_SHA (physical)** | `9ac74ca3849013e729dbc8c893f64427ef588f94` |
| **IMPLEMENTATION_SHA** | @ P7 publication commit on `development` (post batch-gate fix) |

## Preflight L1/L2

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py -q` | **28 passed** — `.tmp/session/dist-scale-x-p5-l9/l1-l2.log` |

## P5 L3/L4 selection

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p5_qualification_batch.py -q` | **1 passed** (~99s inner selection) — session `2026-10-27` |
| Inner targets | `p5_network_unavailable`, `p5_j_evidence_contract`, `p5_resource_exhaustion` — **9 passed** |

Envelope: [DIST_SCALE_X_P5_ENVELOPE_RECORD.md](DIST_SCALE_X_P5_ENVELOPE_RECORD.md).

## Representative manifests (F02)

| run_id | Dimension | Module |
| --- | --- | --- |
| `p5-i-pg-4505025d49b1` | I-02 | `p5_network_unavailable` |
| `p5-i-rabbit-44661d76496e` | I-01 | `p5_network_unavailable` |
| `p5-l-adm-6b641a72fd3e` | L | `p5_resource_exhaustion` |

Local tree: `.tmp/proof/dist-scale-x/<run_id>/dist_scale_x_evidence_manifest.json` @ **EXECUTION_SHA**.

## Evidence docs

[I](DIST_SCALE_X_I_EVIDENCE.md) · [J](DIST_SCALE_X_J_EVIDENCE.md) · [L](DIST_SCALE_X_L_EVIDENCE.md)
