# DIST-SCALE-X P5 — Physical run provenance

## Disposition

**P5 I/J/L — READY FOR INDEPENDENT AUDIT (scoped)** — parent DIST-SCALE-X **OPEN**; not CLOSED; no FRZ PASS.

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **EXECUTION_SHA (pre-publication)** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **IMPLEMENTATION_SHA** | @ P7 publication commit on `development` |

## Preflight L1/L2

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py -q` | **28 passed** — `.tmp/session/dist-scale-x-p5-l9/l1-l2.log` |

## P5 L3/L4 selection

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/p5_network_unavailable.py tests/qualification/dist_scale_x/p5_j_evidence_contract.py tests/qualification/dist_scale_x/p5_resource_exhaustion.py -q` | **9 passed** (~98s) — `.tmp/session/dist-scale-x-p5-l9/p5-inner-selection.log` |

Envelope: [DIST_SCALE_X_P5_ENVELOPE_RECORD.md](DIST_SCALE_X_P5_ENVELOPE_RECORD.md).

## Representative manifests (F02)

| run_id | Dimension | Module |
| --- | --- | --- |
| `p5-i-pg-dc2c42221919` | I-02 | `p5_network_unavailable` |
| `p5-i-rabbit-2ac2a37b3906` | I-01 | `p5_network_unavailable` |
| `p5-l-adm-f62a0e1a39b5` | L | `p5_resource_exhaustion` |

Local tree: `.tmp/proof/dist-scale-x/<run_id>/dist_scale_x_evidence_manifest.json` @ **EXECUTION_SHA**.

## Evidence docs

[I](DIST_SCALE_X_I_EVIDENCE.md) · [J](DIST_SCALE_X_J_EVIDENCE.md) · [L](DIST_SCALE_X_L_EVIDENCE.md)
