# DIST-SCALE-X P6 — Physical run provenance

## Disposition

**P6 M/N — PARTIAL (scoped)** — WL-7 **harness PASS** @ 120 s; **exit 11** (14400 s soak) **BLOCKED** until operator sets `DIST_SCALE_X_WL7_DURATION_SECONDS=14400`. Chaos matrix **4/4 PASS**.

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **EXECUTION_SHA (chaos + harness)** | `9ac74ca3849013e729dbc8c893f64427ef588f94` |
| **IMPLEMENTATION_SHA** | @ P7 publication commit on `development` (post batch-gate fix) |

Envelope: [DIST_SCALE_X_P6_ENVELOPE_RECORD.md](DIST_SCALE_X_P6_ENVELOPE_RECORD.md).

## Consolidated P6 batch

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/test_dist_scale_x_p6_qualification_batch.py -q` | **1 passed** (~225s) — `.tmp/session/dist-scale-x-p5-l9/p6-batch.log` |

## WL-7 soak (M)

| Field | Value |
| --- | --- |
| **Command** | `wl_07_soak_mixed_load.py` (via P6 batch) |
| **Duration** | 120 s (default) |
| **Result** | **skipped after harness** — manifest `soak_complete_envelope=0`; 0 errors @ 262 writes |
| **run_id** | `wl07-soak-41ad330f8ef2` |
| **p95_latency_ms** | ~463 (manifest) |

**Blocker:** full 4 h soak not executed in agent session.

## Chaos matrix (N)

| run_id | Seed |
| --- | --- |
| `chaos-N-01-9c41abea775e` | N-01 |
| `chaos-N-02-32821746debb` | N-02 |
| `chaos-N-03-c5498f0bdbcf` | N-03 |
| `chaos-N-05-6da48c1a08d1` | N-05 |

## Evidence docs

[M](DIST_SCALE_X_M_EVIDENCE.md) · [N](DIST_SCALE_X_N_EVIDENCE.md)
