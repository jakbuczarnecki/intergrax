# DIST-SCALE-X P6 — Physical run provenance

## Disposition

**P6 M/N — PARTIAL (scoped)** — WL-7 **harness PASS** @ 120 s; **exit 11** (14400 s soak) **BLOCKED** until operator sets `DIST_SCALE_X_WL7_DURATION_SECONDS=14400`. Chaos matrix **4/4 PASS**.

## SHA lineage

| Label | SHA |
| --- | --- |
| **START_HEAD** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **EXECUTION_SHA (chaos + harness)** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **IMPLEMENTATION_SHA** | @ P7 publication commit |

Envelope: [DIST_SCALE_X_P6_ENVELOPE_RECORD.md](DIST_SCALE_X_P6_ENVELOPE_RECORD.md).

## WL-7 soak (M)

| Field | Value |
| --- | --- |
| **Command** | `uv run pytest tests/qualification/dist_scale_x/wl_07_soak_mixed_load.py -q` |
| **Duration** | 120 s (default) |
| **Result** | **1 skipped** (expected) — manifest written; `soak_complete_envelope=0` |
| **run_id** | `wl07-soak-555c362086fc` (representative) |
| **Log** | `.tmp/session/dist-scale-x-p5-l9/p6-wl07-rerun.log` |

**Blocker:** full 4 h soak not executed in agent session.

## Chaos matrix (N)

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dist_scale_x/p6_chaos_matrix.py -q` | **4 passed** (~103s) — `.tmp/session/dist-scale-x-p5-l9/p6-chaos-rerun.log` |

| run_id | Seed |
| --- | --- |
| `chaos-N-01-8ebd1fee78b7` | N-01 |
| `chaos-N-02-90738f521683` | N-02 |
| `chaos-N-03-9b5aa4d0eeca` | N-03 |
| `chaos-N-05-ff575107fad1` | N-05 |

**Note:** First chaos attempt skipped (Rabbit down after prior outage); rerun after `docker start intergrax-rabbitmq`.

## Evidence docs

[M](DIST_SCALE_X_M_EVIDENCE.md) · [N](DIST_SCALE_X_N_EVIDENCE.md)
