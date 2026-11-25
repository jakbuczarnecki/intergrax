# PT-V04 S1 diagnostic baseline replay (P1)

**Execution SHA:** `4b7acac4ba32326b32f446e23d624f91393d963e`  
**Branch / worktree:** `af-fc/v04` @ `D:/Projekty/_wt/intergrax/af-fc-v04`  
**Recorded:** 2026-11-25  
**Context:** S1 non-certifying diagnostic per WP12 `PT-V04` (ROADMAP-REPLAY-X, STATE-X/TRACE-X, TENANT-X, F10)  
**Scratch stdout:** `D:/Projekty/_wt/intergrax/_scratch/af-fc-v04/`

## Prior record check

No in-repo `PT_V04_REPLAY.md` cited this SHA family with a completed four-check diagnostic. RC0 documents `BLOCKED_EVIDENCE` only; ledger `PT-V04` remains `PARTIAL`. This run is fresh at the base commit (not a HEAD cite).

## Commands (existing published entry points)

| Check | Command | Scratch log |
| --- | --- | --- |
| F10 `--check` | `uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py --check` | `F10_check.stdout.txt` |
| ROADMAP-REPLAY-X counters | `uv run pytest -q --tb=line tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py` | `ROADMAP_REPLAY_X.stdout.txt` |
| STATE-X / TRACE-X (7 qualification nodes) | `uv run pytest -q --tb=line tests/qualification/state_x/test_state_x_p0_baseline.py tests/qualification/trace_x/test_trace_x_p0_baseline.py tests/qualification/trace_x/test_trace_x_p6_closed_world_gates.py` | `STATE_TRACE_X.stdout.txt` |
| `tenant_x` | `uv run pytest -q --tb=line tests/qualification/tenant_x/test_tenant_x_p0_fast.py` | `tenant_x.stdout.txt` |

Canonical bundled runner (not used here; same pytest set + F10): `scripts/qualification/run_pt_v04_s1_baseline.py`.

## Results

| Check | Exit code | Verdict | Summary |
| --- | ---: | --- | --- |
| F10 `--check` | 0 | **PASS** | `IN SYNC` |
| ROADMAP-REPLAY-X counters | 1 | **FAIL** | 3 failed, 21 passed in `mechanical_exit_gates` (prod-Δ / closure counter / B6 R2) |
| STATE-X / TRACE-X | 0 | **PASS** | 102 passed |
| `tenant_x` | 0 | **PASS** | 64 passed |

## Notes

- Diagnostic only; no product or test edits.
- ROADMAP-REPLAY-X failures are recorded in scratch stdout; disposition belongs to SCOPE / W3, not this replay commit.