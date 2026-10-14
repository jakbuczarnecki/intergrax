# PROD-Q-R1-R7-D2-R2 — Startup failure safety & shared pipeline regression closure

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** — independent GitHub SHA audit required; **not** CLOSED |
| **Branch** | `development` |
| **START_HEAD** | `2f9483d268f9ecc708e67f364ba1f88384168022` |
| **Parent** | **PROD-Q-R1-R7-D2** — **NOT CLOSED** |

## Corrective scope

| Finding | Remediation |
|---------|-------------|
| **D2-R2-B01** invocation store after slot reserve | `open_provider_invocation_store_at` moved into the same exception-safe startup acquisition block as continuation store; `_abort_production_opening_slot` on any startup failure in that block |
| **D2-R2-B02** six reference launcher failures | Baseline vs HEAD comparison — **identical** (`bad025f9` vs `2f9483d2`); root cause `MemoryProviderAdmissionError` (`qualification_evidence_missing`) in `wire_application_environment` / memory admission — **not** D2-R1 `run_deploy_and_activate` delta |

## Evidence logs

`.tmp/session/PROD-Q-R1-R7-D2-R2/`

| Command | Result |
|---------|--------|
| `uv run pytest tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py -p no:xdist -q` | **18 passed** |
| Reference launcher @ `bad025f9` (worktree) | **6 failed**, 2 passed (same six tests) |
| Reference launcher @ `2f9483d2` | **6 failed**, 2 passed |
| Consolidated regression bundle (D2 + reference + GCA + D1/D1b/R6 + prod_q) `-p no:xdist -q` | **110 passed**, **6 failed** (reference launcher), 2 skipped |
| `uv run pyright` on `production_process_lifecycle.py`, `process_lifecycle_resource_cleanup.py`, `deploy_activate_pipeline.py` | **0 errors** |

## FRZ-PRD (scoped — global OPEN)

| Criterion | D2-R2 contribution |
|-----------|-------------------|
| FRZ-PRD-04 | Opening-slot release on invocation-store acquisition failure |
| FRZ-PRD-05 | Owned-store cleanup preserved; caller-injected invocation store not closed |
| FRZ-PRD-08 | No new bypass or duplicate recovery mechanism |

## Recommended disposition

**D2-R2 READY FOR AUDIT** — B01 corrected with tests; B02 classified **TRACKED FREEZE DEBT** (memory provider qualification evidence for research reference bundles). Do **not** promote global FRZ or close D2 parent without independent review.
