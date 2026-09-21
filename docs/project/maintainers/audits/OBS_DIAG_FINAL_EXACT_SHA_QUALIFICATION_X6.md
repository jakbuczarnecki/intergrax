# OBS/DIAG — Final Exact-SHA Enterprise Qualification (X6)

## QUALIFICATION_SHA

`d452ed0ff65b7e87a45b2dfb04ec7843c3e29afd`

## Branch / provenance (G0)

| Field | Value |
| --- | --- |
| branch | `development` |
| HEAD | `d452ed0ff65b7e87a45b2dfb04ec7843c3e29afd` |
| origin/development | `d452ed0ff65b7e87a45b2dfb04ec7843c3e29afd` |
| worktree during Phase B | clean |

## Repair lineage

| Milestone | SHA |
| --- | --- |
| FAILED_X6_1_SHA | `6c31c4840ba45c2ed449ba169a581086f9976bae` |
| X6_R1_REPAIR_SHA | `fb03ae9b4df475e398c40e5aa1398464928913e3` |
| FAILED_X6_2_SHA | `d008bec248cc96a8b660d36a86a476beee485a26` |
| SOURCE_TYPING_REPAIR_SHA | `701cdb7b30b45c863bf296a988f67604bb016ce5` |
| FINAL_REPAIR_SHA | `d452ed0ff65b7e87a45b2dfb04ec7843c3e29afd` |

## Defect closure (Phase A)

| Family | Found | Fixed | Remaining |
| --- | ---: | ---: | ---: |
| Agent/HITL contract drift | yes | yes | 0 |
| HITL governance/resume test wiring | yes | yes | 0 |
| worktree pollution (`strict_host_kv.db`) | yes | yes | 0 |
| anti-drift gate (test/support Agent) | n/a | added | 0 |

### HITL repair summary

- **Root cause:** concrete test `Agent` subclasses (especially `_HitlAgent`) missing public `run()` after Tier-2 contract hardening.
- **Canonical abstraction:** `testing_support/nexus_hitl_test_agent.py` (`NexusBasicHitlTestAgent`, `NexusTimedHitlTestAgent`, `prepare_nexus_hitl_resume_task`).
- **Production `Agent` contract changed:** NO
- **Abstract contract weakened:** NO
- **Anti-drift gate:** `tests/unit/architecture/test_obs_diag_agent_run_contract_drift_gate.py`

## Phase B — full matrix (frozen SHA)

Command: single sequential pytest invocation (150 tests) covering G1–G8 regression surfaces including Kafka (X5), SQLite (X5), HITL (X4/G6), HARDENING-3/vendor, X2/X3, 4B–4E, conformance, G6 debug API.

| collected | passed | failed | skipped | exit |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 150 | 0 | 0 | 0 |

Log: `.tmp/session/obs-diag-x6-final/phase-b-full-matrix.log`

## Gate summary

| Gate | Result |
| --- | --- |
| G0 provenance | PASS |
| G1 contracts/composition | PASS (in matrix) |
| G2 adoption/zero-bypass | PASS (in matrix) |
| G3 diagnostics | PASS (in matrix) |
| G4 product host/read | PASS (in matrix) |
| G5 Kafka P4 | PASS (X5 integration) |
| G6 HITL P4 | PASS (X4 spine + long-running + G6 API) |
| G7 providers | PASS (X5/X5A + SQLite) |
| G8 vendor/layers | PASS (HARDENING-3 + vendor governance) |
| G9 static | not re-run as dedicated pyright/ruff batch in Phase B |
| G10 regressions | PASS |
| G11 docs | this artifact only |

## Verdict

**PASS — OBSERVABILITY & DIAGNOSTICS ENTERPRISE QUALIFICATION COMPLETE** (code qualification at `QUALIFICATION_SHA`; independent GitHub audit still required).
