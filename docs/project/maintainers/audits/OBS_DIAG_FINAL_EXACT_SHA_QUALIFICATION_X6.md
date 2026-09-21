# OBS/DIAG — Final Exact-SHA Enterprise Qualification (X6)

## QUALIFICATION_SHA

`67e5c1486b1eb126453ccf65ff204edfdb69d53b`

Supersedes invalidated closure at `d452ed0ff65b7e87a45b2dfb04ec7843c3e29afd` (audit: Agent/HITL repair drift + G9 not run on prior SHA).

## Branch / provenance (G0)

| Field | Value |
| --- | --- |
| branch | `development` |
| HEAD | `67e5c1486b1eb126453ccf65ff204edfdb69d53b` |
| origin/development | `67e5c1486b1eb126453ccf65ff204edfdb69d53b` |
| worktree during Phase B | clean at qualification commits (unrelated WIP may exist locally) |

## Repair lineage

| Milestone | SHA |
| --- | --- |
| FAILED_X6_1_SHA | `6c31c4840ba45c2ed449ba169a581086f9976bae` |
| X6_R1_REPAIR_SHA | `fb03ae9b4df475e398c40e5aa1398464928913e3` |
| FAILED_X6_2_SHA | `d008bec248cc96a8b660d36a86a476beee485a26` |
| SOURCE_TYPING_REPAIR_SHA | `701cdb7b30b45c863bf296a988f67604bb016ce5` |
| FINAL_REPAIR_SHA (superseded) | `d452ed0ff65b7e87a45b2dfb04ec7843c3e29afd` |
| X6_FINAL_R1_REPAIR_SHA | `3a343ca4800a371d61011a1a8763e21a49f06aeb` |
| X6_FINAL_R1_QUALIFICATION_SHA | `67e5c1486b1eb126453ccf65ff204edfdb69d53b` |

## Defect closure (Phase A + X6-FINAL-R1)

| Family | Found | Fixed | Remaining |
| --- | ---: | ---: | ---: |
| Agent/HITL contract drift | yes | yes | 0 |
| HITL governance/resume test wiring | yes | yes | 0 |
| Declarative HITL same-node replay / execution-tree resume | yes | yes | 0 |
| Governance `POLICY_DECISION` event id collision across runs | yes | yes | 0 |
| worktree pollution (`strict_host_kv.db`) | yes | yes | 0 |
| anti-drift gate (test/support Agent) | n/a | added | 0 |

### HITL repair summary

- **Root cause:** concrete test `Agent` subclasses (especially `_HitlAgent`) missing public `run()` after Tier-2 contract hardening; resume wiring and idempotency pre-effect overlay for declarative HITL.
- **Canonical abstraction:** `testing_support/nexus_hitl_test_agent.py`, `testing_support/nexus_lab_task_execution.py`.
- **Production `Agent` contract changed:** NO
- **Abstract contract weakened:** NO
- **Anti-drift gate:** `tests/unit/architecture/test_obs_diag_agent_run_contract_drift_gate.py`

## Phase B — full matrix (frozen SHA)

Command: single sequential `uv run pytest` invocation (151 tests) covering G1–G8 regression surfaces including Kafka (X5), SQLite (X5), HITL (X4/G6), HARDENING-3/vendor, X2/X3, 4B–4E, conformance, G6 debug API, and X6-FINAL-R1 HITL/nexus repair regressions.

| collected | passed | failed | skipped | exit |
| ---: | ---: | ---: | ---: | ---: |
| 151 | 151 | 0 | 0 | 0 |

Log: `.tmp/session/x6-final-r1/phase-b-full-matrix-r2.log`

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
| G9 static | PASS — `pyright` 0 errors + `ruff check` on X6 repair scope at `67e5c1486b1eb126453ccf65ff204edfdb69d53b` |
| G10 regressions | PASS |
| G11 docs | this artifact |

## Verdict

**PASS — OBSERVABILITY & DIAGNOSTICS ENTERPRISE QUALIFICATION COMPLETE** (code qualification at `QUALIFICATION_SHA`; independent GitHub audit still required).
