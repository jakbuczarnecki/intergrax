# ROADMAP-REPLAY-X — B6 R2 consolidated correction verdict

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` |
| **B6 IMPLEMENTATION_SHA** | `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` |
| **B6_R2_CORRECTION_SHA** | `6abe074eacc4be957b90a57bece0e4d6d4d102f1` |
| **B6_R2_TEST_EXECUTION_SHA** | `6abe074eacc4be957b90a57bece0e4d6d4d102f1` |
| **B6 R2 L1 (preliminary @ AUDITED_HEAD)** | `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` — 31 passed @ `.tmp/proof/roadmap-replay-x-b6-r2-exec-5f54ba76/` |
| **B6 R2 L1 (immutable @ correction SHA)** | **32 passed**, exit **0** — log SHA256 `3b3a7706f9d5d542315ffd4dbdafaa275c79de3201f6a7395b30eaf98315218c` @ `.tmp/proof/roadmap-replay-x-b6-r2-immutable-6abe074/` |
| **Publication range** | `5f54ba76d68e9065256ac4597bc8e8d9e012b6d2` → `6abe074eacc4be957b90a57bece0e4d6d4d102f1` on `development` |
| **Verdict** | **ISSUED — PASS (SCOPED)** — **B6 CLOSED**; parent **CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)** under **OD-REPLAYX-08** |

## R2 scope closure

| ID | Outcome |
| --- | --- |
| R2-01 | Bidirectional blocker/reopen/ECAP-counter reconciliation in `_replay_bidirectional_reconciliation.py` + negative fixtures |
| R2-02 | Independent discovery in `_replay_source_adapters.py`; ledger inventories reconciled both directions |
| R2-03 | Scoped regression chain in `_scoped_regression_evidence.py` (named test, batch, SHA, log digest / approved ref) |
| R2-04 | SHA ladder documents `b7bf014` vs `5f54ba76`; immutable L1 **32 passed**, exit **0** @ `B6_R2_TEST_EXECUTION_SHA` |
| R2-05 | ECAP / Protocol 03 / closure matrix / drift log AUDITED_HEAD aligned; gates stay OPEN |
| R2-06 | This verdict — superseded by independent **B6 PASS (SCOPED)** + **OD-REPLAYX-08** |

## Scoped regression proof table

| Prod path | Historical invariant | Regression test (exists) | L2 batch | Exit | TEST_EXECUTION_SHA | Log / ref |
| --- | --- | --- | --- | ---: | --- | --- |
| `intergrax/runtime/resilience/dependency_attempt_execution_boundary.py` | CTRL-X admission/shutdown unchanged | `test_ctrl_x_semantic_boundary_pyright_zero_errors` | `uv run pytest tests/qualification/control_planes/ctrl_x/ -q` | 0 | `22fbc0e837e819fe2a068295df5f527612dcf307` | `l2-ctrl_x.log` SHA256 `c9567cff…` @ immutable 6fbc4a8 + drift log ref |

## Six counters @ R2 L1 HEAD

| Counter | Value |
| --- | ---: |
| reopened_historical_stages | 0 |
| unresolved_replay_blockers | 0 |
| roadmap_drift | 0 |
| historical_evidence_invalidated_by_later_changes | 0 |
| unclassified_replay_findings | 0 |
| vendor_neutrality_replay_failures | 0 |

## Independent auditor brief

| Check | Evidence |
| --- | --- |
| Correction landed | `git log -1 6abe074eacc4be957b90a57bece0e4d6d4d102f1` — `fix(qual): ROADMAP-REPLAY-X B6 R2 evidence-grounded closure (R2-01..R2-06)` |
| Diff scope | `git diff 5f54ba76d68e9065256ac4597bc8e8d9e012b6d2..6abe074eacc4be957b90a57bece0e4d6d4d102f1` — 19 paths (R2 modules, fixtures, mechanical tests, ECAP/Protocol03/closure/drift/ladder/verdict) |
| L1 replay batch | `uv run pytest tests/qualification/roadmap_replay_x/ -q` @ **`B6_R2_TEST_EXECUTION_SHA`** → **32 passed**, exit **0** |
| Log integrity | `.tmp/proof/roadmap-replay-x-b6-r2-immutable-6abe074/pytest.log` SHA256 `3b3a7706f9d5d542315ffd4dbdafaa275c79de3201f6a7395b30eaf98315218c` |
| Historical SHAs | `22fbc0e`, `8515728649a280d435b04c23c8700fbd169495f5`, `b7bf014ee03e32cd3510aa3a509eed6a25e750e2` unchanged except new ladder fields |
| Not claimed | Operator **B6 PASS**, **OD**, parent **CLOSED** |

## Remaining gates (closed @ W10)

| Field | Status |
| --- | --- |
| **INDEPENDENT_ACCEPTANCE_BASELINE_SHA** | `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` |
| **OPERATOR_DECISION_SHA** | Recorded @ W10 publication commit |
| **FINAL_PUBLICATION_SHA** | Recorded @ W10 publication / pin commit |

- External **B6 Protocol 03** — **ISSUED — PASS (SCOPED)** ([`ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md)).
- **Operator OD-REPLAYX-08** — **APPROVED** 2026-10-30.
- Parent **ROADMAP-REPLAY-X** — **CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)**.
