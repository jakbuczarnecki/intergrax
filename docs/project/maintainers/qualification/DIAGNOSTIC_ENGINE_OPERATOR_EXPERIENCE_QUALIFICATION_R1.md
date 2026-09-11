# Central Diagnostic Engine — Operator Experience qualification (R7 / R1)

**Task:** `DIAGNOSTIC-ENGINE-OPERATOR-EXPERIENCE-R1`

**Tests:** `tests/unit/runtime/diagnostics/test_diagnostic_operator_investigation_r7_qualification.py`

## Matrix

| ID | Scenario | Expected |
| -- | -------- | -------- |
| R7-A1 | Simple execution failure | Failure boundary + proven evidence in investigation |
| R7-A2 | Nested lineage E1→E4, fail E4 | Boundary E4, impact root E1 |
| R7-A3 | Fan-out, fail E3 | Only E3 failed; siblings healthy |
| R7-A4 | Decision correlation on failure | Decision visible; no causal claims |
| R7-A5 | Standard failure read | Explicit unknowns + limitations |
| R7-A6 | Extension SPI harness | Enrichment visible; single Problem |
| R7-A7 | Cross-tenant read | Investigation unavailable |

## Quality gates (automated)

| Gate | Test |
| ---- | ---- |
| Forbidden second-engine symbols | `test_r7_quality_gates_no_second_engine_and_no_heuristic_root_cause` |
| No heuristic root cause field | Same test — `root_cause_status` stays `UNKNOWN` |

## Operator questions (Definition of Done)

After R7, `get_investigation` supports bounded answers for:

1. What happened? — `assistant_payload.what_happened` + evidence summary  
2. Where did it fail? — `failure_boundary` / `failure_investigation.failure_boundaries`  
3. What evidence proves it? — `evidence_summary` (PROVEN)  
4. What is impact? — `impact_graph` + `affected_execution_ids`  
5. What decisions are related? — `decision_context` + `related_decision_ids`  
6. What remains unknown? — `explicit_unknowns` + limitations  
7. What next? — `recommendations` (non-automatic)

## Status

**ACCEPTED** when gate tests pass at operator revision.
