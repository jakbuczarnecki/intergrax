# DIAGNOSTIC-ENGINE-DECISION-EXECUTION-LINEAGE — Qualification R1

**Task:** `DIAGNOSTIC-ENGINE-DECISION-EXECUTION-DIAGNOSTIC-LINEAGE-R1`

## Matrix

| ID | Scenario | Expected | Test |
| -- | -------- | -------- | ---- |
| R4-A1 | Decision → successful execution | Correlation stored; no Problem | `test_r4_a1_*` |
| R4-A2 | Decision → failed execution | Failure at execution boundary; `DecisionContext` attached; decision ≠ cause | `test_r4_a2_*` |
| R4-A3 | Multiple decisions, one failure | Multiple related decisions; no causal inference | `test_r4_a3_*` |
| R4-A4 | Retry | Same `decision_id`; distinct `decision_attempt_id` records | `test_r4_a4_*` |
| R4-A5 | Cross tenant | No cross-tenant correlation reads | `test_r4_a5_*` |
| R4-A6 | Missing decision evidence | Execution diagnostic OK; decision context unavailable | `test_r4_a6_*` |
| R4-A7 | Decision enrichment outage | Execution diagnostic OK; context `DEGRADED` | `test_r4_a7_*` |

Harness: `testing_support/runtime/decision_execution_lineage_r4_harness.py` (extends R2 execution failure closure harness).

Contract tests: `tests/unit/contracts/test_decision_execution_correlation.py`.

Architecture gate symbols: `test_r4_quality_gates_forbidden_symbols_and_single_engine`.

## Verdict

PASS when `uv run pytest tests/unit/contracts/test_decision_execution_correlation.py tests/unit/runtime/diagnostics/test_decision_execution_lineage_r4_qualification.py` succeeds on `development`.
