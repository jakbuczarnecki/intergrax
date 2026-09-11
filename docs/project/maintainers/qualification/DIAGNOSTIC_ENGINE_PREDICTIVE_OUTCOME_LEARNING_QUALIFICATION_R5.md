# Central Diagnostic Engine — Predictive Outcome Learning Qualification R5

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INTELLIGENCE-OUTCOME-LEARNING-R5`  
**Branch:** `development`

## Capability matrix

| Capability | Result | Primary proof |
| ---------- | ------ | ------------- |
| Outcome contracts | **PASS** | `test_true_positive_evaluation` |
| Resolver SPI | **PASS** | `EvidenceBackedPredictiveOutcomeResolver` + engine |
| Evaluation engine | **PASS** | `test_prediction_outcome_audit_complete` |
| Persistence | **PASS** | `InMemoryPredictionOutcomePersistence` via engine |
| Analyzer quality update | **PASS** | `test_true_positive_evaluation` |
| Confidence calibration | **PASS** | `test_confidence_calibration_uses_quality_profile` |
| Auditability | **PASS** | `test_prediction_outcome_audit_complete` |
| Read model | **PASS** | `test_read_model_prediction_outcome_history` |
| CRM showcase | **PASS** | `test_crm_showcase_outcome_learning_phases` |
| Plugin isolation | **PASS** | `test_failed_outcome_resolver_is_contained` |
| Tenant isolation | **PASS** | `test_outcome_evaluation_tenant_isolated` |
| No second authority | **PASS** | `test_prediction_cannot_create_problem` |
| Regression | **PASS** | `test_prediction_without_outcome_unchanged` |

## Deterministic test matrix

```bash
uv run pytest tests/unit/runtime/prediction/test_predictive_outcome_learning_r5.py -q
uv run pytest tests/unit/runtime/prediction/test_predictive_quality_governance_r4.py -q
```

Use a single pytest process (no mass parallel `uv` workers) on operator workstations.

## Enterprise CRM scenario

| Phase | Narrative |
| ----- | --------- |
| 1 | Customer API latency rising — `HIGH_LATENCY_RISK` ~0.78 |
| 2 | Error/retry surge — confidence rises (~0.93) |
| 3 | Incident CRM unavailable (evidence refs) |
| 4 | Evaluation `TRUE_POSITIVE` with `INC-4521` |
| 5 | `latency_trend` profile precision improves |

Fixtures: `crm_agent_incident_prevention_context`, `crm_agent_day2_future_evidence` in `tests/unit/runtime/prediction/conftest.py`.

## Definition of Done

| Gate | Result |
| ---- | ------ |
| Outcome contracts | **PASS** |
| Outcome resolver SPI | **PASS** |
| Evaluation engine | **PASS** |
| Persistence | **PASS** |
| Analyzer quality update | **PASS** |
| Confidence calibration | **PASS** |
| Auditability | **PASS** |
| Read model | **PASS** |
| CRM showcase | **PASS** |
| Plugin isolation | **PASS** |
| Tenant isolation | **PASS** |
| Regression R1–R4 | **PASS** |
| No second authority | **PASS** |
