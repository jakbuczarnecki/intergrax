# Central Diagnostic Engine — Predictive Historical Risk Intelligence Qualification R2

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INCIDENT-INTELLIGENCE-R2-HISTORICAL-RISK-INTELLIGENCE`  
**Branch:** `development`

## Capability matrix

| Capability | Result | Primary proof |
| ---------- | ------ | ------------- |
| One diagnostic authority preserved | **PASS** | `test_prediction_history_not_problem_store`; R1 `test_predictive_authority_boundary` |
| Prediction history separated from incidents | **PASS** | `test_prediction_history_not_problem_store` |
| Outcome tracking | **PASS** | `test_prediction_outcome_confirmation` |
| Analyzer metrics | **PASS** | `test_analyzer_metrics` |
| Tenant isolation | **PASS** | `test_prediction_tenant_isolation` |
| Plugin compatibility | **PASS** | In-memory + DocumentStore persistence ports |
| Auditability | **PASS** | `PredictiveRiskHistoryRecord` + codec |
| Read model integration | **PASS** | `test_prediction_history_read_model` |
| No ML dependency | **PASS** | Rule-based resolver and metrics |
| Future ML ready | **PASS** | Versioned history schema; architecture R2 §7 |

## Deterministic test matrix

```bash
uv run pytest tests/unit/runtime/prediction/history/ -q
```

| Test module | Intent |
| ----------- | ------ |
| `test_prediction_history_not_problem_store.py` | History/resolver never owns Problem authority |
| `test_prediction_outcome_confirmation.py` | Prediction + future evidence → CONFIRMED |
| `test_false_positive_tracking.py` | Wrong prediction does not create incident |
| `test_prediction_tenant_isolation.py` | Tenant A ≠ Tenant B history |
| `test_analyzer_metrics.py` | Quality metrics calculation |
| `test_prediction_history_read_model.py` | `DiagnosticInvestigationView.prediction_history` |

## Definition of Done

| Gate | Result |
| ---- | ------ |
| Historical prediction model exists | **PASS** |
| Prediction outcomes tracked | **PASS** |
| No Prediction Problem authority | **PASS** |
| No second incident store | **PASS** |
| Analyzer quality measurable | **PASS** |
| Tenant isolation proven | **PASS** |
| Read model enriched | **PASS** |
| Future ML integration possible | **PASS** |
