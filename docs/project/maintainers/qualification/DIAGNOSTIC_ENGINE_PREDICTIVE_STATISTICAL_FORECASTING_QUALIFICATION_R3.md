# Predictive Statistical Forecasting — Qualification (R3)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INCIDENT-INTELLIGENCE-R3-STATISTICAL-FORECASTING`

**Suite:** `tests/unit/runtime/prediction/test_predictive_statistical_forecasting_r3_qualification.py`

**Regression:** `tests/unit/runtime/prediction/` (R1/R2)

---

## Qualification matrix

| ID | Scenario | Expected | Test |
| -- | -------- | -------- | ---- |
| R3-A1 | Latency degradation (120→180→260 ms) | `LATENCY_DEGRADATION` signal | `test_r3_a1_latency_degradation` |
| R3-A2 | Failure acceleration (5→20→60) | `FAILURE_ACCELERATION_RISK` | `test_r3_a2_failure_acceleration` |
| R3-A3 | Retry storm (4.8 vs 1.2 baseline) | `RETRY_STORM_RISK` | `test_r3_a3_retry_storm` |
| R3-A4 | Flat latency — no trend | No signals | `test_r3_a4_false_positive_control_no_trend` |
| R3-A5 | Lower historical precision | Lower confidence | `test_r3_a5_historical_quality_influence` |
| R3-A6 | One analyzer throws | Others succeed; `PLUGIN_UNAVAILABLE` | `test_r3_a6_plugin_isolation` |
| R3-A7 | Cross-tenant signal | `ValueError` fail closed | `test_r3_a7_tenant_isolation` |
| R3-A8 | 1000 latency points | Wall time &lt; 500ms | `test_r3_a8_bounded_execution` |

---

## Authority checks (manual / existing suites)

| Check | Evidence |
| ----- | -------- |
| Single diagnostic authority | `test_predictive_authority_boundary` — no Problem minting in `runtime/prediction/` |
| No forecast Problem store | `test_prediction_history_not_problem_store` (R2) |
| Consumer model | Architecture R3 §1 — CDE owns Problems |

---

## Definition of Done

| Criterion | Status |
| --------- | ------ |
| Predictive layer consumer of CDE | Architecture + boundary tests |
| `DIAGNOSTIC_AUTHORITY_COUNT = 1` | No second engine / Problem store |
| Pluginable analyzers | `PredictiveForecastAnalyzerRegistry` |
| Feature extraction separated | `PredictiveFeatureExtractor` SPI |
| Evidence-based confidence | `compose_forecast_confidence` |
| Historical quality calibration | `HistoricalRiskIntelligence` |
| Read model readonly `forecast_risk_signals` | `DiagnosticInvestigationView` |
| R3-A1–A8 | Qualification module |
| R1/R2 regression | Full `tests/unit/runtime/prediction/` |
| Architecture doc | `DIAGNOSTIC_ENGINE_PREDICTIVE_STATISTICAL_FORECASTING_ARCHITECTURE_R3.md` |

---

## Known gaps (accepted for R3)

- Resource exhaustion analyzer not individually gated in A1–A8 (covered by registry default + integration).
- Showcase investigation path may return empty forecasts until execution metrics are wired into `build_predictive_context_for_investigation`.
