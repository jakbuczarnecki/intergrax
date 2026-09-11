# Central Diagnostic Engine — Predictive Statistical Forecasting (R3)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INCIDENT-INTELLIGENCE-R3-STATISTICAL-FORECASTING`

**Status:** Architecture frozen (Predictive R3)

**Invariant:** `PREDICTION_IS_NOT_DIAGNOSIS` · `DIAGNOSTIC_AUTHORITY_COUNT = 1`

---

## 1. Executive model

R3 adds **deterministic statistical forecasting** on top of R1 bounded risk signals and R2 historical quality. The predictive layer remains a **consumer** of diagnostic evidence — never a second diagnostic engine.

```text
Evidence
   |
   v
Central Diagnostic Engine  ← Problem / diagnostic truth
   |
   +------------------+------------------+
   |                                     |
   v                                     v
Diagnostic Analysis              Predictive Intelligence
   |                                     |
   v                                     v
Problems                         Risk Forecasts (R3)
```

Forbidden (R3): `PredictiveEngine` as diagnostic authority, `ForecastProblemStore`, `PredictionIncidentLifecycle`, `PredictiveRootCauseEngine`, `ForecastTruthDatabase`.

---

## 2. Authority matrix

| Component | Owns |
| --------- | ---- |
| `intergrax.runtime.diagnostics` | Problem truth, diagnostic truth, failure interpretation |
| Prediction Engine (R1) | Orchestration, audit envelope |
| Statistical Forecast Engine (R3) | Trend / anomaly / bounded forward risk |
| Prediction History (R2) | Outcomes, analyzer precision — **not** incidents |
| ML (R4+) | Probability models — **out of R3 scope** |

---

## 3. Data sources (readonly)

Forecast analyzers read **only** via `PredictiveContext`:

- Diagnostic read facts (via context builder)
- `HistoricalRiskIntelligence` (R2-derived precision)
- `ExecutionPatternSnapshot`
- `PerformanceMetricPoint` / `failure_history`
- No application DB, raw logs, vendor telemetry, or direct execution storage

---

## 4. SPI

### 4.1 `PredictiveFeatureExtractor`

Separates observation from analysis.

| Contract | Module |
| -------- | ------ |
| `PredictiveFeatureExtractor` | `intergrax/contracts/predictive_feature_extractor.py` |
| `PredictiveFeatureSet` | `intergrax/contracts/predictive_feature_set.py` |
| Default implementation | `intergrax/runtime/prediction/forecasting/default_feature_extractor.py` |

`PredictiveFeatureSet` is immutable and **must not** contain `ProblemId`, root cause, recommendations, or actions.

### 4.2 `StatisticalForecastAnalyzer`

| Contract | Module |
| -------- | ------ |
| `StatisticalForecastAnalyzer` | `intergrax/contracts/statistical_forecast_analyzer.py` |
| `ForecastAnalyzerDescriptor` | `intergrax/contracts/forecast_analyzer_descriptor.py` |
| Registry | `intergrax/runtime/prediction/forecasting/forecast_registry.py` |
| Orchestrator | `intergrax/runtime/prediction/forecasting/statistical_forecast_engine.py` |

Ordering: `priority` → `namespace` → `analyzer_id`.

Plugin failure → `PLUGIN_UNAVAILABLE` on that analyzer only; other analyzers continue.

---

## 5. Built-in analyzers (R3)

| Analyzer | `risk_type` | Purpose |
| -------- | ----------- | ------- |
| Latency Degradation Forecast | `LATENCY_DEGRADATION` | Slow latency growth |
| Failure Acceleration Forecast | `FAILURE_ACCELERATION_RISK` | Accelerating failure counts |
| Retry Storm Forecast | `RETRY_STORM_RISK` | Retry amplification vs baseline |
| Resource Exhaustion Forecast | `RESOURCE_EXHAUSTION_RISK` | Rising utilization toward limits |

Modules: `intergrax/runtime/prediction/forecasting/analyzers/`.

---

## 6. Confidence model

```text
confidence = evidence_strength × analyzer_historical_precision × data_completeness
```

Implementation: `intergrax/runtime/prediction/forecasting/confidence_model.py` + R2 `HistoricalRiskIntelligence.precision_for()`.

No random or static confidence.

---

## 7. Prediction lifecycle

Unchanged from R2:

```text
Prediction → Outcome Observation → Historical Intelligence
```

R3 does **not** introduce `PredictionProblem` or any Problem minting path.

---

## 8. Operator read model

`DiagnosticInvestigationView` extended (readonly):

| Field | Content |
| ----- | ------- |
| `forecast_risk_signals` | R3 statistical forecasts (`RelatedPredictiveRiskSignalView`) |
| `related_risk_signals` | R1 rule-based signals (unchanged) |
| `prediction_history` | R2 history (unchanged) |

Projection: `project_forecast_risk_signals` in `intergrax/runtime/prediction/predictive_investigation_projection.py`.

---

## 9. Enterprise constraints

Per analyzer `ForecastResourceBudget`:

- `max_execution_time_ms`
- `max_input_points` (default extractor caps at 1000 points per metric)
- `max_memory_kb`

Engine-level time budget: `DEFAULT_FORECAST_TIME_BUDGET_MS` (500ms). Analyzers must not mutate context, persist Problems, or emit diagnostic failure truth.

---

## 10. Security and tenant isolation

- All signals carry `tenant_id` from context; cross-tenant emission fails closed.
- History and forecast stores remain tenant-scoped (R2).

---

## 11. Known limitations

- Rule-based / statistical only — no ML, embeddings, or LLM prediction in R3.
- Feature extraction depends on metric names (`latency_ms`, `retry_per_execution`, `memory_utilization_pct`, `failure_count`).
- Investigation context builder may supply sparse execution metrics until enriched upstream.
- Relative latency growth requires ≥2 samples; flat series produce no signal (false-positive control).

---

## 12. Implementation map

| Concern | Module |
| ------- | ------ |
| Historical intelligence input | `intergrax/contracts/predictive_historical_intelligence.py` |
| Context field | `PredictiveContext.historical_risk_intelligence` |
| R1 bridge adapter | `StatisticalForecastPredictiveAnalyzer` |
| Investigation attachment | `PredictiveInvestigationService.forecast_risk_signals` |

Qualification: `tests/unit/runtime/prediction/test_predictive_statistical_forecasting_r3_qualification.py`.
