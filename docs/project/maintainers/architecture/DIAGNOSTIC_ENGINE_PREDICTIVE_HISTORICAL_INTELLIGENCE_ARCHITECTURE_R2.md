# Central Diagnostic Engine — Predictive Historical Risk Intelligence (R2)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INCIDENT-INTELLIGENCE-R2-HISTORICAL-RISK-INTELLIGENCE`

**Status:** Architecture frozen (Predictive R2)

**Invariant:** `PREDICTION_IS_NOT_DIAGNOSIS`

---

## 1. Executive model

Predictive R1 introduced bounded risk signals. R2 adds **auditable prediction memory** and **outcome feedback** without a second diagnostic or incident authority.

```text
Evidence Producers
        |
        v
Central Diagnostic Engine  ← incident truth (Problems)
        |
        +---------------------+
        |                     |
        v                     v
Problem Lifecycle       Predictive Intelligence
        |                     |
        v                     v
Problems              PredictiveRiskSignal
                              |
                              v
                      Prediction History (R2)
                              |
                              v
                      Outcome + Analyzer metrics
                              |
                              v
                      DiagnosticInvestigationView (readonly)
```

---

## 2. Frozen Architecture Decision — Prediction History Authority

**Mandatory contract:**

> Predictive History stores prediction lifecycle.  
> It does **not** store incident truth.  
> Incident truth remains owned by Diagnostic Engine.

```text
PredictionHistory ≠ IncidentStore
PredictionHistory ≠ ProblemStore
PredictionHistory ≠ RootCauseStore
```

---

## 3. Authority Matrix

| Component | Owns |
| --------- | ---- |
| Runtime Evidence | execution facts |
| Diagnostic Engine | diagnosis |
| ProblemLifecycleEngine | incidents/problems |
| Prediction Engine | future risk estimation |
| Prediction History | prediction outcomes |
| Analyzer Registry | execution ordering |
| ML Models | probability estimation only |

---

## 4. Outcome lifecycle (frozen)

```text
CREATED
   |
   v
OBSERVED
   |
   +------------+
   |            |
   v            v
CONFIRMED   FALSE_POSITIVE
   |            |
   +-----+------+
         v
     EVALUATED
```

Terminal outcome values (before or at evaluation): `CONFIRMED`, `FALSE_POSITIVE`, `EXPIRED`, `UNKNOWN`, `INSUFFICIENT_EVIDENCE`.

`PredictionOutcomeResolver` classifies outcomes only — it **never** creates Problems.

---

## 5. Implementation map (R2)

| Concern | Module |
| ------- | ------ |
| History contract | `intergrax/contracts/predictive_history.py` |
| Analyzer governance | `intergrax/contracts/predictive_analyzer_descriptor.py` |
| Quality metrics | `intergrax/contracts/predictive_analyzer_quality.py` |
| Persistence port | `intergrax/runtime/prediction/history/predictive_history_persistence.py` |
| In-memory store | `intergrax/runtime/prediction/history/in_memory_predictive_history_persistence.py` |
| Document store | `intergrax/runtime/prediction/history/document_store_predictive_history_persistence.py` |
| Outcome resolver | `intergrax/runtime/prediction/outcome/prediction_outcome_resolver.py` |
| History service | `intergrax/runtime/prediction/history/predictive_history_service.py` |
| Investigation projection | `intergrax/runtime/prediction/history/predictive_history_investigation_projection.py` |
| Read model field | `DiagnosticInvestigationView.prediction_history` |

---

## 6. Enterprise showcase — Customer Operations Platform

```text
Customer Request → Supervisor Agent → CRM Agent / Billing Agent → External CRM API
```

| Day | Event |
| --- | ----- |
| Day 1 | Prediction: CRM degradation risk ~82%; evidence latency +120%, retry +40%, timeout +25% |
| Day 2 | Incident: CRM API unavailable; Diagnostic Engine creates Problem; history outcome **CONFIRMED** |

Fixture: `crm_agent_showcase_context` and `crm_agent_day2_future_evidence` in `tests/unit/runtime/prediction/conftest.py`.

Analyzer governance example:

```json
{
  "id": "latency-trend",
  "version": "1.0",
  "scope": "execution",
  "risk": "performance_degradation"
}
```

---

## 7. Future ML readiness

History rows are structured, tenant-scoped, and analyzer-versioned — suitable for offline training without coupling to Problem persistence.
