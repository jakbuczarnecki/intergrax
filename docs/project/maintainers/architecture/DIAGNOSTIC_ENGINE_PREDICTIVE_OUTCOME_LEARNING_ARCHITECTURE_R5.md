# Central Diagnostic Engine — Predictive Outcome Learning (R5)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INTELLIGENCE-OUTCOME-LEARNING-R5`

**Status:** Architecture frozen (Predictive R5 outcome loop)

**Invariant:** `PREDICTION_IS_NOT_DIAGNOSIS`

---

## 1. Executive model

Predictive R5 closes the feedback loop without a second diagnostic authority:

```text
Prediction (R1–R4)
        |
        v
Incident / Outcome Facts (readonly evidence)
        |
        v
PredictiveOutcomeResolver (SPI)
        |
        v
PredictionOutcomeEngine
        |
   +----+----+
   |         |
   v         v
Persistence  Analyzer quality + calibration
        |
        v
Future governed confidence (R4 composition)
```

Predictions never mutate Problems, diagnostic truth, or production configuration.

---

## 2. Outcome lifecycle

| Stage | Artifact | Authority |
| ----- | -------- | --------- |
| Predict | `PredictiveRiskSignal` | Predictive layer |
| Observe facts | `PredictiveOutcomeResolverContext` | Evidence producers / history reads |
| Evaluate | `PredictionOutcomeEvaluation` | Outcome engine + resolver SPI |
| Audit | `PredictionOutcomeAuditRecord` | Append-only audit chain |
| Learn | `PredictiveAnalyzerQualityProfile` | Tenant-scoped quality store |
| Display | `RelatedPredictionOutcomeHistoryView` | Investigation read model (readonly) |

Outcome types (`PredictionOutcomeType`) are extensible; base values include `TRUE_POSITIVE`, `FALSE_POSITIVE`, `INCIDENT_OCCURRED`, `NO_INCIDENT`, `UNKNOWN`. Labels must cite `evidence_refs`.

---

## 3. Resolver SPI

`PredictiveOutcomeResolver.evaluate(prediction, context) -> PredictionOutcomeEvaluation`

**May:** analyze evidence, map outcomes, attach evaluation confidence.

**Must not:** create Problems, write diagnostic stores, persist privately.

Default platform resolver: `EvidenceBackedPredictiveOutcomeResolver` (wraps R2 history logic).

Engine properties: deterministic resolver ordering by `resolver_id`, per-resolver timeout, failure containment, tenant match enforcement.

---

## 4. Persistence model

`PredictionOutcomePersistence` stores evaluations only:

- `prediction_run_id`, `prediction_signal_id`, tenant, analyzer
- outcome type, evaluation status, evidence refs, metadata

No `PredictiveProblemStore`, no incident database, no parallel diagnostic engine.

---

## 5. Calibration model

`PredictiveConfidenceCalibrator` composes:

```text
final_confidence =
    raw_confidence
    × context_reliability
    × analyzer confidence_calibration (derived from outcome history)
    × evidence_completeness
```

Example: raw `0.95` with historical calibration `~0.65` yields governed confidence near `0.65` when context factors are neutral.

Profile fields: precision, recall, false positive rate, false negative rate, `confidence_calibration`.

---

## 6. Analyzer quality update

`apply_outcome_evaluation` increments tenant-scoped counters from evidence-backed evaluations only when status is `EVALUATED`. Unknown outcomes preserve uncertainty (no forced TP/FP).

---

## 7. Governance boundaries

Predictive layer is **not** incident manager, problem manager, root cause engine, or autonomous decision engine.

Forbidden: LLM as outcome truth, self-learning without audit, application-local prediction authority, automatic production changes.

---

## 8. Security and tenant isolation

Evaluations require matching `tenant_id` on prediction and context. Persistence queries are tenant-scoped.

---

## 9. Limitations

- Resolvers depend on supplied evidence; absent facts → `UNKNOWN` / `INSUFFICIENT_EVIDENCE`.
- In-memory persistence is qualification/default; production adapters must preserve immutability semantics.
- Calibration requires sufficient labeled outcomes; cold start uses conservative defaults.

---

## 10. Read model

`DiagnosticInvestigationView.prediction_outcome_history` exposes readonly outcome rows linked to signals and evidence refs — not incident authority.
