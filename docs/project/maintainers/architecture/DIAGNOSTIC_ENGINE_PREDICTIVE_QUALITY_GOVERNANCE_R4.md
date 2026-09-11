# Central Diagnostic Engine — Predictive Quality Governance (R4)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-CONTEXT-INTELLIGENCE-QUALITY-GOVERNANCE-R4`

**Status:** Architecture frozen (Predictive R4 governance)

**Invariant:** `PREDICTION_IS_NOT_DIAGNOSIS`

---

## 1. Executive model

Predictive R4 context intelligence supplies bounded inputs to analyzers. R4 governance adds **auditable quality**, **provenance**, **immutable snapshots**, and **confidence composition** without a second diagnostic authority.

```text
PredictiveRiskSignal
        |
        v
Prediction Governance Layer
        |
   +----+----+----+
   |    |    |    |
   v    v    v    v
Context  Analyzer  Evidence  Audit
Quality  Quality   Quality   Chain
        |
        v
Historical Intelligence Feedback (R2 outcomes → analyzer profiles)
```

---

## 2. Quality model

`PredictiveQualityAssessment` decomposes governed confidence:

| Dimension | Meaning |
| --------- | ------- |
| `context_quality` | Reliability from completeness, coverage, freshness |
| `analyzer_quality` | Historical precision from tenant-scoped profiles |
| `evidence_quality` | Strength of cited evidence refs |
| `confidence_quality` | Product of governed factors |
| `completeness` | `PredictiveContextCompleteness` enum |

**Confidence governance (frozen):**

```text
governed_confidence =
    raw_evidence_confidence
    × context_reliability
    × analyzer_historical_precision
    × evidence_completeness
```

Raw `0.95` without factors is forbidden on audit records.

---

## 3. Context provenance

Each provider fragment may attach `PredictiveContextProvenance`:

```text
source, version, generated_at, tenant_scope
```

Aggregated provenance is stored on `PredictiveContext.metadata.provenance` so operators can answer: *where did this prediction context come from?*

---

## 4. Context snapshot

`PredictiveContextSnapshot` binds:

```text
snapshot_id + captured_at + immutable PredictiveContext
```

`PredictionAuditRecord.context_snapshot_id` links a prediction run to the frozen inputs used at generation time.

---

## 5. Audit chain

`PredictionAuditRecord` (contracts) includes:

- `prediction_run_id`
- `context_snapshot_id`
- `analyzer_ids` / `analyzer_versions`
- `provider_versions`
- `quality_assessment`
- `signal_ids`, `analyzer_outcomes`, `degraded`

Reconstructs: *why did we predict failure 24h ago?*

---

## 6. Analyzer lifecycle

1. Analyzer registered with governance metadata (`PredictivePluginRegistration`).
2. Emits bounded `PredictiveRiskSignal` (evidence refs only).
3. Governance adjusts confidence using `PredictiveAnalyzerQualityProfile`.
4. `PredictionOutcomeEvaluation` updates profiles (TRUE_POSITIVE / FALSE_POSITIVE).

---

## 7. Plugin governance

Registry ordering remains deterministic:

```text
priority (desc) → namespace → plugin_id
```

Each registration carries: `id`, `namespace`, `version`, `owner`, `capabilities`, `quality_profile`, `resource_budget`.

No hidden plugins; no random execution order.

---

## 8. Investigation read enrichment

`RelatedPredictiveRiskSignalView` adds:

- `prediction_quality`
- `context_quality`
- `prediction_explanation`

Attached to `DiagnosticInvestigationView.forecast_risk_signals` via projection — readonly only.

---

## 9. Limitations

- Governance does not mint Problems or incident truth.
- No Predictive Problem Store or Predictive Lifecycle engine.
- No LLM authority; no heuristic root cause.
- Analyzer profiles are tenant-scoped; cold-start uses conservative default precision.
- Snapshots capture context at run time; they do not replay live diagnostic writes.

---

## 10. Authority matrix (unchanged)

| Component | Owns |
| --------- | ---- |
| Diagnostic Engine | incident truth |
| Prediction Engine | forward risk estimation |
| Governance layer | quality + audit envelope |
| Prediction History (R2) | prediction outcomes |
