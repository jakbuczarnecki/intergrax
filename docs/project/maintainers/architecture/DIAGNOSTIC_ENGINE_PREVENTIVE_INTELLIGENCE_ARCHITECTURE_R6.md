# Central Diagnostic Engine — Preventive Intelligence (R6)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-PREVENTIVE-INTELLIGENCE-R6`

**Status:** Architecture frozen (Preventive R6 recommendation layer)

**Invariant:** `PREVENTION_IS_NOT_REMEDIATION`

---

## 1. Executive model

Preventive R6 converts governed predictive risk into auditable operator recommendations:

```text
PredictiveRiskSignal (R1–R5)
        |
        v
PreventiveAnalysisInput (context + history + diagnostic refs)
        |
        v
PreventiveAnalyzer SPI (plugins)
        |
        v
PreventiveIntelligenceEngine
   validation → confidence → governance → audit
        |
        v
PreventiveRecommendation
        |
        v
Human / Governance decision (execution_allowed=false)
        |
        v
RecommendationOutcomeEvaluation (optional learning)
```

The layer never mutates production state, Problems, or diagnostic truth.

---

## 2. Recommendation model

`PreventiveRecommendation` fields: `recommendation_id`, `prediction_run_id`, `risk_signal_id`, `category`, `description`, `expected_impact`, `confidence`, `evidence_refs`, `governance`, `created_at`.

Categories use `PreventiveRecommendationCategory` namespace (base strings + dotted plugin ids).

---

## 3. SPI — `PreventiveAnalyzer`

**Input:** `PreventiveAnalysisInput` (`PredictiveContext`, `PredictiveRiskSignal`, `HistoricalOutcome`, `DiagnosticEvidenceContext`).

**Output:** `PreventiveRecommendationCandidate` (evidence-bound).

**Forbidden:** persist recommendations, create Problems, execute actions, hidden remediation.

---

## 4. Engine

`PreventiveIntelligenceEngine` (`intergrax/runtime/prevention/`):

- deterministic analyzer ordering (registry)
- per-analyzer timeout / failure containment
- evidence validation (reject candidates without refs)
- `PreventiveConfidenceEvaluator` composition
- append-only `PreventiveRecommendationAuditRecord`
- tenant isolation on input bundle

---

## 5. Confidence model

```text
confidence =
    risk_confidence
    × analyzer_quality (R4/R5 profile)
    × historical_success (recommendation outcomes)
    × context_completeness (coverage/reliability blend)
```

Example: `0.9 × 0.8 × 0.7 × 0.9 ≈ 0.45`.

LLM output is never authority.

---

## 6. Evidence requirements

Each recommendation carries `RecommendationEvidenceReference` (`source_type`, `source_id`, `relation`).

Recommendations without evidence are rejected at validation.

---

## 7. Governance boundaries

`PreventiveRecommendationGovernance`:

- `risk_level` from signal severity
- `required_approval` for high severity / sensitive categories / HIGH priority
- `execution_allowed` **always false**

---

## 8. Outcome learning

`RecommendationOutcomeEvaluation` records operator accept/reject and effectiveness (`TRUE_PREVENTION`, etc.).

`PreventiveOutcomeEngine` updates historical success rates and may increment analyzer quality (R5 store) on evidenced prevention.

---

## 9. Read model

`DiagnosticInvestigationView.preventive_recommendations`: readonly `RelatedPreventiveRecommendationView` tuples projected from engine output — not actions.

---

## 10. CRM enterprise showcase

| Phase | Narrative |
| ----- | --------- |
| T-60m | `HIGH_LATENCY_RISK` ~0.82 → investigate connector / timeout config |
| T-20m | `connector_retries:+500%` evidence → HIGH priority provider validation |
| Outcome | Incident avoided → `TRUE_PREVENTION` recorded |

Analyzer: `CrmLatencyPreventiveAnalyzer`.

---

## 11. Security model

- Tenant-scoped inputs and outputs
- No customer payload copies on contracts
- Plugins isolated; failures contained
- Audit trail on every engine run

---

## 12. Limitations

- Not remediation, deployment, or self-healing
- No Problem Store, Incident Store, Evidence Store, or second Diagnostic Engine
- No autonomous agent decisions
- Automation is explicitly future / out-of-scope

---

## 13. Operator workstation constraint (agents)

Run **one** `uv run pytest …` process per verification step. Do **not** spawn dozens of parallel `uv`/`python` workers on a laptop — it causes memory pressure and OS freezes.
