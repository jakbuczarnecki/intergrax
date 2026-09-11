# Central Diagnostic Engine — Preventive Intelligence Governance (R6-Q)

**Task:** `DIAGNOSTIC-ENGINE-PREVENTIVE-INTELLIGENCE-ENTERPRISE-HARDENING-R6-Q`

**Status:** Architecture frozen (enterprise governance envelope)

**Invariant:** `PREVENTION_IS_NOT_REMEDIATION` — recommendations never hold execution authority.

---

## 1. Safety boundary

`PreventiveSafetyAssessment` (`intergrax/contracts/preventive/safety.py`):

- `execution_allowed` is **always** `false` — rejected at construction if `true`.
- `requires_human_review`, `risk_level`, `governance_status` surface operator obligations.
- Parallel guard on `PreventiveRecommendationGovernance.execution_allowed`.

Preventive language is advisory: *„Na podstawie dowodów rekomendujemy sprawdzenie X”* — never autonomous remediation.

---

## 2. Lifecycle governance

`PreventiveRecommendationLifecycleState`:

```text
GENERATED → VALIDATED → PRESENTED → ACCEPTED | REJECTED → EVALUATED
```

Forbidden: `GENERATED → EXECUTED` (no `EXECUTED` state exists).

Engine mints recommendations at `VALIDATED`; investigation projection advances to `PRESENTED`; outcome learning records `EVALUATED`.

---

## 3. Evidence qualification

Each `PreventiveRecommendation` carries:

| Field | Role |
| ----- | ---- |
| `evidence_refs` | Auditable links |
| `reasoning_summary` | Human-readable rationale |
| `confidence` | Composed score |
| `known_limitations` | Explicit gaps (e.g. missing provider telemetry) |
| `evidence_quality` | `HIGH` / `MEDIUM` / `LOW` from context quality |

Candidates without evidence are rejected before minting.

---

## 4. Plugin governance

`PreventiveAnalyzerDescriptor`: `id`, `namespace`, `version`, `owner`, `capabilities`, `quality_profile_id`, `resource_budget`.

Registry rejects plugins with descriptor / SPI identity mismatch.

---

## 5. Conflict semantics

`PreventiveRecommendationConflictResolver` emits `CONFLICTING_RECOMMENDATIONS` markers for same-scope, incompatible analyzer outputs.

**Never** merge or pick a winner (no last-writer-wins).

---

## 6. Audit trail

Per recommendation: `PreventiveAuditRecord` (`recommendation_id`, `prediction_signal_id`, `context_snapshot_id`, `analyzer_id`, `analyzer_version`, `evidence_refs`, `confidence`, `created_at`).

Run-level: `PreventiveRecommendationAuditRecord` (analyzer outcomes, degradation).

---

## 7. Tenant model

`PreventiveAnalysisInput` enforces tenant alignment across context, signal, history, and diagnostic refs.

Recommendations and audit records are tenant-scoped; no cross-tenant evidence binding.

---

## 8. Plugin failure containment

Analyzer exceptions → `PLUGIN_UNAVAILABLE` in run audit; pipeline continues with remaining plugins.

---

## 9. Explicit prohibitions

No automatic execution, action executor, self-healing, remediation agents, Problem/Incident stores, second diagnostic authority, LLM decisioning, or hidden actions.

---

## 10. Regression

Predictive R1–R6 paths remain unchanged when prevention is not invoked.
