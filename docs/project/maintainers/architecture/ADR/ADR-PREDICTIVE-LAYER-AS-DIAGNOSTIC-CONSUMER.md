# ADR-PREDICTIVE-LAYER-AS-DIAGNOSTIC-CONSUMER

| Field | Value |
| ----- | ----- |
| **Status** | Accepted (Predictive R1) |
| **Date** | 2026-09-11 |
| **Task** | `DIAGNOSTIC-ENGINE-PREDICTIVE-INCIDENT-INTELLIGENCE-R1` |

---

## Problem

Should failure/incident **prediction** be:

**A)** part of the Diagnostic Engine as diagnostic authority, or  
**B)** a separate layer that **consumes** diagnostic evidence?

---

## Decision

**B — Predictive Layer = Consumer of Diagnostic Evidence**

```text
Diagnostic Engine  →  (read evidence / read model)  →  Prediction Engine  →  PredictiveRiskSignal
```

Prediction does **not** write Problems, does **not** reinterpret canonical failure evidence, and does **not** share diagnostic authority with `intergrax.runtime.diagnostics`.

---

## Reasons

### 1. Single authority

Avoids:

```text
Diagnostic Engine
        +
Prediction Engine (as truth)
        +
Application Diagnostics
```

Operators retain **one** diagnostic truth surface; prediction remains advisory.

### 2. Audytowalność

Every risk signal and run must be reconstructable:

| Field | Role |
| ----- | ---- |
| `prediction_run_id` | Run identity (`prun_*`) |
| `analyzer_metadata.analyzer_id` | Plugin identity |
| `analyzer_metadata.analyzer_version` | Plugin version |
| `evidence_refs` | Pointers into evidence / aggregates |
| `generated_at` | Emission time |

Audit envelope: `PredictionAuditRecord` on `PredictionEngineResult`.

### 3. Enterprise extensibility

Analyzer chain without platform churn:

```text
Rule Based Analyzer
        |
        v
Statistical Analyzer
        |
        v
ML Analyzer
        |
        v
External AI Forecast Service
```

All implement `PredictiveAnalyzer`; registry ordering and failure containment unchanged.

---

## Consequences

- `PredictiveInvestigationService` attaches readonly signals to `DiagnosticInvestigationView`.
- Plugin failure → `PLUGIN_UNAVAILABLE`, degraded run, engine continues.
- No Predictive Problem Store; Problems remain `ProblemLifecycleEngine` only.
- ML/LLM may explain or score risk **only** inside versioned analyzers — never as diagnostic authority.

---

## Compliance

Architecture: `DIAGNOSTIC_ENGINE_PREDICTIVE_INCIDENT_INTELLIGENCE_ARCHITECTURE_R1.md`  
Qualification: `DIAGNOSTIC_ENGINE_PREDICTIVE_INCIDENT_INTELLIGENCE_QUALIFICATION_R1.md`
