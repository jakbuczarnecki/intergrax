# Central Diagnostic Engine — Predictive Incident Intelligence (R1)

**Task:** `DIAGNOSTIC-ENGINE-PREDICTIVE-INCIDENT-INTELLIGENCE-R1-DOCUMENTATION-AND-ENTERPRISE-PROOF`

**Status:** Architecture frozen (Predictive R1)

**Invariant:** `PREDICTION_IS_NOT_DIAGNOSIS`

---

## 1. Executive model

```text
Execution Runtime
        |
        v
Runtime Evidence
        |
        v
Central Diagnostic Engine
        |
        +----------------+
        |                |
        v                v
Problem Lifecycle    Predictive Intelligence
        |                |
        v                v
Problems             Risk Signals
```

Predictive Intelligence **consumes** diagnostic read evidence and execution-derived context. It **never** mints Problems, never replaces failure evidence, and never becomes a second diagnostic engine.

**Forbidden topologies:**

- Second diagnostic engine (application-local or scenario-local)
- Predictive Problem Store
- `Execution → Prediction → Problem` shortcut
- ML or LLM as source of diagnostic truth
- Automatic Problem creation from prediction

---

## 2. Frozen Architecture Decision — Predictive Intelligence Authority Model

**Mandatory contract:**

| Rule | Meaning |
| ---- | ------- |
| Predictive Intelligence is **advisory** | Operators see risk estimates, not proven failures |
| May estimate **future operational risk** | Bounded `PredictiveRiskSignal` with confidence and window |
| **Cannot** establish diagnostic truth | Only Central Diagnostic Engine interprets canonical evidence |
| **Cannot** create Problems | No writes to `ProblemLifecycleEngine` or Problem persistence |
| **Cannot** replace evidence | Signals carry `evidence_refs` only; no raw dumps on contract |

```text
Prediction IS NOT Diagnosis.
```

---

## 3. Authority Matrix

| Component | Responsibility | Authority |
| --------- | -------------- | --------- |
| ExecutionRuntime | execution facts | **YES** |
| RuntimeEvent | failure evidence | **YES** |
| CausalEvidence | causal relations | **YES** |
| Diagnostic Engine | diagnosis | **YES** |
| ProblemLifecycleEngine | problem lifecycle | **YES** |
| Prediction Engine | risk estimation | **NO** diagnosis authority |
| Analyzer plugins | risk models | **NO** |
| LLM | explanation | **NO** |

---

## 4. Canonical Data Flow (frozen)

**Allowed:**

```text
Execution
   |
   v
Evidence
   |
   v
Diagnostic Read Model
   |
   v
Predictive Context Builder
   |
   v
Predictive Analyzer Registry
   |
   v
PredictiveRiskSignal
   |
   v
Diagnostic Investigation View
```

**Forbidden:**

```text
Execution
   |
   v
Prediction
   |
   v
Problem
```

Implementation map (R1):

| Stage | Module |
| ----- | ------ |
| Context | `intergrax.runtime.prediction.predictive_context_builder` |
| Registry | `intergrax.runtime.prediction.predictive_registry` |
| Engine | `intergrax.runtime.prediction.prediction_engine` |
| Investigation attach | `intergrax.runtime.prediction.predictive_investigation_service` |
| Operator read | `DiagnosticInvestigationView.related_risk_signals` |

Contracts: `intergrax/contracts/predictive_*.py`

---

## 5. Prediction Lifecycle Contract

1. **Context Collection** — Readonly `PredictiveContext` from diagnostic read facts, execution patterns, performance samples, historical Problem **refs** (correlation only).
2. **Context Normalization** — Tenant-bound snapshot (`input_snapshot_id`, `as_of`); no cross-tenant mixing.
3. **Analyzer Selection** — `PredictiveAnalyzerRegistry` orders by `priority` (higher first), then namespace, then `analyzer_id`.
4. **Analyzer Execution** — Bounded time budget per run; pure `analyze(context)` — no persistence, no Problem API.
5. **Signal Validation** — Tenant match, non-empty `evidence_refs`, confidence in `[0,1]`, stamped `prediction_run_id` and `analyzer_metadata`.
6. **Audit Persistence** — `PredictionAuditRecord` per run (`prediction_id`, analyzer outcomes, degraded flag).
7. **Read Model Exposure** — `RelatedPredictiveRiskSignalView` on `DiagnosticInvestigationView` (readonly enrichment).

---

## 6. Plugin Architecture (SPI hardening)

`PredictiveAnalyzer` is a **Service Provider Interface** under `intergrax.contracts.predictive_analyzer`.

Each plugin must be:

| Property | Requirement |
| -------- | ----------- |
| Isolated | Failure contained; no shared mutable diagnostic state |
| Versioned | `analyzer_metadata.analyzer_version` + `model_version` on signals |
| Deterministic | Same context → same signal set (R1 rule analyzers) |
| Time-bounded | Engine enforces `time_budget_ms` |
| Audited | Outcomes recorded on `PredictionAuditRecord` |

**Analyzer failure contract:**

```text
DO NOT BREAK PREDICTION ENGINE
```

Result outcome: **`PLUGIN_UNAVAILABLE`** — run continues, `audit.degraded = true`.

**R1 built-in analyzers (priority):**

| Analyzer | priority |
| -------- | -------- |
| `LatencyTrendAnalyzer` | 100 |
| `FailurePatternAnalyzer` | 90 |
| `FutureMLAnalyzer` (roadmap) | 50 |

---

## 7. Enterprise Showcase — Autonomous Customer Operations Platform

**Scenario:** multi-agent customer operations with external CRM dependency.

```text
Customer Request
       |
       v
Supervisor Agent
       |
       +----------------+
       |                |
       v                v
CRM Agent        Pricing Agent
       |
       v
External CRM API
```

**Historical window (7 days) — evidence-derived inputs to `PredictiveContext`:**

| Signal | Trend |
| ------ | ----- |
| CRM API latency | +180% |
| Timeout rate | +45% |
| Execution failures | +32% |
| Retries | +70% |
| Prior incident ref | INC-4521 |

**Prediction output (illustrative):**

| Field | Value |
| ----- | ----- |
| Summary | Potential CRM Agent degradation |
| Probability (confidence) | ~86% (rule blend from latency + failure pattern) |
| Prediction window | 30 minutes (`FailurePatternAnalyzer`) |
| Confidence label | SUPPORTED (evidence-backed, not proven failure) |
| Evidence refs | latency trend, failure pattern, historical similarity |

**Diagnostic investigation (operator):**

```text
Current Problems
      +
Historical Incidents (refs)
      +
Predictive Risk Signals  →  DiagnosticInvestigationView.related_risk_signals
```

Proof fixture: `tests/unit/runtime/prediction/conftest.py` (`crm_agent_showcase_context`).

---

## 8. Future Roadmap Contract

| Version | Capability |
| ------- | ---------- |
| Predictive R1 | Rule-based risk signals |
| Predictive R2 | Persistent risk history |
| Predictive R3 | Statistical models |
| Predictive R4 | ML forecasting |
| Predictive R5 | Preventive recommendations |
| Predictive R6 | Governance-controlled remediation |

R4+ ML analyzers remain **plugins** behind the same SPI; platform invariants in §2–§4 unchanged.

---

## 9. Related artifacts

| Artifact | Path |
| -------- | ---- |
| ADR | `docs/project/maintainers/architecture/ADR/ADR-PREDICTIVE-LAYER-AS-DIAGNOSTIC-CONSUMER.md` |
| Qualification | `docs/project/maintainers/qualification/DIAGNOSTIC_ENGINE_PREDICTIVE_INCIDENT_INTELLIGENCE_QUALIFICATION_R1.md` |
| Single authority baseline | `DIAGNOSTIC_ENGINE_SINGLE_AUTHORITY_ARCHITECTURE_R1.md` |
