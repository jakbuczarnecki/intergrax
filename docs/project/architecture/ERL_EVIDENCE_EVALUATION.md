# ERL evidence evaluation

> **Purpose:** Document the platform evidence quality gate between reconciliation materialization and resolution planning.

## Problem framing

Reconciliation produces `ExternalEffectEvidence`, but resolution planning must not run on incomplete, inconsistent, or untrusted bundles. The platform answers a generic question:

**Is the collected evidence sufficient for reliability processing to continue toward resolution?**

This is not business acceptance (for example payment approval). It is a domain-neutral readiness check.

## Lifecycle position

```text
UNKNOWN external effect
        ↓
Reconciliation orchestration + probe execution
        ↓
ExternalEffectEvidence (materialization)
        ↓
Evidence evaluation  ← this capability
        ↓
Resolution planning / execution
```

Reliability case lifecycle aligns with `EVIDENCE_AVAILABLE` → `RESOLUTION_PENDING`; evidence evaluation is the platform step that justifies that transition without introducing a parallel lifecycle.

## Ownership

| Concern | Owner |
| --- | --- |
| Evaluation context, outcomes, pure evaluation rules | `intergrax/contracts/enterprise_reliability/evidence_evaluation.py` |
| Orchestration after reconciliation | `intergrax/runtime/enterprise_reliability/evidence_evaluation.py` |
| Resolution planning integration | `intergrax/runtime/enterprise_reliability/resolution_orchestration.py` |
| Optional future strategy hook (not gateway-registered) | `EvidenceEvaluatorStrategy` in `plugin_spi.py` |
| Evidence record shape (no duplicate model) | `reconciliation_evidence.py` — `ExternalEffectEvidence` |

## Input contract

`EvidenceEvaluationContext` carries:

- `tenant_id`, `correlation_id`, `contract_id` — correlation references
- `evidence_items` — one or more `ExternalEffectEvidence` records (refs + provenance via `operation_link`)
- `collected_at` — optional timestamp anchor

Platform checks (domain-neutral):

| Dimension | Rule (summary) |
| --- | --- |
| Availability | At least one evidence item |
| Completeness | At least one definitive, non-`INSUFFICIENT` verdict |
| Provenance | Operation link matches tenant, correlation, contract |
| Consistency | No conflicting definitive verdicts across items |
| Confidence | Uses existing `ExternalEffectEvidenceConfidence` on each item |

## Output contract

`EvidenceEvaluationResult.outcome`:

| Outcome | Meaning |
| --- | --- |
| `READY_FOR_DECISION` | Resolution planning may proceed |
| `INSUFFICIENT_EVIDENCE` | Defer — more reconciliation or evidence required |
| `CONFLICTING_EVIDENCE` | Fail closed — contradictory definitive reads |
| `EVALUATION_FAILED` | Fail closed — invalid episode, provenance, or evaluator |

Rationale strings are operational (for example `missing_evidence`, `invalid_provenance_correlation`) — never business verdicts.

## Relationship with reconciliation

- Reconciliation plugins and probe executors produce `ReconciliationProbeResult`.
- Runtime `materialize_external_effect_evidence_from_probe` builds `ExternalEffectEvidence`.
- `evaluate_external_effect_evidence` consumes that output (and optional additional items) without re-running probes.

## Relationship with resolution

`plan_external_effect_resolution` invokes evaluation first:

- `READY_FOR_DECISION` — existing disposition and plugin strategy flow unchanged
- `INSUFFICIENT_EVIDENCE` — defer plan with `evidence_evaluation` attached
- `CONFLICTING_EVIDENCE` / `EVALUATION_FAILED` — `ResolutionOrchestrationError` (no silent fallback)

## Future extension

`EvidenceEvaluatorStrategy` allows optional composition-time extensions. It is intentionally **not** registered on `EnterpriseReliabilityPluginGateway` in this iteration to avoid a duplicate plugin surface.

## Related documents

- [`ERL_RECONCILIATION_PLUGIN_INTEGRATION.md`](ERL_RECONCILIATION_PLUGIN_INTEGRATION.md)
- [`ENTERPRISE_RELIABILITY_LAYER.md`](ENTERPRISE_RELIABILITY_LAYER.md)
- [`RECONCILIATION.md`](RECONCILIATION.md)
