# ERL-QUAL-004 — Payment evidence evaluator plugin

## Why this plugin exists

Reconciliation materializes generic `ExternalEffectEvidence`, but enterprise payment qualification must judge **payment-specific reconciliation quality** (PSP confirmation identifiers, settlement posture, SoR consistency, source reliability) without pushing those rules into `intergrax/`.

`PaymentEvidenceEvaluatorPlugin` demonstrates that scenario-owned code can extend evidence evaluation through the existing `EvidenceEvaluatorStrategy` SPI while returning only platform outcomes.

## Platform vs scenario ownership

| Owner | Responsibility |
| --- | --- |
| **Platform** (`intergrax/`) | `EvidenceEvaluationContext`, `EvidenceEvaluationOutcome`, orchestration in `evaluate_external_effect_evidence`, `ExternalEffectEvidence` shape |
| **Scenario** (`erl_integration/`) | Payment attribute model (`PaymentReconciliationEvidence`), lookup ports, PSP/settlement interpretation, plugin implementation |

The platform never emits `PAYMENT_APPROVED` or `PAYMENT_FAILED`. The plugin maps payment reads to `READY_FOR_DECISION`, `INSUFFICIENT_EVIDENCE`, `CONFLICTING_EVIDENCE`, or `EVALUATION_FAILED`.

## Contract usage

Composition passes the plugin explicitly (no gateway registry):

```python
from intergrax.runtime.enterprise_reliability import evaluate_external_effect_evidence
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_evidence_evaluator import (
    PaymentEvidenceEvaluatorPlugin,
)

evaluator = PaymentEvidenceEvaluatorPlugin(_lookup=payment_evidence_lookup)
result = evaluate_external_effect_evidence(
    state=state,
    evidence=reconciliation_evidence,
    tenant_id=tenant_id,
    contract_id=contract_id,
    evaluator_strategy=evaluator,
)
```

The plugin applies when materialized evidence refs use the scenario prefix `evidence://erl-qual-004/`. It loads `PaymentReconciliationEvidence` by `correlation_id` and cross-checks probe verdicts against settlement and capture fields.

## Example outcomes (evidence-driven)

| Evidence signal | Typical platform outcome |
| --- | --- |
| PSP confirmation + settled capture aligned with definitive success probe | `READY_FOR_DECISION` |
| Definitive failure probe with unsettled, uncaptured payment bundle | `READY_FOR_DECISION` (negative resolution readiness) |
| Reconciliation unavailable / missing PSP confirmation / degraded source tier | `INSUFFICIENT_EVIDENCE` |
| Definitive success probe but settlement shows no capture | `CONFLICTING_EVIDENCE` |

Variant fixtures (A/B/C) differ only in **dataset evidence content** — the evaluator does not branch on variant identifiers.

## Related artifacts

- Reconciliation plugin: `erl_integration/plugins/external_reality_reconciliation.py`
- Mapping: `erl_integration/mapping/payment_evidence_evaluation.py`
- Tests: `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_payment_evidence_evaluator_plugin.py`
