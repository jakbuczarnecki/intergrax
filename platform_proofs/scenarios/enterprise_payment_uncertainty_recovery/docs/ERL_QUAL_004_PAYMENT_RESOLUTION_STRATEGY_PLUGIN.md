# ERL-QUAL-004 — Payment resolution strategy plugin

## Why this plugin exists

After reconciliation and payment evidence evaluation reach `READY_FOR_DECISION`, the enterprise must decide **what the platform should do next** (continue, stop, escalate) without embedding payment policy in `intergrax/`.

`PaymentResolutionStrategyPlugin` implements the platform `ResolutionStrategy` SPI and returns only generic `ResolutionDecision` values.

## Platform vs scenario ownership

| Owner | Responsibility |
| --- | --- |
| **Platform** (`intergrax/`) | `ResolutionStrategy`, `ResolutionStrategyEvaluationRequest`, `ResolutionDecision`, `ResolutionPlatformAction`, resolution orchestration and lifecycle advice |
| **Scenario** (`erl_integration/`) | Payment reconciliation bundle interpretation, order-continuation policy, mapping to platform actions |

The platform never emits `PaymentApproved`, `OrderCancelled`, or similar business verdict types. The plugin maps payment truth and reconciliation probes to `CONTINUE`, `STOP`, or `ESCALATE`.

## Contract usage

Register with the scenario reconciliation bundle (same `plugin_id` as reconciliation for gateway resolution calls):

```python
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.wiring import (
    register_scenario_reconciliation_plugins,
)

register_scenario_reconciliation_plugins(
    registry,
    external_reality_lookup,
    payment_evidence_lookup=payment_evidence_lookup,
)
```

Orchestration invokes the strategy through `EnterpriseReliabilityPluginGateway.evaluate_resolution` during `plan_external_effect_resolution`. Inputs are `ExternalEffectEvidence`, `EnterpriseReliabilityStrategyContext`, and `ExternalEffectContract` — the same resolution planning context documented on the platform contract.

Payment attributes are loaded via `PaymentReconciliationEvidenceLookupPort` and cross-checked with `advise_payment_evidence_outcome` before a resolution action is chosen.

## Example decisions (evidence-driven)

| Evidence signal | Typical platform action |
| --- | --- |
| Definitive success probe + consistent PSP/settlement bundle | `CONTINUE` — continue business process |
| Definitive failure probe + consistent negative payment bundle | `STOP` — stop or cancel affected process |
| Insufficient probe, unavailable reconciliation, or incomplete payment bundle | `ESCALATE` — wait or escalate per reliability policy |
| Probe vs settlement mismatch | `ESCALATE` — contain automation until conflict is resolved |

Variant fixtures differ only in **dataset evidence content** — the strategy does not branch on variant identifiers.

## Related artifacts

- Evidence evaluator: `erl_integration/plugins/payment_evidence_evaluator.py`
- Mapping: `erl_integration/mapping/payment_resolution_decision.py`
- Tests: `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_payment_resolution_strategy_plugin.py`
