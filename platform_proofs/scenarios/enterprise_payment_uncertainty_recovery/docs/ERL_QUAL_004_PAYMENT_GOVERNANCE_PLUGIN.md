# ERL-QUAL-004 — Payment Governance Policy Plugin

## Why this plugin exists

After reconciliation and resolution recommend continuing enterprise work, a payment capture may still require **human or policy approval** (amount limits, customer risk, missing order context). The platform exposes a generic `GovernanceStrategy` SPI and `GovernanceDecision` outcomes; this scenario supplies **payment-specific policy** without pushing business rules into `intergrax/`.

## Ownership

| Layer | Responsibility |
| --- | --- |
| **Platform (`intergrax/`)** | `GovernanceStrategy`, `GovernanceStrategyEvaluationRequest`, `GovernanceDecision`, `HumanApprovalRequirement`, orchestration (`evaluate_external_effect_governance`) |
| **Scenario (`platform_proofs/.../erl_integration/`)** | `PaymentGovernancePolicyPlugin`, enterprise thresholds (`PaymentEnterpriseGovernancePolicy`), business context lookup (`PaymentGovernanceBusinessContextLookupPort`), mapping to platform decisions |

The plugin does **not** implement reconciliation, evidence collection, recovery execution, or ERL lifecycle internals.

## Governance flow

```text
ResolutionDecision (e.g. CONTINUE)
        +
RecoveryDecision
        +
ExternalEffectEvidence
        +
EnterpriseReliabilityStrategyContext
        ↓
PaymentGovernancePolicyPlugin.evaluate(...)
        ↓
GovernanceDecision (ALLOW | APPROVAL_REQUIRED | DENY)
```

Registration is optional via `register_scenario_reconciliation_plugins(..., payment_governance_lookup=...)` on the same scenario plugin id used for reconciliation and resolution.

## Policy examples (business context driven)

1. **Standard payment** — amount below `human_approval_threshold_amount`, currency matches policy, no elevated customer risk → `ALLOW`.
2. **High-value payment** — amount at or above threshold with resolution `CONTINUE` → `APPROVAL_REQUIRED` with `HumanApprovalRequirement` (`erl-qual-004:governance:high_value_payment`).
3. **Insufficient context** — lookup failure or missing payment amount → `APPROVAL_REQUIRED` (fail closed).
4. **No risky continuation** — resolution action other than `CONTINUE` → `ALLOW` with rationale `no_risky_continuation_proposed`.

Decisions are derived from **amount, currency, and risk metadata** in `PaymentGovernanceBusinessContext`, not from dataset variant ids.

## Replaceability

- No singletons or global registries; policy and lookup ports are constructor-injected.
- Unit tests use `InMemoryPaymentGovernanceBusinessContextLookup`.
- Production-like lab wiring can bind PostgreSQL-backed context lookup in a later task.

## Related artifacts

- Plugin: `erl_integration/plugins/payment_governance_policy.py`
- Policy mapping: `erl_integration/mapping/payment_governance_decision.py`
- Tests: `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_payment_governance_policy_plugin.py`
