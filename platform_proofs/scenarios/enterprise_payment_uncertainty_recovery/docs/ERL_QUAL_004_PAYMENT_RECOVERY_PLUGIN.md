# ERL-QUAL-004 — Payment Recovery Strategy Plugin

## Why this plugin exists

After resolution recommends continue, stop, or escalate, the enterprise must **execute payment-specific recovery** (resume fulfillment, release inventory reservation, open an operations queue) without embedding workflow logic in `intergrax/`. The platform exposes `RecoveryStrategy` and `RecoveryDecision`; this scenario supplies business interpretation and action execution through replaceable ports.

## Platform vs scenario ownership

| Layer | Responsibility |
| --- | --- |
| **Platform (`intergrax/`)** | `RecoveryStrategy`, `RecoveryStrategyEvaluationRequest`, `RecoveryDecision`, `RecoveryLifecycleAction`, recovery orchestration (`recommend_external_effect_recovery_lifecycle`) |
| **Scenario (`platform_proofs/.../erl_integration/`)** | `PaymentRecoveryStrategyPlugin`, `PaymentRecoveryActionPort`, mapping from `ResolutionDecision` to workflow steps and platform lifecycle posture |

The plugin does **not** implement reconciliation, evidence collection, governance policy, or ERL lifecycle mutation.

## Recovery flow

```text
ResolutionDecision (CONTINUE | STOP | ESCALATE | UNKNOWN | …)
        +
ExternalEffectEvidence
        +
EnterpriseReliabilityStrategyContext
        ↓
PaymentRecoveryStrategyPlugin.evaluate(...)
        ↓
PaymentRecoveryActionPort.execute(...)   ← scenario business adapter
        ↓
RecoveryDecision (CONTINUE | TERMINATE | ESCALATE | WAIT)
```

Registration is part of `register_scenario_reconciliation_plugins(...)` on the same scenario plugin id used for reconciliation and resolution.

## Example behavior (resolution-driven)

| Resolution action | Business recovery step | Platform lifecycle | Scenario execution status |
| --- | --- | --- | --- |
| `CONTINUE` | Resume fulfillment workflow | `CONTINUE` | `SUCCESS` |
| `STOP` | Release reservation / stop flow | `TERMINATE` | `SUCCESS` |
| `ESCALATE` / `COMPENSATION_REQUIRED` | Operational follow-up | `ESCALATE` | `ESCALATED` |
| `UNKNOWN` | Operational follow-up while truth pending | `WAIT` | `WAITING` |

Decisions follow **resolution and evidence context**, not dataset variant identifiers.

## Replaceability

- No singletons or global registries; `PaymentRecoveryActionPort` is constructor-injected.
- Lab tests use `InMemoryPaymentRecoveryActionPort` to assert business steps without ERP or PSP integrations.
- Production-like wiring can bind workflow adapters in a later task.

## Related artifacts

- Plugin: `erl_integration/plugins/payment_recovery_strategy.py`
- Action port: `erl_integration/contracts/payment_recovery_action.py`
- Mapping: `erl_integration/mapping/payment_recovery_decision.py`
- Tests: `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_payment_recovery_strategy_plugin.py`
