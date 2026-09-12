# ERL-QUAL-004 — Full execution integration

End-to-end proof composition for the enterprise payment uncertainty recovery scenario. Business semantics stay in dataset slices and external SoR persistence; the execution layer orchestrates existing ports only.

## Flow

```text
Dataset (manifest + variant slice)
  → Provisioning (ScenarioProvisioningPort / reference provisioner)
  → Application workflow (EnterprisePaymentWorkflowService)
  → External payment boundary (ExternalPaymentCaptureService)
  → ERL admission (admit_external_effect_unknown_with_contract)
  → Reconciliation probe (ScenarioExternalRealityReconciliationPlugin)
  → Evidence evaluation (PaymentEvidenceEvaluatorPlugin + platform evaluate)
  → Resolution (PaymentResolutionStrategyPlugin)
  → Governance (PaymentGovernancePolicyPlugin)
  → Recovery (PaymentRecoveryStrategyPlugin)
  → ScenarioExecutionProofResult
```

## Ownership

| Layer | Location | Role |
| --- | --- | --- |
| Execution composition | `application/execution/` | Orchestrates phases; no parallel engines |
| Application | `application/services/` | Order load + capture request |
| External payment | `external_payment/` | SoR truth + UNKNOWN integration channel |
| ERL plugins | `erl_integration/` | Payment-specific SPI implementations |
| Platform ERL | `intergrax/runtime/enterprise_reliability/` | Admission, planning, probe execution, governance orchestration |

## Boundaries

- Lookups after capture read **persisted external reality**, not variant labels.
- Variant id selects **dataset materialization** only (provisioning + reconciliation metadata).
- No `PaymentScenarioEngine`, custom workflow engine, or duplicated platform contracts.
- Lab wiring: `build_lab_execution_composition()` in `application/execution/composition.py`.

## Variant behavior (data-driven)

| Variant slice | External SoR fact | Typical recovery posture |
| --- | --- | --- |
| `payment_completed_after_unknown` | Paid truth available | Continue fulfillment after governance |
| `payment_failed_after_unknown` | Failed truth available | Controlled stop / reservation release |
| `payment_truth_unavailable` | Truth unavailable | Escalate or wait |

## Proof output

`ScenarioExecutionProofResult` carries scenario id, variant id, correlation id, lifecycle outcome, and references to platform `ResolutionDecision`, `GovernanceDecision`, and `RecoveryDecision` plus evidence refs — no duplicate platform models.

## Tests

- `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_erl_qual_004_full_execution_e2e.py`
- `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_erl_qual_004_full_execution_architecture.py`
