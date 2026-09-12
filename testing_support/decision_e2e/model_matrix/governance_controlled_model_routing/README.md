# Governance-controlled model routing (DS-E2E-15J-L5)

## Responsibility

Approve, block, or flag human approval for an existing `ModelSelectionRecommendation`. Governance does **not** choose a model and does **not** execute models or call the Execution Engine.

## Data flow

```text
ModelSelectionRecommendation + GovernanceTaskContext + capability evidence + policy refs
        → GovernanceEvaluationRequest
        → GovernanceEvaluationEngine (injected PolicyEvaluator plugins)
        → GovernanceDecision (disposition + audit metadata)
```

Downstream orchestration decides what to do with the decision.

## Inputs

- `model_recommendation` — output of the selection layer (may be absent; yields a controlled BLOCK).
- `task_context` — scenario, data sensitivity, risk tier.
- `applicable_policies` — which policy plugins to run (empty means all injected evaluators).
- `capability_evidence` — existing `ModelCapabilityProfile` rows used for compliance checks.

## Outputs

- `disposition`: `ALLOW`, `BLOCK`, or `REQUIRE_APPROVAL`
- `reason_references`, `policy_references`, `audit_metadata` (decision id, timestamps, data sources)

## Adding policies

Implement `PolicyEvaluator` with stable `policy_id` / `policy_version` and pass it in `GovernanceEvaluationEngine(evaluators=(..., YourPolicy()))`. Do not edit the engine.

Default plugins live in `policies.py` (`default_governance_policies()`).
