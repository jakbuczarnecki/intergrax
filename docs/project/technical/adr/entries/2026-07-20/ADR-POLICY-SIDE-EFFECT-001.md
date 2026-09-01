# ADR-POLICY-SIDE-EFFECT-001: Meaningful external side effects require policy authorization before execution

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC |
| **Related** | [ADR-GOVERNED-CONTINUATION-001](ADR-GOVERNED-CONTINUATION-001.md) · [ADR-EXTWORK-002](ADR-EXTWORK-002.md) · GEC-5 · Platform consolidation [`governed_external_execution.md`](../../../platform/governed_external_execution.md) |

## Context

External actions can create commitments, mutations, disclosures, or irreversible consequences. Existing runtime policy (`PolicyDecision` / `PolicyAction` / `PolicyEngine`) covers agent decisions, interrupts, and pre-LLM/output gates, but not a reusable pre-execution gate for meaningful **external** side effects. Without an explicit boundary, consumers risk provider-first calls, embedding approval rules in Tier-2, or treating continuation evidence as authorization.

## Decision

1. Introduce a minimal platform request contract `MeaningfulSideEffectRequest` (action, coarse `MeaningfulSideEffectKind`s, identity, correlation, context) - **not** a large action taxonomy and **not** quote-/payment-specific fields.
2. Extend `RuntimePolicyEngine` / `PolicyEngine` with `evaluate_meaningful_side_effect` returning existing `PolicyDecision`. Default is **fail closed** (DENY when identity/principal missing or no matching rule / indeterminate).
3. Expose injectable `MeaningfulSideEffectEvaluator` Protocol for Tier-3 composition roots.
4. Map `REQUIRE_HUMAN` / `ESCALATE` to Governed Continuation composition (GEC-4); policy never resumes Nexus.
5. External Work is the first consumer: gate `CREATE_EXTERNAL_WORK`, `ACCEPT_QUOTE`, and `CANCEL_EXTERNAL_WORK` before provider-bound Protocol methods. Quote receipt remains observational.

Rejected: inventing `INDETERMINATE` as a new `PolicyAction` (map to DENY); embedding business rules in Tier-2; treating `QuoteAcceptanceEvidence` as automatic ALLOW; provider-side authorization.

## Consequences

### Positive

- Reuses policy vocabulary and Governed Continuation
- Provider-neutral; reusable beyond External Work
- Fail-closed prevents silent side effects

### Negative

- Callers must inject an evaluator and supply principal + real Nexus `run_id`
- Product policy packs / thresholds remain separate host work

## Compliance

- No new policy or interrupt runtime
- No provider transport
- Tier-2 owns description + composition only

## Implementation notes

- Contracts: `intergrax/contracts/meaningful_side_effect.py`
- Policy: `intergrax/runtime/policy/meaningful_side_effect.py`, `runtime_policy_engine.py`, `policy_engine.py`
- Consumer: `agents/external_contractor_adapter/external_work_adapter.py`
- Host injection: `settings.meaningful_side_effect_policy`
- Verify: `uv run pytest tests/unit/runtime/policy/test_meaningful_side_effect_policy.py agents/external_contractor_adapter/tests -q`
