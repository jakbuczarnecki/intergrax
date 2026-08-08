# Approve policy learning adaptation

Operational runbook for human-gated `POLICY_LEARNING` adaptive proposals (Phase W-ADAPT-5.9).

## When to use

- An `AdaptationProposalPackage` with `AdaptiveLoopKind.POLICY_LEARNING` passed governance gates.
- Security team review is required before `AdaptationExecutor.apply()`.
- Tool deny-list or policy fragment changes are proposed.

## Prerequisites

- Approver identity registered in `PolicyLearningApprovalStore`.
- Proposal id and loop envelope visible in `build/adaptive_harness/proposals.json`.
- Security sign-off recorded in change ticket.

## Steps

1. Review proposal summary and `ProfileVersionDraft` payload (policy fragment).
2. Run V-SEC adversarial harness baseline (prompt/tool/retrieval suites).
3. Record approver in the approval store:

```python
approval_store.approve(
    proposal_id="prop_abc123",
    approver_id="owner:security",
)
```

4. Only after approval, call `AdaptationExecutor.apply()` with the candidate version id.
5. Monitor VerificationLoop for the applied profile during the SLO window.

## Forbidden

- Applying policy learning proposals without approver (**KPI target: zero** unauthorized applies).
- Bypassing `require_policy_learning_approval()` in strict environments.

## Related artifacts

- `build/adaptive_harness/proposals.json`
- [guides/HARNESS_ENVIRONMENT.md](../../docs/project/technical/guides/HARNESS_ENVIRONMENT.md) — Phase V security contracts
