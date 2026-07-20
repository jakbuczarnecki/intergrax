# ADR-GOVERNED-CONTINUATION-001: Governed Continuation as Nexus composition

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC |
| **Related** | [`ADR-EXTWORK-001`](ADR-EXTWORK-001.md) · [`ADR-EXTWORK-002`](ADR-EXTWORK-002.md) · GEC host / Tier-2 plans · Platform consolidation [`governed_external_execution.md`](../../../platform/governed_external_execution.md) |

## Context

GEC-4 must prove:

```text
execution › governance interruption › governance decision › continuation evidence › resume
```

without inventing a second orchestration stack. The platform already provides `ExecutionInterrupt`, `ExecutionInterruptHandler`, HITL `HumanDecisionRecord`, ACP/UAEP resume, and (for External Work) `QuoteAcceptanceEvidence` with `hitl_decision_id` / `interrupt_id` / `policy_decision_ref`.

Risks without an explicit decision:

- a quote-specific interrupt/lifecycle engine (`QuoteInterrupt`, `QuoteRuntime`)
- a parallel `ContinuationRuntime` / `ContinuationManager`
- Tier-2 evaluating approvals or resuming Nexus

## Decision

1. **Governed Continuation is composition**, not a new runtime — helpers live in `intergrax.contracts.governed_continuation` and map onto existing `ExecutionInterrupt` + `AgentDecision` + HITL evidence refs.
2. Introduce a **generic** `ContinuationReason` (`quote`, `security`, `legal`, `procurement`, `compliance`, `publication`). External Work supplies `QUOTE` only.
3. Keep **`QuoteAcceptanceEvidence`** as the minimum QUOTE continuation evidence — do not redesign it; expose reason-agnostic `ContinuationEvidenceRefs` that mirror its governance pointers.
4. Tier-2 may **surface** `GovernedContinuationRequest` and **forward** continuation evidence; it must not decide, authorize, or resume.
5. Nexus remains the only orchestration runtime; policy and HITL remain the decision owners.

Rejected: `ContinuationEngine`, `QuoteLifecycleEngine`, quote-only interrupt types, Tier-2-owned pause/resume.

## Consequences

### Positive

- Reusable for future interruption reasons beyond commercial quote
- Clear reuse of Nexus interrupt / HITL / resume
- External Work stays a specialization, not the axis of the capability

### Negative

- Host/UX wiring for presenting quotes and collecting HITL remains later product work
- Non-QUOTE reasons have no domain evidence mapper yet (by design)

## Compliance

- No new interruption framework classes
- No transport / partner SDK in continuation modules
- Tier boundaries preserved

## Implementation notes

- `intergrax/contracts/governed_continuation.py`
- Tier-2: `ExternalWorkAdapter.surface_continuation_blocker` / `forward_continuation_evidence`
- Verify: `uv run pytest tests/unit/contracts/test_governed_continuation.py agents/external_contractor_adapter/tests -q`
