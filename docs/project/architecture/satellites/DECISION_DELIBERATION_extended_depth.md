# DECISION_DELIBERATION — extended architecture

**Parent hub:** [`DECISION_DELIBERATION.md`](../DECISION_DELIBERATION.md)

## Strategy architecture

**DecisionStrategy** is the extension point for proposal and optional multi-participant deliberation. Council is one strategy — not a platform runtime.

## Strategy capability model

| Capability | Description |
| ---------- | ----------- |
| Single-shot proposal | Emit one or more candidate versions |
| Multi-round deliberation | Bounded rounds under shared Nexus budget |
| Parallel proposals | Branching candidates with preserved lineage |
| Disagreement capture | Structured artifact when participants diverge |
| Synthesis | Optional merged candidate for verification — does not erase dissent |

## Participant contracts

Participants are configured **roles** with visibility policies — not hard-coded persona names in platform core.

## Related satellites

| Topic | Route |
| ----- | ----- |
| Council / disagreement | [`DECISION_DELIBERATION_council_disagreement.md`](DECISION_DELIBERATION_council_disagreement.md) |
| Independence / visibility | [`DECISION_DELIBERATION_independence_context_visibility.md`](DECISION_DELIBERATION_independence_context_visibility.md) |
| Bounded execution | [`DECISION_DELIBERATION_bounded_execution.md`](DECISION_DELIBERATION_bounded_execution.md) |
