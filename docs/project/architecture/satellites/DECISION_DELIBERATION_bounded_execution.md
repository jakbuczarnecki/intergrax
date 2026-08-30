# DECISION_DELIBERATION — bounded execution

**Parent hub:** [`DECISION_DELIBERATION.md`](../DECISION_DELIBERATION.md)

## Rounds

Deliberation rounds are bounded by strategy configuration and Nexus budget.

## Budget sharing

Council, verification, and revision share the hosting execution budget — no separate Council budget engine.

## Parallelism

Parallel participant proposals are allowed; finalize semantics remain lifecycle-owned.

## Continuation vs revision vs technical retry

| Kind | Owner |
| ---- | ----- |
| Deliberation continuation | DecisionStrategy |
| Decision revision | Decision Lifecycle |
| Technical retry | Nexus Reliability |

## Crash / resume

Strategy state needed for resume is persisted through Nexus checkpoints — not a second scheduler.
