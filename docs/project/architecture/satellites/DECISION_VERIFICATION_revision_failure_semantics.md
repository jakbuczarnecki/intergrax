# DECISION_VERIFICATION — revision and failure semantics

**Parent hub:** [`DECISION_VERIFICATION.md`](../DECISION_VERIFICATION.md)

## Verifier does not mutate decision

Stages emit results only. Revision is lifecycle-owned.

## Challenge → revision request

Challenge carries structured insufficiency signals consumed by revision policy to mint `v(n+1)`.

## Unavailable verifier

Required verifier unavailable → fail closed (profile may route to `UNRESOLVED` or HITL — never silent pass).

## Conflicting verifiers

Irreconcilable required stage conflict → adjudication or `UNRESOLVED` — not arbitrary last-stage wins.

## Required verifier failure

Failed required stage blocks acceptance — may trigger bounded revision or terminal `REJECTED` / `UNRESOLVED`.

## Bounded revision

Revision loops respect configured iteration ceilings; resume cannot expand budget.

## Stage verdict vs Decision Resolution

Passing verification stages is necessary but not sufficient for ACCEPTED — Lifecycle applies resolution and finalization rules.
