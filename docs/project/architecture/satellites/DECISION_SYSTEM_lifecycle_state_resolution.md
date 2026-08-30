# DECISION_SYSTEM — lifecycle, state, and resolution

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

## 1. Stage vs resolution

**Lifecycle stage** describes orchestration progress (proposal, verifying, revising, …). **Decision Resolution** describes the substantive outcome: `ACCEPTED` | `REJECTED` | `UNRESOLVED`.

## 2. Resolution semantics

| Outcome | Meaning |
| ------- | ------- |
| ACCEPTED | A specific Decision Version satisfies gates — Authoritative Accepted Decision |
| REJECTED | Process completed — no version accepted as correct |
| UNRESOLVED | Insufficient basis for responsible resolution |

## 3. Termination independence

Execution may complete, fail, cancel, or budget-stop while Decision Resolution remains `UNRESOLVED`. Infrastructure failure does not auto-map to `REJECTED`.

## 4. Revision transitions

Challenges and adjudication requests mint **new versions** through explicit revision — verification never mutates candidates in place.

## 5. Adjudication

Optional stage resolves competing branches, verifier conflict, deadlocked Council, or human adjudication — may end in any resolution outcome including `UNRESOLVED`.

## 6. Finalization mapping

| Resolution | Finalization artifact |
| ---------- | --------------------- |
| ACCEPTED | Authoritative Accepted Decision |
| REJECTED / UNRESOLVED | Authoritative Resolution Record (no fake accepted version) |
