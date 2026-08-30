# DECISION_SYSTEM — authority and finalization

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

## 1. Candidate vs authoritative

Candidates are proposals under test. **Authoritative** outcomes are terminal per scope — either one accepted version or one resolution record.

## 2. One terminal outcome

Finalize guard enforces **at most one** terminal authoritative lifecycle outcome per decision scope.

## 3. Decision ≠ Authorization ≠ Execution

| Layer | Question |
| ----- | -------- |
| Decision | What did the system conclude? |
| Authorization | May this action proceed under policy? |
| Execution | What did Nexus actually do? |

## 4. Policy / HITL / Execution Authority

Decision System invokes HITL; Policy/Governed Execution owns execution authorization. Correct ACCEPTED decisions may still be blocked pending human approval.

## 5. Version-bound authorization

Execution authorization must cite the exact Decision Version. Post-revision approvals require re-authorization.

## 6. Finalization contract

Finalization persists durable authoritative artifacts and closes the lifecycle for the scope — without deleting candidate history.
