# DECISION_SYSTEM — identity, versioning, and lineage

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

## 1. Decision identity

**Decision ID** identifies a decision scope across the hosting execution tree. It is stable for the life of the decision thread.

## 2. Decision Version identity

Each **Decision Version** is immutable once minted. Revisions append `v(n+1)` — they never mutate `v(n)` in place.

## 3. Binding dimensions

Every record that can affect authority must bind:

- Decision ID
- Decision Version
- decision scope (domain-defined)
- tenant
- execution identity (`TaskId` / `RunId` / `AttemptId` / TARGET `ExecutionId`)

## 4. Lineage model

```text
v1 (candidate) → challenge → v2 → branch → v2A | v2B → adjudication / UNRESOLVED / accepted version
```

Parent/branch lineage is preserved for audit even after finalization.

## 5. Stale approval protection

Human or policy approval for `v1` is **invalid** after a revision mints `v2`. Authorization records must reference exact version or fail closed.

## 6. Exact version binding

Verification results, challenges, adjudication outcomes, and execution authorization records are all version-bound — loose context dicts are not authority identity.
