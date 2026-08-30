# DECISION_SYSTEM — concurrency and recovery

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

## 1. Parallel branches

```text
       → v2A
v1 ──┤
       → v2B
```

Both branches preserve immutable history. **No last-write-wins** finalization.

## 2. Finalize conflict

Concurrent finalize attempts for the same scope must be idempotent and conflict-detected — duplicate authoritative outcomes are forbidden.

## 3. Idempotency

Resume and retry paths must not mint duplicate terminal outcomes or duplicate side effects tied to the same decision scope.

## 4. Crash / resume

Lifecycle stage, version lineage, and finalize guard state are persisted via **Nexus checkpoint** — no Decision-owned checkpoint engine.

## 5. Budget preservation

Resume cannot expand a previously granted Nexus budget ceiling. Deliberation, verification, and revision share the hosting execution budget.

## 6. Duplicate prevention

Finalize guards and idempotent persistence keys prevent double authoritative decisions after process death.
