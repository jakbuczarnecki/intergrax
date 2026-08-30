# DECISION_SYSTEM — implementation integration

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

## Nexus integration (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-NEXUS-01 | P0 | Graph / UAEP hooks → Decision Lifecycle | **Planned** |
| DS-NEXUS-02 | P1 | Lifecycle stage persistence via Nexus checkpoint ports | **Planned** |

## Governance / HITL / Execution Authority (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-GOV-01 | P1 | Version-bound authorization handoff to Governed Execution | **Planned** |
| DS-GOV-02 | P1 | HITL invocation for approver / adjudicator (remove L2 Critic) | **Planned** |

## Observability / Diagnostics (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-OBS-01 | P1 | Decision lifecycle audit events on observability spine | **Planned** |
| DS-OBS-02 | P2 | Diagnostics feed boundaries (no lifecycle ownership) | **Planned** |

## Persistence / recovery / concurrency (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-REC-01 | P0 | Finalize idempotency + conflict detection | **Planned** |
| DS-REC-02 | P1 | Crash resume without duplicate authoritative outcome | **Planned** |
| DS-REC-03 | P1 | Budget ceiling preserved on resume | **Planned** |

## Critic clean-cut migration (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-MIG-01 | P0 | Wire graph/UAEP paths to Decision Lifecycle | **Planned** |
| DS-MIG-02 | P0 | Retire `CriticOrchestrator` after pipeline parity | **Planned** |
| DS-MIG-03 | P1 | Remove L2 from verification model; route HITL via Lifecycle | **Planned** |
| DS-MIG-04 | P1 | DELETE CRITIC_VERIFICATION docs + retire `intergrax/runtime/critic/**` | **Planned** |
| DS-MIG-05 | P2 | Update application CriticProfile/CVL references | **Planned** |

## Failure / security hardening (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-SEC-01 | P0 | Execution identity binding on all decision records | **Planned** |
| DS-SEC-02 | P1 | Stale approval protection across revisions | **Planned** |

## Cross-scenario qualification + E2E (PLANNED)

See Phase DS-E2E rows in parent hub — real Docker E2E required before production-qualified claim.

## Implementation history

No implementation history yet — populate when DS-CORE work begins.
