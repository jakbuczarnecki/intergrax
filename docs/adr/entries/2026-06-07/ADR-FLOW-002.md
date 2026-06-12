# ADR-FLOW-002: Reserved lifecycle states (`WAITING_FOR_RESOURCES`, `EXPIRED`)

| Field | Value |
|-------|-------|
| **Status** | Accepted (FLOW-10) |
| **Date** | 2026-06-07 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/NEXUS_EXECUTION_FLOW.md`](../../architecture/NEXUS_EXECUTION_FLOW.md) §8 · plan `FLOW-10` / `FLOW-GAP-08` |

## Context

`TaskState` includes `WAITING_FOR_RESOURCES` and `EXPIRED` with valid transitions in `TaskLifecycle`, trace mappings in `trace_bridge.py`, and inclusion in `LongRunningCoordinator.paused_states()`. No Nexus graph runner yet **enters** these states during a standard single-run execution path.

Authors and auditors flagged these as "dead" enum values (`FLOW-GAP-08`).

## Decision

Adopt **Option A — retain as reserved v1 lifecycle states**:

1. **`WAITING_FOR_RESOURCES` and `EXPIRED` remain** in `TaskState` and `TaskLifecycle` for forward compatibility with long-running, scheduler, and HITL timeout flows.
2. **Harness v1 Nexus graph execution** does not transition into these states; long-running coordinator and partial-result templates may reference them for pause/escalation UX.
3. **Future work** (outside Phase FLOW): dedicated scheduler or resource-gate runner sets `WAITING_FOR_RESOURCES`; HITL timeout policy sets `EXPIRED`.
4. **Documentation** in flow reference §8 marks both states as **reserved / future** — not a runtime bug.

**Not chosen (v1):**

| Option | Why deferred |
|--------|--------------|
| Enum trim | Breaks trace bridge, partial results, and coordinator contracts; high churn for low gain |
| Full scheduler implementation | Tier-1 scope creep; belongs to long-running product band |

## Consequences

- `FLOW-GAP-08` closed as documentation + ADR acceptance; no false "unused enum" audit findings.
- Gate tests and lifecycle diagrams stay honest about reserved transitions.
- Product teams can implement resource-wait and expiry runners without enum migration.

## Compliance (FLOW-10 acceptance)

- ADR status → **Accepted**
- Flow reference §23 `FLOW-GAP-08` paydown
- No runner changes required for v1 closeout
