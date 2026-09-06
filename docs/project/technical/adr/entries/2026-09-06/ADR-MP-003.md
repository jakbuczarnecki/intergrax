# ADR-MP-003: WorkItem vs Nexus Task — Shared Work ownership

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and contract planning only; MP-2 runtime implementation NOT STARTED |
| **Date** | 2026-09-06 |
| **Deciders** | Intergrax platform architecture (MP-2 ownership freeze) |
| **Related** | [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [ADR-MP-001](../2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../2026-08-11/ADR-MP-002.md) |

## Context

Multiplayer AI MP-2 requires platform-owned shared work primitives:

- **WorkItem** — durable collaborative work identity and lifecycle,
- **Assignment** — explicit collaborative assignment to principals,
- optimistic concurrency and idempotency for authoritative mutations,
- zero..N execution linkage to Nexus/UER identities without conflating planes.

MP-0 provisionally listed `ORCHESTRATION`, `UNIFIED_EXECUTION_RUNTIME`, and `BACKGROUND_TASKS` as likely MP-2 owners pending bounded verification. Repository canon already states:

- **CW-INV-06:** `WorkItem != Nexus Task`; collaborative work plane != execution plane.
- **COLLABORATIVE_WORK** ownership boundary lists WorkItem, Assignment, WorkArtifact, Decision, Activity as future MP-2…MP-6 extensions of the same collaborative work plane.
- **MP-INV-14:** Nexus does not own collaborative WorkItem lifecycle.
- **UER** owns Task, Run, Attempt, Execution lifecycle — not collaborative work semantics.
- **Orchestration** owns topology and graph policy — not WorkItem or Assignment lifecycle.
- **Background Tasks** owns enqueue/dispatch/worker execution of registered background handlers — not collaborative WorkItem authority.

Split ownership would violate PLATFORM-INV-003 (single ownership) and create lifecycle authority conflicts (e.g. inferring WorkItemState from TaskState or deleting WorkItem when a run ends).

## Alternatives considered

### 1. ORCHESTRATION as primary owner

Rejected. Orchestration owns accepted topology, planner selection, and graph-level policy. It may consume WorkItem context and emit execution linkage; it must not own WorkItem lifecycle, Assignment lifecycle, or collaborative authority.

### 2. UNIFIED_EXECUTION_RUNTIME / Nexus as primary owner

Rejected. UER owns execution units (`Task`, `run_id`, `Attempt`, `RuntimeEvent`). WorkItems are durable, independently addressable, may exist with zero active executions, span multiple tasks/runs, and survive execution completion. Subclassing Task, renaming Task to WorkItem, or wrapping Task as collaborative source of truth is forbidden.

### 3. BACKGROUND_TASKS as primary owner

Rejected. Background task infrastructure (`TaskDefinition`, `TaskRequest`, `TaskHandle`, worker dispatch) schedules and executes registered handlers. It may execute work associated with a WorkItem but cannot own WorkItem, Assignment, or WorkItemState. Background task records are execution/transport semantics, not collaborative work authority.

### 4. Split ownership across execution domains

Rejected. Would preserve ambiguous lifecycle authority and encourage silent substitution (TaskState → WorkItemState, background job status → shared work state).

### 5. COLLABORATIVE_WORK extends MP-1 plane (Accepted)

Accepted. MP-1 established the collaborative work plane for identity and authority. MP-2 extends the same domain with WorkItem and Assignment as first-class collaborative primitives, reusing MP-1 authority, repository concurrency/idempotency patterns, and durable persistence architecture.

## Decision

1. **COLLABORATIVE_WORK** is the single canonical owner of MP-2 Shared Work:

   | Owned by Collaborative Work | Not owned (reuse only) |
   |-----------------------------|-------------------------|
   | WorkItem identity and lifecycle | Nexus Task lifecycle |
   | WorkItemState (collaborative) | TaskState / run lifecycle |
   | Assignment identity and lifecycle | Execution scheduling and retries |
   | Collaborative ownership / assignment semantics | Workflow graph execution |
   | Collaborative optimistic concurrency | Worker/process scheduling |
   | Collaborative idempotency for authoritative mutations | Background task runtime ownership |
   | Work-level tenant/workspace isolation | Orchestration topology |
   | Work-level authority requirements (via MP-1) | UER runtime events as authority |
   | Links from WorkItem to execution identities | |

2. **WorkItem != Nexus Task** (hard invariant):

   - WorkItem is durable collaborative work; independently addressable; may exist with zero active executions; may span multiple tasks/runs; may survive execution completion; may have assignments to multiple principals; may be reopened/reassigned per collaborative lifecycle.
   - Nexus Task is an execution unit in the execution/orchestration plane; may be created to advance a WorkItem; is not the collaborative source of truth.
   - No 1:1 identity assumption.

3. **Assignment != AgentAssignment** (naming boundary):

   - **Assignment** (MP-2) is a collaborative primitive: `work_item_id`, `principal_id`, assignment state, revision, authority/lifecycle provenance; supports multiple principals and assignment history.
   - **AgentAssignment** (when used elsewhere in canon) denotes runtime/agent execution assignment — not collaborative WorkItem assignment. Do not encode assignments as a single `WorkItem.assignee_id` when MP-2 requires multi-principal semantics.

4. **Execution linkage model:**

   ```text
   WorkItem → zero..N execution links
   ```

   Each link may reference `task_id`, `run_id`, and `attempt_id` where justified. Execution identities are references/provenance, not WorkItem authority. Deleting or ending a run must not delete WorkItem. WorkItem state must not be inferred solely from TaskState. No concrete persistence schema is frozen in this ADR.

5. **Authority reuse (MP-1):**

   Collaborative mutations (create WorkItem, assign, reassign, change state, close/reopen, cancel) pass through Collaborative Work authority/policy enforcement. MP-2 owns resource semantics; MP-1 owns collaborative authority foundation. No separate WorkItem ACL engine.

6. **Lifecycle, concurrency, idempotency:**

   - WorkItem lifecycle is collaborative — explicit, validated, deterministic transitions; optimistic-concurrency protected; authority checked; auditable. Do not copy TaskState. No arbitrary caller-supplied state replacement.
   - Reuse MP-1 repository semantics: revision 0 create, `expected_revision` CAS, typed conflict, idempotency replay. Required for WorkItem and Assignment authoritative mutations. No silent last-write-wins.

7. **Durability:**

   WorkItem and Assignment are authoritative collaborative state. Production implementations use durable persistence via Collaborative Work repository ports (in-memory reference → SQLite local/dev → production-qualified relational adapter). No separate SharedWork database subsystem.

8. **Extension boundaries (not MP-2):**

   - MP-3: WorkArtifact / WorkArtifactVersion — not WorkItem payload.
   - MP-4: Decision / Approval — not encoded in WorkItem state machine.
   - MP-6: Collaborative Activity — mutations expose stable identity/revision for future projection; no activity feed in MP-2.
   - MP-7 / LKW: channel/thread/message IDs remain adapter mappings, not canonical WorkItem identity.

9. **Dependency direction:**

   Collaborative Work may reference neutral execution identity contracts. Execution runtime must not depend on Tier-3 Multiplayer types. Prefer narrow neutral reference contracts over circular imports.

10. **Reused execution domains (non-owners):**

    | Domain | Role for MP-2 |
    |--------|----------------|
    | ORCHESTRATION | May consume WorkItem context; explicit bridge only |
    | UNIFIED_EXECUTION_RUNTIME / NEXUS | Task/run/attempt identities; execution outcomes |
    | BACKGROUND_TASKS | May schedule/execute work associated with a WorkItem |
    | OBSERVABILITY / PROOF_RECEIPTS | Provenance and evidence consumption |

## Consequences

### Positive

- Single lifecycle authority for shared work; no TaskState substitution.
- MP-2 builds on proven MP-1 authority, concurrency, and persistence patterns.
- Clear bridge point for orchestration and Nexus without tier violations.
- Reusable across LKW, autonomous workers, enterprise workflows, and external agents.

### Negative

- Collaborative Work domain scope grows; implementation waves (COLLAB-WORK-2A…2G) required before MP-2 closure.
- Execution bridge (COLLAB-WORK-2F) must be designed carefully to avoid implicit state propagation.

## Compliance

- PLATFORM-INV-001 / PLATFORM-INV-003: single domain ownership preserved.
- CW-INV-06, MP-INV-14, MP-INV-19…21 honored.
- Tier boundaries preserved: UER/Nexus do not import Tier-3 Multiplayer types.
- Linked architecture, plan, and feature docs updated in MP-2 ownership freeze task.

## Implementation notes

- Implementation begins at **COLLAB-WORK-2A** (contracts + lifecycle semantics) after this ADR is accepted.
- Verification: `python scripts/maintenance/check_harness_adr.py`; documentation link integrity tests; `git diff --check`.
