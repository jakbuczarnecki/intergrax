# ADR-MP-002: Principal / Membership / Delegation

| Field | Value |
|-------|-------|
| **Status** | Accepted - semantic contracts frozen; persistence and runtime implementation NOT STARTED |
| **Date** | 2026-08-11 |
| **Deciders** | Intergrax platform architecture (MP-1A ownership freeze) |
| **Related** | [ADR-MP-001](ADR-MP-001.md) · [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) |

## Context

ADR-MP-001 assigns COLLABORATIVE_WORK as owner. MP-1 requires a coherent semantic block for collaborative identity and authority before any persistence, API, or service implementation.

Existing contracts provide **execution and enforcement hooks**, not collaborative semantics:

- `RequestIdentity` + `PrincipalType` (`USER`, `SERVICE`, `ORG_SYSTEM`) - authenticated principal for one agent run.
- `DelegationSpec` - Nexus graph child-run delegation (objective, child agent, isolated memory namespace).
- `MeaningfulSideEffectRequest.principal_id` - optional field for policy evaluation before external side effects.

These must be reused at enforcement boundaries without becoming the canonical collaborative Principal, Membership, or authority-Delegation model.

## Decision

### Principal

Collaborative Principal is a platform semantic distinct from `AgentDefinition`, `AgentAssignment`, `AgentRun`, and run-scoped `RequestIdentity`.

Conceptual kinds (semantic, not a frozen implementation enum):

| Kind | Meaning |
|------|---------|
| `HUMAN` | Human collaborator |
| `AGENT` | Platform or product agent acting as collaborative actor |
| `SERVICE` | Non-human service principal |
| `EXTERNAL_AGENT` | Future external interoperable actor (MP-8 boundary) |

Implementation enum/class location remains open until MP-1 implementation justifies placement in `intergrax/contracts/`.

### WorkspaceMembership

- Membership is **explicit** collaborative membership in a workspace scope.
- Scope carries `tenant_id` + `workspace_id` where applicable.
- Role and capability direction is part of membership semantics; membership is **not inferred** from tenant/workspace identifiers alone.
- `tenant_id` is ownership/admin scope - not authorization proof.

### Delegation (authority)

Distinct from `DelegationSpec` (execution graph child run):

- **Delegator principal** - authority source.
- **Delegate principal** - actor receiving scoped authority.
- **Scoped authority** - bounded subset of delegator authority.
- Optional **resource** and **time** boundaries (direction only; concrete fields at implementation).
- **Never amplifies** delegator authority.
- Agent authority must not silently equal delegating human authority.

### Effective authority

Resolved authority for a collaborative mutation is the intersection of all applicable constraints:

```text
principal authority
  ∩ WorkspaceMembership
  ∩ Delegation where applicable
  ∩ workspace policy
  ∩ resource policy
  ∩ runtime/tool policy
```

When required authority cannot be proven for a privileged or meaningful mutation, the platform **fails closed**.

Policy/runtime mechanisms evaluate the resolved intersection at enforcement points; COLLABORATIVE_WORK owns the semantic source of truth for Principal, Membership, and authority Delegation.

### Explicit non-decisions (deferred to MP-1 implementation)

- Persistence model (database tables, repositories).
- HTTP/API surfaces.
- Concrete RBAC storage.
- Enum placement and exact contract field names in code.

## Distinction register

| Concept | Owner | Not equivalent to |
|---------|-------|-------------------|
| Collaborative Principal | COLLABORATIVE_WORK | `RequestIdentity`, `AgentRun` |
| WorkspaceMembership | COLLABORATIVE_WORK | tenant/workspace ID alone |
| Authority Delegation | COLLABORATIVE_WORK | `DelegationSpec` (Nexus graph) |
| Effective authority | COLLABORATIVE_WORK (semantics) + Policy (evaluation) | any single ID field |
| Run intake identity | UNIFIED_EXECUTION_RUNTIME | collaborative Principal |
| Execution delegation | UNIFIED_EXECUTION_RUNTIME / Nexus | authority Delegation |

## Consequences

### Positive

- One semantic block for MP-1 implementation.
- Clear reuse of policy enforcement without duplicate identity models.
- Preserves WorkItem != Task and Decision != HITL boundaries.

### Negative

- Bridge mapping required between collaborative Principal and run-scoped intake at execution boundaries.

## Compliance

- No persistence or service implementation in MP-1A.
- Memory not used as Membership/Delegation source of truth.
- LKW remains consumer.

## Implementation notes

- Contract stubs land under COLLAB-WORK-1A after MP-1 review acceptance.
- Contract tests per [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) COLLAB-WORK-1A proof requirements.
