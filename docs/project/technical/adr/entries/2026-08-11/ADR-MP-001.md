# ADR-MP-001: Collaborative Work Plane ownership

| Field | Value |
|-------|-------|
| **Status** | Accepted - architecture and contract planning only; runtime implementation NOT STARTED |
| **Date** | 2026-08-11 |
| **Deciders** | Intergrax platform architecture (MP-1A ownership freeze) |
| **Related** | [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`plan/COLLABORATIVE_WORK.md`](../../../../maintainers/plans/COLLABORATIVE_WORK.md) · [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [ADR-MP-002](ADR-MP-002.md) |

## Context

Multiplayer AI MP-1 requires platform-owned collaborative identity and authority:

- Principal,
- WorkspaceMembership,
- Delegation,
- effective authority semantics.

Before runtime implementation, Intergrax requires exactly one architectural owner satisfying PLATFORM-INV-003 (single ownership). Provisional candidates in MP-0 included Platform Foundation, Application Hosting, and Unified Execution Runtime. None cleanly owns collaborative membership and authority semantics without violating existing boundaries.

Evidence from repository canon:

- **Application Hosting** owns process lifecycle and supervision; it explicitly does not own cognition, orchestration, business tasks, or identity/authorization semantics.
- **Unified Execution Runtime** owns execution (`Task`, `run_id`, UAEP). `RequestIdentity` is run-scoped authenticated intake; `DelegationSpec` is Nexus graph child-run delegation - not authority between collaborative principals.
- **Platform Foundation** owns tier structure and spine governance - not collaborative product semantics.
- **Policy** evaluates authority via incidental `principal_id` fields on requests such as `MeaningfulSideEffectRequest` - enforcement, not semantic source of truth for membership/delegation.
- **Memory** is explicitly not the membership/delegation source of truth in Multiplayer canon.

LKW is the first consumer. Platform primitives must not be owned by LKW or split across Nexus + Hosting + Runtime.

## Alternatives considered

### 1. PLATFORM_FOUNDATION

Rejected. Tier governance and verification gates - not collaborative identity semantics. Would dilute an already cross-cutting foundation domain.

### 2. APPLICATION_HOSTING

Rejected. Hosting answers how an application instance lives; not who collaborates or with what authority. ADR-HOST-001 established a dedicated domain when reusable hosting was discovered - the same pattern applies here for a distinct capability.

### 3. UNIFIED_EXECUTION_RUNTIME

Rejected. Execution plane owns tasks/runs and run-scoped `RequestIdentity`. Collaborative membership and authority delegation are durable cross-execution semantics that must outlive individual Nexus tasks.

### 4. POLICY (as storage owner)

Rejected. Policy must evaluate and enforce resolved authority without silently becoming the canonical store of Membership/Delegation records.

### 5. Document only under MULTIPLAYER_AI feature hub

Rejected as primary ownership. Feature docs coordinate domains; reusable platform capabilities require a domain pair per PLATFORM-INV-001 and PLATFORM-INV-003.

### 6. Create dedicated COLLABORATIVE_WORK domain

Accepted.

## Decision

1. Introduce platform domain pair:

   ```text
   docs/project/architecture/COLLABORATIVE_WORK.md
   docs/project/maintainers/plans/COLLABORATIVE_WORK.md
   ```

2. **COLLABORATIVE_WORK** is the single canonical owner of:

   - Principal (collaborative),
   - WorkspaceMembership,
   - Delegation (authority),
   - effective authority semantics,
   - future MP-2…MP-6 primitives on the collaborative work plane (WorkItem, WorkArtifact, Decision, Activity - each gated by its own ADR/MP phase).

3. **Dependencies compose existing capabilities:**

   | Capability | Role |
   |------------|------|
   | Policy / Runtime | enforcement at mutation boundaries |
   | Nexus / UER | execution under resolved authority |
   | UCL / Context Engineering / Memory | context composition (MP-5) |
   | Evidence / Observability | proof and trace linkage |
   | Application Hosting | host/application lifecycle only |

4. **Explicit non-ownership:**

   - Nexus does not own collaborative membership or WorkItem lifecycle.
   - Application Hosting does not own collaborative identity.
   - LKW remains consumer - no `LkwPrincipal`, `LkwWorkspaceMember`, `LkwDelegation`.
   - Memory is not Membership/Delegation source of truth.

5. **Multiplayer feature hub** coordinates MP phases but does not replace domain ownership.

6. **Migration principle:** existing incidental `principal_id` fields and `RequestIdentity` remain execution/policy bridge contracts; collaborative semantics migrate to COLLABORATIVE_WORK contracts without renaming execution identities.

## Consequences for MP-1…MP-6

| Phase | Consequence |
|-------|-------------|
| MP-1 | Implements contract slice under COLLABORATIVE_WORK |
| MP-2 | WorkItem / Assignment extend same domain (ADR-MP-003 gate) |
| MP-3 | WorkArtifact collaborative ownership extends same domain (ADR-MP-004 gate) |
| MP-4 | Decision semantics extend same domain; HITL remains bridge only |
| MP-5 | Principal-scoped ContextView boundary owned here; UCL/Memory compose |
| MP-6 | Activity collaborative semantics extend same domain |
| MP-7 | LKW adopts platform contracts without ownership transfer |

## Consequences

### Positive

- Single owner satisfies PLATFORM-INV-003.
- Preserves Nexus execution vs collaborative work separation.
- Enables MP-2…MP-6 to extend one plane without domain dumping.
- Reuses policy enforcement without conflating enforcement with storage.

### Negative

- New domain pair to maintain.
- Bridge contracts needed between collaborative Principal and run-scoped `RequestIdentity`.

## Compliance

- Tier boundaries preserved - contracts in Tier-0; no application-owned primitives.
- PLATFORM-INV-001, PLATFORM-INV-002, PLATFORM-INV-003 satisfied.
- Linked architecture, plan, and Multiplayer hub docs updated.
- No runtime implementation in MP-1A.

## Implementation notes

- MP-1 runtime begins at COLLAB-WORK-1A after review acceptance.
- ADR-MP-002 freezes semantic contracts for Principal / Membership / Delegation.
