# ADR-MP-008: Workspace platformization / LKW migration

| Field | Value |
|-------|-------|
| **Status** | Accepted — architecture and adoption contract gate only; MP-7B+ runtime implementation NOT STARTED |
| **Date** | 2026-09-19 |
| **Deciders** | Intergrax platform architecture (MP-7A ownership gate) |
| **Related** | [`capabilities/architecture/MULTIPLAYER_AI.md`](../../../../capabilities/architecture/MULTIPLAYER_AI.md) · [`capabilities/plan/MULTIPLAYER_AI.md`](../../../../capabilities/plan/MULTIPLAYER_AI.md) · [`architecture/COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) · [ADR-MP-001](../2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../2026-08-11/ADR-MP-002.md) · [ADR-MP-003](../2026-09-06/ADR-MP-003.md) · [MP-7A gate](../../../../maintainers/qualification/MP-7A_LKW_MULTIPLAYER_ADOPTION_ARCHITECTURE_GATE.md) |

## Context

MP-6 closed enterprise certification for Collaborative Activity & Provenance. MP-7 must prove **LKW** (`local_workspace_application`, Tier-3) as the **first reference consumer** of platform Multiplayer primitives (MP-1…MP-6) without transferring ownership, forking contracts, or duplicating authoritative state.

LKW today owns a **product** Hybrid Knowledge Workspace: managed workspaces, knowledge configuration, sources, Hybrid Ask, Conversation Context, Nexus `Task` orchestration, shadow outputs, and tenant/workspace-scoped authorization for **product** data. At the MP-7A audit HEAD, LKW does **not** import `intergrax.collaborative_work` implementation modules or wire public Multiplayer contracts.

Multiplayer owns collaborative identity, membership, delegation, shared work, collaborative artifacts, decision **bindings**, principal-scoped ContextView, and Collaborative Activity. LKW must not become a second authority (**MP-INV-30**).

LKW **product workspace** is not automatically identical to **Collaborative Work workspace scope**. Silent ID sharing (`lkw_workspace_id` as platform `workspace_id` without decision) is forbidden.

## Alternatives considered

### A. LKW `workspace_id` is the canonical platform Collaborative Workspace identity

Rejected. LKW workspaces carry product lifecycle and knowledge configuration with product-specific authority. Collapsing IDs merges lifecycles incorrectly.

### B. LKW product workspace remains distinct; stores explicit Collaborative Workspace reference (Accepted)

Accepted. LKW keeps product `workspace_id` and stores typed **`collaborative_workspace_ref`** (`tenant_id`, `workspace_id`) for platform scope.

### C. Principal mapping only; no workspace binding

Rejected. Multiplayer scoped mutations require platform workspace scope (**MP-INV-06**).

## Decision

1. **LKW is a reference consumer, never Multiplayer authority.** Adoption does not transfer ownership of MP-1…MP-6 primitives to LKW. Adoption does **not** transfer ownership of MP-1…MP-6 primitives to LKW.

2. **Workspace (Option B):** Product workspace remains LKW-owned. Collaborative workspace scope is platform-owned. Correlate via explicit typed reference, not shared string identity.

3. **Principal:** Typed binding from LKW authenticated actor to platform Principal — no inference from Slack/channel identifiers.

4. **Membership / Delegation:** Platform-owned when adopted. Deferred in MP-7 first subset until multi-principal LKW journeys ship.

5. **WorkItem / Assignment:** Deferred. Nexus `Task` remains execution transport (**MP-INV-07/08**). No `thread_id` / `channel_id` as WorkItem identity.

6. **WorkArtifact:** Deferred. LKW shadow/synthesis outputs are not automatic WorkArtifacts.

7. **Decision binding:** Deferred. HITL remains Governance/Nexus (**MP-INV-09**).

8. **ContextView vs Conversation Context:** LKW Conversation Context is product-owned durable thread memory. ContextView is platform-owned principal-scoped composition — related only via future adapter/composition, never silent equality.

9. **Collaborative Activity:** Deferred read adoption; LKW never appends to activity stores (**MP-INV-25/26**).

10. **Contracts only in domain/application:** `intergrax/contracts/collaborative_*`; providers and `intergrax.collaborative_work.*` repositories only in Tier-3 composition (GR-6 class boundary).

11. **MP-7 first subset:** MP-7B — Principal binding, collaborative workspace reference, enforcement wiring. Remaining primitives per [`MP-7A_LKW_MULTIPLAYER_ADOPTION_ARCHITECTURE_GATE.md`](../../../../maintainers/qualification/MP-7A_LKW_MULTIPLAYER_ADOPTION_ARCHITECTURE_GATE.md).

12. **Migration:** Strangler; no MP-7A data migration; dual-write forbidden by default; LKW orchestrates backfill only in application boundary when scheduled.

13. **Product ownership retained:** workflows, knowledge configuration, Hybrid Ask, Conversation Context, frontends, source lifecycle UX — not platform primitives.

14. **MP-7 parallel track** does not override direct LKW Product 1.0 rows without operator decision.

## Consequences

### Positive

- Unambiguous workspace and identity boundaries before MP-7B code.
- Minimal first subset aligned with real LKW Product Alpha journeys.

### Negative

- Operators manage two workspace identifiers during adoption.
- Composition wiring work in MP-7B+.

## Compliance

- **MP-INV-30**, tier boundaries, Vendor Knowledge neutrality, no LKW fields in platform DTOs.

## Implementation notes

- MP-7A: docs/gates only.
- Independent audit required before MP-7B implementation.
