# Collaborative Work

**Status:** Canonical architecture (domain pair 1:1) — **MP-1 ownership frozen; runtime implementation NOT STARTED**
**Plan (1:1):** [`plan/COLLABORATIVE_WORK.md`](../maintainers/plans/COLLABORATIVE_WORK.md)
**Feature coordination:** [`capabilities/architecture/MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md)
**Architecture governance:** [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
**ADR:** [ADR-MP-001](../technical/adr/entries/2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../technical/adr/entries/2026-08-11/ADR-MP-002.md)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** §Purpose, §Ownership boundary, §Normative invariants, §Integration boundaries.
- **MP-1 implementation:** this read-scope block + [`plan/COLLABORATIVE_WORK.md`](../maintainers/plans/COLLABORATIVE_WORK.md) active `COLLAB-WORK-1*` row only.
- **Cross-feature context:** [`capabilities/architecture/MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md) active `MP-*` section only.

---

## Purpose

Collaborative Work is the platform domain that owns **who collaborates**, **in which workspace**, **with what membership and delegation**, and **what effective authority applies** when principals act on shared work.

```text
Principal
  → WorkspaceMembership
  → Delegation (authority, not execution graph)
  → Effective authority resolution
        │
        ├── scopes Shared Work (MP-2)
        ├── scopes WorkArtifact authority (MP-3)
        ├── scopes Decision authority (MP-4)
        ├── scopes Principal-scoped ContextView (MP-5)
        └── scopes Collaborative Activity semantics (MP-6)
```

The domain answers:

> Who is a collaborative actor, who belongs to a workspace, who may act for whom, and what authority intersection authorizes a collaborative mutation?

It does not answer:

- how Nexus executes tasks or runs,
- how applications are hosted as processes,
- how policy rules are authored,
- how memory stores conversational content,
- how HITL pauses execution,
- how LKW product workflows are composed.

---

## Ownership boundary

### Collaborative Work owns

- collaborative **Principal** semantics and canonical identity contracts,
- **WorkspaceMembership** (explicit membership; never inferred from IDs alone),
- **Delegation** of authority between principals (scoped; non-amplifying),
- **effective authority** composition semantics,
- future MP-2…MP-6 collaborative primitives that extend the same work plane:
  WorkItem, Assignment, WorkArtifact, Decision, Activity collaborative semantics.

### Policy / runtime enforcement owns (reuse, not storage)

- `PolicyEngine`, `ToolAccessPolicy`, `evaluate_meaningful_side_effect`,
  `MeaningfulSideEffectRequest` — **enforcement** of resolved authority at execution boundaries.
- incidental `principal_id` / `tenant_id` fields on execution/policy contracts.

### Unified Execution Runtime owns

- `Task`, `run_id`, `RuntimeEvent`, UAEP execution lifecycle,
- `RequestIdentity` / `PrincipalType` for **run-scoped** authenticated intake,
- `DelegationSpec` for **Nexus graph child-run** delegation (execution, not authority).

### Application Hosting owns

- `HostedApplicationProfile`, process lifecycle, supervision, OS adapters — not collaborative identity.

### Nexus / Orchestration owns

- execution orchestration, graph delegation to child agents, task lifecycle — not WorkItem lifecycle.

### Memory / UCL / Context Engineering own

- context assembly, memory namespaces, optimization artifacts — not membership or delegation source of truth.

### Evidence / Observability own

- proof receipts, trace linkage, event spine — consume resolved principal context; do not define membership.

### Tier-3 applications (e.g. LKW) own

- product adoption and consumer integration — **not** platform primitive ownership.

---

## Normative invariants

- **CW-INV-01:** Every reusable collaborative identity primitive is platform-owned under this domain.
- **CW-INV-02:** Membership is explicit; `tenant_id` and `workspace_id` alone do not authorize.
- **CW-INV-03:** Delegation never amplifies delegator base authority.
- **CW-INV-04:** Agent collaborative Principal remains distinct from delegating human Principal.
- **CW-INV-05:** `Principal != AgentDefinition != AgentRun != RequestIdentity`.
- **CW-INV-06:** `WorkItem != Nexus Task`; collaborative work plane != execution plane.
- **CW-INV-07:** `Decision != HITL`; collaborative decision semantics != execution pause runtime.
- **CW-INV-08:** Memory is not the source of truth for Membership or Delegation.
- **CW-INV-09:** Policy evaluates effective authority; this domain owns the semantic source of truth.
- **CW-INV-10:** Fail closed when required authority cannot be proven for privileged mutations.
- **CW-INV-11:** LKW and other applications consume contracts; no `LkwPrincipal`, `LkwWorkspaceMember`, or `LkwDelegation` ownership.
- **CW-INV-12:** `WorkspaceMembershipRole` is collaborative classification; explicit `PrincipalAuthorityGrant.authority_scopes` own base authority.
- **CW-INV-13:** Collaborative Work ALLOW satisfies only the collaborative authority slice; workspace, resource, and runtime/tool policy remain required for execution authorization.

---

## MP-1 contract direction

MP-1 freezes semantic contracts only (see ADR-MP-002):

| Contract | Direction |
|----------|-----------|
| **Principal** | `HUMAN`, `AGENT`, `SERVICE`, future `EXTERNAL_AGENT` — semantic kinds; implementation enum location remains open until justified |
| **WorkspaceMembership** | explicit membership in `tenant_id + workspace_id` scope; role is collaborative classification only — not an authority source |
| **PrincipalAuthorityGrant** | explicit authoritative base-authority scopes per principal in workspace scope; one grant per principal per workspace |
| **Delegation** | delegator + delegate principals; scoped authority; optional resource/time bounds; never amplifies delegator base authority |
| **Effective authority** | base principal authority ∩ membership ∩ delegation ∩ workspace policy ∩ resource policy ∩ runtime/tool policy |

Persistence, APIs, repositories, and enforcement implementation remain **out of scope** until the MP-1 implementation gate opens.

---

## Future extension boundary (MP-2…MP-6)

Future Multiplayer phases that belong on the collaborative work plane extend **this domain**, governed by their respective ADR/MP gates:

| Phase | Expected extension |
|-------|-------------------|
| MP-2 | WorkItem, Assignment, shared-work lifecycle |
| MP-3 | WorkArtifact, WorkArtifactVersion collaborative ownership |
| MP-4 | Decision / DecisionResponse collaborative semantics |
| MP-5 | Principal-scoped ContextView boundary (composition with UCL/Memory) |
| MP-6 | Collaborative Activity + provenance linkage |

Architecture and implementation rows for MP-2+ remain in their future gates; this hub establishes the plane boundary only.

---

## Integration boundaries

```text
Collaborative Work (identity + authority semantics)
  → Policy / Runtime (enforcement at mutation boundaries)
  → Nexus (execution under resolved authority)
  → UCL / Context Engineering / Memory (context composition)
  → Evidence / Observability (proof and trace linkage)
  → Application Hosting (host lifecycle only)
  → LKW (first consumer)
```

---

## Related documents

| Document | Role |
|----------|------|
| [`MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md) | Multi-layer feature coordination |
| [`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md) | Hosting boundary |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | Execution boundary |
| [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) | PLATFORM-INV-001…003 |
