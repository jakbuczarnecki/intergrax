# Collaborative Work

**Status:** Canonical architecture (domain pair 1:1) - **MP-1 — CLOSED / FINAL INDEPENDENT REVIEW PASS**
**Plan (1:1):** [`plan/COLLABORATIVE_WORK.md`](../maintainers/plans/COLLABORATIVE_WORK.md)
**Feature coordination:** [`capabilities/architecture/MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md)
**Architecture governance:** [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
**ADR:** [ADR-MP-001](../technical/adr/entries/2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../technical/adr/entries/2026-08-11/ADR-MP-002.md) · [ADR-MP-003](../technical/adr/entries/2026-09-06/ADR-MP-003.md)

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
  `MeaningfulSideEffectRequest` - **enforcement** of resolved authority at execution boundaries.
- incidental `principal_id` / `tenant_id` fields on execution/policy contracts.

### Unified Execution Runtime owns

- `Task`, `run_id`, `RuntimeEvent`, UAEP execution lifecycle,
- `RequestIdentity` / `PrincipalType` for **run-scoped** authenticated intake,
- `DelegationSpec` for **Nexus graph child-run** delegation (execution, not authority).

### Application Hosting owns

- `HostedApplicationProfile`, process lifecycle, supervision, OS adapters - not collaborative identity.

### Nexus / Orchestration owns

- execution orchestration, graph delegation to child agents, task lifecycle - not WorkItem lifecycle.

### Memory / UCL / Context Engineering own

- context assembly, memory namespaces, optimization artifacts - not membership or delegation source of truth.

### Evidence / Observability own

- proof receipts, trace linkage, event spine - consume resolved principal context; do not define membership.

### Tier-3 applications (e.g. LKW) own

- product adoption and consumer integration - **not** platform primitive ownership.

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
- **CW-INV-14:** Final execution ALLOW requires every applicable mandatory policy layer to return ALLOW; composition is fail closed and never weakens a restrictive decision.
- **CW-INV-15:** Missing or unavailable mandatory policy evaluation is DENY - never implicit ALLOW.
- **CW-INV-16:** Within `tenant_id + workspace_id`, each `principal_id` has at most one authoritative `WorkspaceMembership`; `membership_id` is immutable record identity, not a duplicate-membership selector.
- **CW-INV-17:** Delegated authority requires both delegate and delegator to hold active workspace membership; revoked/suspended/missing delegator membership fails closed.
- **CW-INV-18:** Authoritative Collaborative Work production paths must not use dynamic attribute access (`getattr`, `setattr`, `hasattr`, `vars`, `object.__setattr__`, direct `.__dict__`).
- **CW-INV-19:** Production durable backend - production multi-instance Collaborative Work deployments require a repository backend proven for cross-process transactional concurrency. PostgreSQL is the first production-qualified adapter; SQLite remains a lightweight/local durable adapter. Future production-qualified adapters may implement the same port after equivalent qualification.

### Policy composition boundary (COLLAB-WORK-1E)

Collaborative Work owns the neutral composition boundary that combines pre-evaluated layer decisions:

    collaborative authority ∩ workspace policy ∩ resource policy ∩ runtime/tool policy
    → final ``PolicyDecision``

- Runtime/tool meaningful-side-effect evaluation remains owned by Runtime Policy (`RuntimePolicyEngine` / `PolicyEngine`).
- Workspace and resource policy evaluators are not fabricated in this slice; absent canonical evaluators, composition reports missing mandatory decisions as DENY.
- Layer applicability uses typed `PolicyLayerApplicability` (`REQUIRED`, `NOT_APPLICABLE`, `UNKNOWN`); default `UNKNOWN` fails closed. Only trusted `NOT_APPLICABLE` from future operation classification may skip a layer.
- `compose_policy_decisions` retains contributing layer provenance in `audit_payload` for auditability.

### Workspace and resource policy source (COLLAB-WORK-1F)

Collaborative Work owns authoritative workspace and resource policy persistence and evaluation:

    exact policy key → ``CollaborativePolicyRule`` → ``PolicyDecision``

- **Exact policy keys** (at most one canonical rule each):
  - workspace: `tenant_id + workspace_id + authority_scope`
  - resource: `tenant_id + workspace_id + resource_scope + authority_scope`
- **Matching:** exact normalized strings only - no wildcards, inheritance, or hierarchy.
- **Fail closed:** missing or inactive (`DISABLED`) rules yield DENY; no implicit ALLOW.
- **Resource evaluator** does not fall back to workspace rules; composition combines layers.
- **Output:** existing ``PolicyDecision`` consumable by ``compose_policy_decisions``; no fabricated bundle attestation.
- **Runtime Policy ownership unchanged** - runtime/tool evaluation remains in ``RuntimePolicyEngine`` / ``PolicyEngine``.
- **Applicability classification remains separate** - evaluators answer only when explicitly asked; they do not emit ``NOT_APPLICABLE``.
- **Policy management authorization is out of scope** - creating/updating rules is highly privileged and must itself be authority/policy gated in future administration.

### Trusted operation classification and enforcement gate (COLLAB-WORK-1G)

Collaborative Work owns authoritative operation → policy-layer classification and the reusable final enforcement gate:

    operation_id → ``CollaborativeOperationPolicyProfile`` → authority + workspace + resource + runtime evaluation → ``compose_policy_decisions``

- **Applicability source is authoritative** - operation profiles declare ``REQUIRED`` / ``NOT_APPLICABLE`` per layer; callers must not supply ``PolicyCompositionApplicability`` or skip flags.
- **Profile binds authority scope** - enforcement uses profile-owned ``authority_scope``; caller cannot substitute a weaker scope.
- **Meaningful side-effect requirement forces runtime policy** - contradictory profiles are rejected at contract validation.
- **Gate orchestrates existing owners** - ``CollaborativeWorkAuthorityResolver``, ``CollaborativePolicyEvaluator``, and ``RuntimePolicyEngine`` / ``PolicyEngine`` meaningful-side-effect path; no duplicated composition or runtime evaluator.
- **Missing or inactive profile fails closed** - classification unresolved yields DENY; no operation executes inside the gate.

### Durable authoritative state and production adoption (COLLAB-WORK-1H / COLLAB-WORK-1J)

Collaborative Work owns durable persistence for MP-1 authoritative security/configuration state behind existing repository ports:

```text
Repository Ports
├── InMemory - reference
├── SQLite - local/dev durable
└── PostgreSQL - production scalable durable
```

    repository ports → durable adapter → configured database (composition root)

- **Vendor-neutral domain** - contracts and enforcement gate import no database or observability vendor SDKs; concrete storage is selected at composition root (`open_sqlite_collaborative_work_repositories`, `open_postgresql_collaborative_work_repositories`).
- **Semantic parity** - durable adapters preserve tenant/workspace isolation, revision-0 create, ``expected_revision`` CAS, idempotency replay snapshots, and database-enforced uniqueness for membership (including one membership per principal per workspace), delegation, principal authority, policy exact keys, operation profiles, and idempotency scope/key.
- **Fail closed** - production must not silently fall back to in-memory authority state when durable storage is configured but unavailable.
- **Canonical side-effect boundary** - ``MeaningfulSideEffectAuthorizationBoundary`` invokes ``CollaborativeWorkEnforcementGate`` immediately before a meaningful side effect may proceed; only ``ALLOW`` permits continuation; ``REQUIRE_HUMAN`` / ``ESCALATE`` return upstream without execution.
- **Semantic channel separation** - domain authoritative state ≠ product activity history ≠ audit/evidence ≠ technical logs ≠ error reporting ≠ distributed traces ≠ metrics. None of the observability channels may become authority source-of-truth.
- **Platform modularity** - Multiplayer capabilities depend on platform-level ports/contracts; concrete vendors (persistence, messaging, logs, errors, traces, metrics, activity, audit) are selected by configuration/adapters outside canonical Collaborative Work contracts.

---

## MP-1 contract direction

MP-1 freezes semantic contracts only (see ADR-MP-002):

| Contract | Direction |
|----------|-----------|
| **Principal** | `HUMAN`, `AGENT`, `SERVICE`, future `EXTERNAL_AGENT` - semantic kinds; implementation enum location remains open until justified |
| **WorkspaceMembership** | explicit membership in `tenant_id + workspace_id` scope; role is collaborative classification only - not an authority source |
| **PrincipalAuthorityGrant** | explicit authoritative base-authority scopes per principal in workspace scope; one grant per principal per workspace |
| **Delegation** | delegator + delegate principals; scoped authority; optional resource/time bounds; never amplifies delegator base authority |
| **Effective authority** | base principal authority ∩ membership ∩ delegation ∩ workspace policy ∩ resource policy ∩ runtime/tool policy |

Persistence, APIs, repositories, and enforcement implementation are delivered for MP-1 core. LKW/application adoption (MP-7) remains out of scope until its bounded gate opens.

**MP-2 status:** **IMPLEMENTATION IN PROGRESS** — ADR-MP-003 Accepted; COLLAB-WORK-2A **APPROVED / CLOSED**; COLLAB-WORK-2B **APPROVED / CLOSED**; COLLAB-WORK-2C **APPROVED / CLOSED**.
**Current active task:** **COLLAB-WORK-2D** (SQLite durability parity).

---

## MP-2 — Shared Work (architecture frozen)

**Owning domain:** Collaborative Work (this hub). **ADR:** [ADR-MP-003](../technical/adr/entries/2026-09-06/ADR-MP-003.md).

### Ownership

Collaborative Work owns MP-2 Shared Work:

- WorkItem identity and **WorkItemState** lifecycle,
- **Assignment** identity and assignment lifecycle,
- collaborative ownership / assignment semantics (multi-principal),
- collaborative optimistic concurrency and idempotency,
- work-level `tenant_id` + `workspace_id` isolation,
- work-level authority requirements (via MP-1 enforcement),
- zero..N **execution links** to Nexus/UER identities.

Collaborative Work does **not** own: Nexus Task lifecycle, run/attempt lifecycle, execution scheduling/retries, workflow graph execution, worker/process scheduling, or background task runtime ownership.

**Reused (non-owners):** ORCHESTRATION (graph policy; may consume WorkItem context), UNIFIED_EXECUTION_RUNTIME / NEXUS (`Task`, `run_id`, `attempt`, outcomes), BACKGROUND_TASKS (may execute work associated with a WorkItem), OBSERVABILITY / PROOF_RECEIPTS (provenance consumption).

### WorkItem != Nexus Task

| WorkItem | Nexus Task |
|----------|------------|
| Durable collaborative work | Execution unit |
| Independently addressable | Runtime lifecycle |
| May exist with zero active executions | Created to advance work |
| May span multiple tasks/runs | Does not own collaborative truth |
| May survive execution completion | Belongs to execution plane |
| Multi-principal assignments | Not 1:1 with WorkItem |

No subclassing Task, renaming Task to WorkItem, or wrapper-as-source-of-truth. **WorkItemState != TaskState** — state must not be inferred solely from TaskState.

### Assignment != AgentAssignment

**Assignment** is a separate collaborative primitive (`work_item_id`, `principal_id`, assignment state, revision, authority/lifecycle provenance). Supports human↔human, human↔agent, agent↔agent, and service/external-agent principals through MP-1 `CollaborativePrincipal`. **`principal_id` is immutable** on an Assignment record — reassignment is represented by lifecycle termination of the old Assignment and creation of a new Assignment, not by mutating `principal_id` in place.

Do **not** encode assignments as a single `WorkItem.assignee_id` when multi-principal or assignment history is required. **Assignment != AgentAssignment** when the latter denotes runtime/agent execution assignment elsewhere in canon.

**Reassignment semantics (MP-2 / COLLAB-WORK-2C):** `reassign` = revoke existing Assignment + create a new Assignment — two independently authorized, CAS-protected repository mutations. COLLAB-WORK-2B repositories expose no transactional Unit of Work; MP-2 must not expose a combined atomic reassignment command or simulate rollback across records. Atomic multi-record orchestration requires an explicit transactional boundary (future concern, not MP-2 scope).

### Execution linkage

```text
WorkItem → zero..N execution links
  → optional task_id, run_id, attempt_id (provenance references)
```

Deleting or ending a run must not delete WorkItem. Any orchestration bridge must be explicit — no incidental workflow status propagation into WorkItemState.

### Contract direction (semantic categories only)

**WorkItem:** `work_item_id`, `tenant_id`, `workspace_id`, `WorkItemState`, `created_by_principal_id`; optional title/description or canonical payload reference; `revision`; `created_at` / `updated_at`. No `dict[str, Any]` metadata core; no channel/thread IDs as canonical identity.

**Assignment:** separate from WorkItem body; typed assignment state; revision; provenance for create/reassign/revoke.

### Lifecycle

Collaborative lifecycle only — explicit, validated, deterministic transitions; authority checked; auditable; optimistic-concurrency protected. No approval workflow encoded in WorkItem state (MP-4). No artifact bodies on WorkItem (MP-3). Stable identity/revision for future MP-6 Activity projection without implementing activity feeds in MP-2.

### Concurrency and idempotency

Reuse MP-1 repository semantics: revision 0 create, `expected_revision` CAS, typed conflict, deterministic idempotency replay for WorkItem create, Assignment create, and state transitions subject to external retry. No silent last-write-wins.

### Persistence direction

Authoritative WorkItem and Assignment state uses Collaborative Work repository ports → in-memory reference → SQLite (local/dev) → production-qualified relational adapter (PostgreSQL first). No separate SharedWork database subsystem. Storage selection remains composition-root concern — no provider string switches in core contracts.

### Authority reuse

Mutations (create WorkItem, assign, WorkItem/Assignment state transitions, close/reopen, cancel; reassignment via revoke + create-new Assignment) pass through MP-1 effective authority and policy composition. MP-2 defines work resource semantics; MP-1 owns collaborative authority foundation. No separate WorkItem ACL engine.

### Extension boundaries (MP-3+)

| Phase | Boundary |
|-------|----------|
| MP-3 | WorkArtifact / WorkArtifactVersion — not WorkItem payload |
| MP-4 | Decision / Approval — distinct primitive; not WorkItem state machine |
| MP-6 | Activity projection — hooks via stable identity/revision only |
| MP-7 | LKW/channel IDs — adapter reference mappings only |

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
