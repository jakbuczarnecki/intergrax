# Collaborative Work

**Status:** Canonical architecture (domain pair 1:1) - **MP-1 — CLOSED / FINAL INDEPENDENT REVIEW PASS**
**Plan (1:1):** [`plan/COLLABORATIVE_WORK.md`](../maintainers/plans/COLLABORATIVE_WORK.md)
**Feature coordination:** [`capabilities/architecture/MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md)
**Architecture governance:** [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
**ADR:** [ADR-MP-001](../technical/adr/entries/2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../technical/adr/entries/2026-08-11/ADR-MP-002.md) · [ADR-MP-003](../technical/adr/entries/2026-09-06/ADR-MP-003.md) · [ADR-MP-004](../technical/adr/entries/2026-09-07/ADR-MP-004.md) · MP-4 → [DECISION_APPROVAL_GOVERNANCE](DECISION_APPROVAL_GOVERNANCE.md) / [ADR-MP-005](../technical/adr/entries/2026-09-08/ADR-MP-005.md) · [ADR-MP-006](../technical/adr/entries/2026-09-17/ADR-MP-006.md) (MP-5 ContextView) · [ADR-MP-007](../technical/adr/entries/2026-09-18/ADR-MP-007.md) (MP-6 Activity)

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
- collaborative primitives on the same work plane: WorkItem, Assignment, WorkArtifact (MP-2…MP-3 **CLOSED**),
  **Principal-scoped ContextView** policy and composition semantics (MP-5 — **ownership FROZEN**, ADR-MP-006),
  **Collaborative Activity & Provenance** semantics (MP-6 — **ownership FROZEN**, ADR-MP-007).
- Collaborative Work does **not** own canonical Decision authority (MP-4 reuse only).

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

Persistence, APIs, repositories, and enforcement implementation are delivered for MP-1 core.

**Current Multiplayer core status:** MP-1…MP-7 closed in their certified scopes (capability SSOT: [`MULTIPLAYER_AI` Current Enterprise Maturity Boundary](../capabilities/architecture/MULTIPLAYER_AI.md#current-enterprise-maturity-boundary)).

**MP-2 status:** **APPROVED / CLOSED** — ADR-MP-003 **Accepted; implementation COMPLETE**; COLLAB-WORK-2A…2G **APPROVED / CLOSED**.
**MP-3 — ENTERPRISE CERTIFIED / CLOSED** — ADR-MP-004 **Accepted**; **architecture decomposition — APPROVED / CLOSED**; slices **MP-3A…MP-3H — APPROVED / CLOSED** (MP-3H final cross-slice certification).
**MP-5 — ENTERPRISE CERTIFIED / CLOSED** · **MP-6 — ENTERPRISE CERTIFIED / CLOSED** · **MP-7 — ENTERPRISE BOUNDARY CERTIFIED / CLOSED** (Multiplayer Tier-3 consumability boundary certified — not LKW Multiplayer adoption complete).
**Next Multiplayer capability phase:** **MP-8 — PLANNED / NOT STARTED** (AgentDirectory / external agents). Final hardening before MP-8: MP-FINAL-2…MP-FINAL-4 per capability plan.

### MP-2 final closure summary (COLLAB-WORK-2G)

MP-2 delivered: WorkItem; explicit collaborative lifecycle (`OPEN`/`ACTIVE`/`COMPLETED`/`CANCELLED`); Assignment; multi-principal participation; MP-1 authority enforcement reuse; optimistic concurrency (`revision` from `0`, expected-revision CAS); idempotency; in-memory reference persistence; SQLite durable persistence (**QUALIFIED** `cw.sqlite.repository.v1` v`3.0.0`); PostgreSQL production-qualified persistence (**PRODUCTION_QUALIFIED** `cw.postgresql.repository.v1` v`3.0.0`); cross-process CAS proof; zero..N Unified Execution provenance links via full `ExecutionProvenanceRef` (`TaskId`/`RunId`/`AttemptId`/`ExecutionId`); no Nexus lifecycle ownership; no `TaskState` substitution.

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
- zero..N **execution links** to canonical Unified Execution identities (`ExecutionProvenanceRef`).
- zero..N **collaborative decision bindings** (`CollaborativeDecisionBinding`) to exact canonical `DecisionProposalRef` (optional `WorkArtifactVersionRef`) — association only; Decision System owns Decision truth.

Collaborative Work does **not** own: Task/Run/Attempt/Execution lifecycle, execution scheduling/retries, workflow graph execution, worker/process scheduling, background task runtime ownership, or Nexus orchestration control-plane internals.

**Reused (non-owners):** ORCHESTRATION (graph policy; may consume WorkItem context), UNIFIED_EXECUTION_RUNTIME (`TaskId`, `RunId`, `AttemptId`, `ExecutionId`, outcomes), NEXUS (internal orchestration consumer/producer of Executions when strategy = orchestration — **not** a Collaborative Work contract dependency), BACKGROUND_TASKS (may execute work associated with a WorkItem), OBSERVABILITY / PROOF_RECEIPTS (provenance consumption).

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

### Unified Execution linkage

```text
WorkItem → zero..N execution links
  → each link references exactly one ExecutionProvenanceRef
```

**ExecutionProvenanceRef** (neutral Tier-0 composite; future location e.g. `intergrax/contracts/execution_provenance.py`):

```text
ExecutionProvenanceRef
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
```

All four IDs are **required** for a WorkItem execution link. UEA canonical hierarchy is `TaskId → RunId → AttemptId → ExecutionId → EventId`; a WorkItem link points to a concrete **Execution**, not an individual event (`EventId` is out of scope). One Attempt may contain multiple Executions; Task/Run/Attempt alone cannot identify one concrete Execution.

**Link semantics (provenance only):**

- no lifecycle substitution; no `TaskState → WorkItemState` mapping; no Run completion → WorkItem completion
- no cascade delete; WorkItem survives execution completion/deletion/archive
- WorkItem may have zero links; may link to many Executions
- multiple Executions may belong to the same Run/Attempt; multiple Tasks/Runs may advance one WorkItem
- no 1:1 WorkItem ↔ Task assumption

**Ownership:** Collaborative Work owns `WorkItemExecutionLink` association identity, association persistence, and WorkItem-side lookup of associations. Unified Execution / UER owns `TaskId`, `RunId`, `AttemptId`, `ExecutionId`, runtime lifecycle, and Execution Tree. Nexus owns orchestration runtime control only (readiness/scheduling/fan-out/fan-in for orchestration strategy) — **not** Collaborative Work contract boundary.

**Dependency direction (production):**

```text
Collaborative Work → neutral contracts → execution_identity / ExecutionProvenanceRef
```

Forbidden production dependencies: `intergrax.runtime.nexus.*`, `NexusLoop`, `GraphExecutor`, `TaskState`, `TaskResult`, `RuntimeExecutionContext` as the stored contract, `EmitContext`, runtime observability implementation types. Runtime/test adapters may extract neutral IDs from runtime objects.

Deleting or ending a run/execution must not delete WorkItem. No incidental workflow status propagation into WorkItemState.

**ADR-MP-003 reconciliation:** ADR-MP-003 accepted execution linkage without freezing concrete persistence schema or `ExecutionId` granularity. Later frozen Unified Execution architecture establishes `ExecutionId` as the concrete runtime execution unit; COLLAB-WORK-2F therefore resolves the previously open execution-link granularity to full `TaskId`/`RunId`/`AttemptId`/`ExecutionId` provenance via neutral `ExecutionProvenanceRef`.

### Contract direction (semantic categories only)

**WorkItem:** `work_item_id`, `tenant_id`, `workspace_id`, `WorkItemState`, `created_by_principal_id`; optional title/description or canonical payload reference; `revision`; `created_at` / `updated_at`. No `dict[str, Any]` metadata core; no channel/thread IDs as canonical identity.

**Assignment:** separate from WorkItem body; typed assignment state; revision; provenance for create/reassign/revoke.

### Lifecycle

Collaborative lifecycle only — explicit, validated, deterministic transitions; authority checked; auditable; optimistic-concurrency protected. No approval workflow encoded in WorkItem state (MP-4). No artifact bodies on WorkItem (MP-3). Stable identity/revision for future MP-6 Activity projection without implementing activity feeds in MP-2.

### Concurrency and idempotency

Reuse MP-1 repository semantics: revision 0 create, `expected_revision` CAS, typed conflict, deterministic idempotency replay for WorkItem create, Assignment create, and state transitions subject to external retry. No silent last-write-wins.

### Persistence direction

Authoritative Shared Work persistence bundle (`WorkItem` + `Assignment` + append-only `WorkItemExecutionLink`) uses Collaborative Work repository ports → in-memory reference → SQLite (local/dev) → production-qualified relational adapter (PostgreSQL first). `WorkItemExecutionLink` records are append-only — no update/delete/revision. No separate SharedWork database subsystem. Storage selection remains composition-root concern — no provider string switches in core contracts.

### Authority reuse

Mutations (create WorkItem, assign, WorkItem/Assignment state transitions, close/reopen, cancel; reassignment via revoke + create-new Assignment) pass through MP-1 effective authority and policy composition. MP-2 defines work resource semantics; MP-1 owns collaborative authority foundation. No separate WorkItem ACL engine.

### Extension boundaries (MP-3+)

| Phase | Boundary |
|-------|----------|
| MP-3 | WorkArtifact / WorkArtifactVersion — not WorkItem payload |
| MP-4 | Decision / Approval / Governance — distinct primitives; not WorkItem or WorkArtifact lifecycle — see [`DECISION_APPROVAL_GOVERNANCE`](DECISION_APPROVAL_GOVERNANCE.md) |
| MP-6 | Activity projection — hooks via stable identity/revision only |
| MP-7 | LKW/channel IDs — adapter reference mappings only |

---

## MP-3 — WorkArtifact / WorkArtifactVersion (architecture frozen)

**Owning domain:** Collaborative Work (this hub). **ADR:** [ADR-MP-004](../technical/adr/entries/2026-09-07/ADR-MP-004.md).

### Ownership

Collaborative Work owns MP-3 WorkArtifact collaborative semantics:

- **WorkArtifact** identity and aggregate/reference lifecycle,
- **WorkArtifactVersion** identity (immutable authoritative collaborative output version),
- artifact/version association and collaborative lineage (artifact-level),
- authoritative **current-version pointer** semantics (`current_version_id` must reference a version in the same tenant, workspace, and work artifact),
- collaborative **publication** semantics (append-only versions; CAS-protected pointer updates),
- tenant/workspace/work-item scoping,
- collaborative authority requirements for publication (MP-1 reuse),
- optimistic concurrency and idempotent publication for authoritative mutations.

Collaborative Work does **not** own: UCL `OptimizationArtifact` catalog, memory/session state, proof/evidence records, execution result objects, WorkItem mutable payload embedding, LKW/channel attachment identity, or raw blob/object storage providers.

**Reused (non-owners):** UNIFIED_CONTEXT_LIFECYCLE (may consume published versions as context input), MEMORY (may index references / retrieve derived content), PROOF_RECEIPTS (may attest production/publication), UNIFIED_EXECUTION_RUNTIME / NEXUS (optional `ExecutionProvenanceRef`), DocumentStore/blob adapters (content storage only), LKW (consumer).

### Hard invariants

| Forbidden substitution | Authoritative owner |
|------------------------|---------------------|
| `WorkArtifact` / `WorkArtifactVersion` | Collaborative Work |
| UCL `OptimizationArtifact` | UCL |
| `ProofReceipt` | Proof Receipts |
| Memory record | Memory |
| Execution result object | UER |
| WorkItem payload field | WorkItem body stays artifact-free |
| LKW message/file/thread ID | Adapter mapping only |

### WorkItem relationship

```text
WorkItem → zero..N WorkArtifact
WorkArtifact → one..N WorkArtifactVersion (immutable)
```

- Each WorkArtifact references its owning WorkItem; scoped to one `tenant_id + workspace_id`.
- WorkItem lifecycle and artifact version lifecycle are **distinct**.
- Completing or cancelling a WorkItem does **not** delete artifact versions.
- Deleting or ending execution does **not** delete artifact versions.

### Aggregate model (frozen)

```text
WorkItem → zero..N WorkArtifact → one..N WorkArtifactVersion (immutable append-only)
```

- **WorkArtifact** is the mutable authoritative aggregate/reference — identity, scope, `current_version_id`, `revision`; does **not** embed version bodies.
- **WorkArtifactVersion** is immutable authoritative history; no mutable `revision` or status machine on versions.
- **WorkItem** does **not** embed WorkArtifact bodies; no cascading lifecycle substitution.

### Creation semantics (frozen)

First `WorkArtifactVersion` is created **atomically** with initial `WorkArtifact` creation — one authoritative domain operation; no persisted authoritative `WorkArtifact` with dangling `current_version_id`. Persistence atomicity is enforced by `ArtifactPublicationRepository.create_artifact_with_initial_version(...)` (MP-3B+); failure exposes neither half as a successful authoritative operation. Same idempotency key + same semantic request → original artifact + initial version; same key + changed semantic intent → typed idempotency conflict; server-generated timestamps do not pollute the semantic fingerprint.

### Publication semantics (frozen)

Domain operation `publish new artifact version`:

1. authorize acting principal (MP-1),
2. verify artifact tenant/workspace/work_item scope,
3. verify expected artifact `revision`,
4. append immutable `WorkArtifactVersion`,
5. move `WorkArtifact.current_version_id` to new version,
6. increment `WorkArtifact.revision`,
7. preserve previous versions,
8. return typed publication result (aggregate + new version — not untyped tuple/dict).

No in-place version modification; no silent LWW; no implicit WorkItem state transition.

### Atomic publication boundary (frozen)

Independent `WorkArtifactRepository` and `WorkArtifactVersionRepository` ports cannot alone guarantee that multi-write authoritative operations succeed or fail together. MP-3B introduces a narrowly scoped **`ArtifactPublicationRepository`** covering both authoritative write paths:

- **`create_artifact_with_initial_version(...)`** — atomically persist `WorkArtifact`, first immutable `WorkArtifactVersion`, `current_version_id` referencing that version, initial `revision`, and idempotency result.
- **`publish_version(...)`** — verify `expected_revision`; append immutable version; CAS `current_version_id`; increment `revision`; preserve history; idempotent replay; exactly one store transaction/critical section.

`WorkArtifactRepository` and `WorkArtifactVersionRepository` remain typed read/direct-persistence ports; **authoritative create/publish commands go through `ArtifactPublicationRepository` only** — the service must not coordinate two repository writes manually. No generic `Repository[T]`; no generic UnitOfWork; no service-level compensating rollback. In-memory: atomic critical section per operation. SQLite/PostgreSQL: single transaction per operation.

### Idempotency and concurrency (frozen)

- **Idempotency:** separate for (A) create artifact + first version and (B) publish subsequent version; fingerprint includes semantic intent, not server-generated timestamps unless caller-supplied and authoritative; same key + changed content → typed conflict.
- **Concurrency:** `WorkArtifact.revision` starts at `0`; publication with `expected_revision = N` succeeds at `N+1`; concurrent publications from same revision → exactly one success, others `WorkArtifactRevisionConflict`. No CAS on immutable versions; no LWW.

### Version authority and immutability

- `WorkArtifactVersion` is **append-only / immutable** — corrections create a new version.
- `WorkArtifact` holds an explicit current-version pointer updated under **expected-revision CAS**; stale updates fail explicitly (no silent last-write-wins).
- Historical versions remain independently addressable; rollback is pointer selection via explicit publication semantics — not in-place rewrite.
- Canonical version identity is `WorkArtifactVersionId`; do not derive authority from human-readable version labels.

### Content vs metadata

Architecture separates (A) collaborative identity + lineage from (B) content representation/storage reference:

```text
WorkArtifactVersion → ArtifactContentRef (neutral typed descriptor) → storage adapter (MP-3F)
```

**`ArtifactContentRef` (MP-3A contract):** provider-neutral typed content descriptor — stable content identity/location semantics, media/content type, integrity digest (algorithm not frozen unless platform canon standardizes one), optional size. **Storage provider identity must not become `WorkArtifactVersion` identity.** MP-3F integrates adapters; MP-3F must not redefine the contract.

Collaborative Work owns metadata and version semantics; raw binary content may live in DocumentStore, blob/object storage, or external references. Content deletion/retention policy is **not** automatically artifact deletion.

### Principal provenance (frozen)

- **Mandatory:** `created_by_principal_id`, `published_by_principal_id` — canonical `Principal` for human, agent, service, external agent; not generic `owner_id`.
- **Optional:** `ExecutionProvenanceRef` at version creation only — human-created versions valid with `execution=None`; no mutable post-publication lineage attachment.

### Evidence direction (frozen)

```text
WorkArtifactVersion ← ProofReceipt / evidence references it
```

Evidence must not become artifact identity; do not import ProofReceipt runtime into Collaborative Work contracts.

### Query, delete, and lifecycle boundaries

- **Reads:** get artifact, get current version, get version by ID, list versions for artifact — no search/discovery/activity feed.
- **Deletes:** MP-3 core does not require hard delete; historical versions remain addressable.
- **Lifecycle:** identity + versions + current pointer + publication only — no approval/review/archive status machine (MP-4).

### Authority (frozen)

Publication and version creation require MP-1 effective authority via `CollaborativeWorkEnforcementGate` — trusted operation IDs for artifact create/publish; resource-scoped under existing Collaborative Work policy composition. No `ArtifactAuthorizationService`; no artifact ACL engine.

### Extension boundaries (MP-4 / MP-6 / LKW)

| Phase | Boundary |
|-------|----------|
| MP-4 | Decision / approval — not encoded in WorkArtifact lifecycle |
| MP-6 | Activity projection — stable IDs/timestamps/refs only; no activity feed in MP-3 |
| MP-7 / LKW | Consumer; channel/file IDs are adapter mappings only |

### Persistence and composition (frozen)

Extend the MP-2 Collaborative Work repository bundle using the same extension pattern as MP-1→MP-2:

```text
CollaborativeWorkRepositoriesWithSharedWork
  → CollaborativeWorkRepositoriesWithArtifacts
      core + shared_work + artifacts
        artifacts: work_artifact, work_artifact_version, artifact_publication
```

MP-2-only compositions remain valid until MP-3 composition gate opens. Reuse MP-1 authority gate, revision/CAS, idempotency, tenant/workspace isolation. **`ArtifactPublicationRepository`** is part of the artifact bundle — not a generic UnitOfWork. No duplicate persistence framework; no speculative `ArtifactPlugin` / hook registries.

**MP-3F ordering:** PostgreSQL qualification (MP-3E) qualifies metadata and neutral `ArtifactContentRef` without requiring live content-provider integration; MP-3F follows.

### Implementation roadmap

Decomposition **APPROVED / CLOSED** — full slice rows in [`plan/COLLABORATIVE_WORK.md`](../maintainers/plans/COLLABORATIVE_WORK.md) § COLLAB-WORK-3. Runtime **ENTERPRISE CERTIFIED / CLOSED** (MP-3H).

| Slice | Scope | Status |
|-------|-------|--------|
| MP-3A | Contracts + invariants + `ArtifactContentRef` | APPROVED / CLOSED |
| MP-3B | Ports + in-memory + `ArtifactPublicationRepository` (atomic initial create + publish) | APPROVED / CLOSED |
| MP-3C | Publication service + MP-1 authority | APPROVED / CLOSED |
| MP-3D | SQLite transactional persistence | APPROVED / CLOSED |
| MP-3E | PostgreSQL + qualification | APPROVED / CLOSED |
| MP-3F | Content storage adapters | APPROVED / CLOSED |
| MP-3G | Execution/evidence integration | APPROVED / CLOSED |
| MP-3H | Final enterprise certification | APPROVED / CLOSED |

### MP-3 final closure summary (COLLAB-WORK-3H)

MP-3 delivered: typed `WorkArtifact` / `WorkArtifactVersion` / `ArtifactContentRef` contracts; single **`ArtifactPublicationRepository`** atomic boundary (`create_artifact_with_initial_version`, `publish_version`); **`CollaborativeWorkArtifactService`** as semantic owner with MP-1 **`CollaborativeWorkEnforcementGate`** reuse; in-memory + SQLite durable persistence; PostgreSQL live-backend qualification path (`repository_qualification_suite` artifact checks + cross-process CAS proof module); provider-neutral **`ArtifactContentStore`**; optional **`ExecutionProvenanceRef`** on publication; evidence **`EvidenceArtifactVersionLink` → `WorkArtifactVersionRef`** (one-way). Cross-slice enterprise certification — not full Multiplayer product E2E.

---

## MP-4 boundary (Collaborative Work — reference only)

**Owning domain:** [`DECISION_APPROVAL_GOVERNANCE`](DECISION_APPROVAL_GOVERNANCE.md) — **ADR-MP-005 Accepted**; MP-4A **APPROVED / CLOSED**.

Collaborative Work **does not own** Decision, Approval, HumanReviewState, or GovernanceStatus. WorkItem, WorkArtifact, and WorkArtifactVersion remain artifact/work primitives only.

**Frozen non-leakage:**

- No `Decision`, `Approval`, or governance fields on WorkItem / WorkArtifact / WorkArtifactVersion.
- No `DRAFT` / `REVIEW` / `APPROVED` / `ARCHIVED` on WorkArtifactVersion — Approval references `WorkArtifactVersionId` externally.
- Execution links remain neutral `ExecutionProvenanceRef` references only.

Full ownership, anti-substitution, and dependency rules: [`DECISION_APPROVAL_GOVERNANCE`](DECISION_APPROVAL_GOVERNANCE.md).

---

## Future extension boundary (MP-2…MP-6)

Future Multiplayer phases that belong on the collaborative work plane extend **this domain**, governed by their respective ADR/MP gates:

| Phase | Expected extension |
|-------|-------------------|
| MP-2 | WorkItem, Assignment, shared-work lifecycle |
| MP-3 | WorkArtifact, WorkArtifactVersion collaborative ownership |
| MP-4 | Decision / Approval / Governance collaborative semantics — [`DECISION_APPROVAL_GOVERNANCE`](DECISION_APPROVAL_GOVERNANCE.md) |
| MP-5 | Principal-scoped ContextView — **ENTERPRISE CERTIFIED / CLOSED** (MP-5A…MP-5H) |
| MP-6 | Collaborative Activity + provenance linkage — **MP-6 — ENTERPRISE CERTIFIED / CLOSED**; **MP-6A…MP-6H — CLOSED / RECERTIFIED or CERTIFIED** |

Architecture and implementation rows for MP-6+ remain in their future gates.

---

## Principal-scoped ContextView (MP-5)

**MP-5 ownership — FROZEN** ([ADR-MP-006](../technical/adr/entries/2026-09-17/ADR-MP-006.md) **Accepted**). **MP-5A — APPROVED / CLOSED**. **MP-5B — APPROVED / CLOSED**. **MP-5C — APPROVED / CLOSED**. **MP-5D — APPROVED / CLOSED**. **MP-5E — APPROVED / CLOSED**. **MP-5F — ENTERPRISE SOURCE INTEGRATION CERTIFIED / CLOSED**. **MP-5G — ENTERPRISE E2E / ISOLATION QUALIFICATION CERTIFIED / CLOSED**. **MP-5H — CLOSED / FINAL CERTIFICATION PASSED**. **MP-5 — ENTERPRISE CERTIFIED / CLOSED**.

**Capability statement:** ContextView is a principal-scoped, reference-first, policy-governed projection over source-domain canonical references. MP-5 does **not** hydrate payloads, manage storage, run RAG ranking, own UCL lifecycle, select CW current version, or assemble CE tokens.

**MP-5G qualification:** integrates the real canonical principal identity → effective authority → MP-5C visibility policy → MP-5E composer → MP-5D ports → MP-5F adapters → public Memory / Knowledge / UCL / Collaborative Work read boundaries → final `ContextView` pipeline (`tests/unit/collaborative_work/test_mp5g_context_view_e2e_qualification.py`). **Qualified dimensions:** tenant, workspace, work item, operation/resource, principal identity, category eligibility. **MP-5G does not qualify** CE token assembly, Nexus orchestration, or application UX. **Pluginability:** replaceable MP-5D port and replaceable source `*ReferenceReadPort` (harness + qualification tests).

Collaborative Work owns **who may see which context categories under which collaborative scope** — not how Memory stores data, how RAG retrieves, how UCL persists revisions, or how Context Engineering budgets tokens.

**Public contracts (MP-5B):** [`intergrax/contracts/context_view.py`](../../../intergrax/contracts/context_view.py) — `ContextViewRequest`, `ContextViewScope`, `ContextViewEntry` + typed `ContextViewEntrySourceRef` variants, immutable `ContextView` result; MP-1 `EffectiveAuthorityRequest` linkage only (no duplicate authority model).

**Visibility policy (MP-5C):** [`intergrax/contracts/context_view_visibility_policy.py`](../../../intergrax/contracts/context_view_visibility_policy.py) — `ContextViewVisibilityPolicy`, `ContextViewVisibilityPolicyInput`, immutable `ContextViewPolicyDecision`; `ContextViewVisibilityEvaluator` enforces platform `collaborative_work.context_view.read` via `CONTEXT_VIEW_READ_AUTHORITY_SCOPE` before any injected policy; replaceable strategies own category / visibility-class eligibility only. Default implementation [`intergrax/collaborative_work/context_view_visibility.py`](../../../intergrax/collaborative_work/context_view_visibility.py) reuses MP-1 `CollaborativeWorkAuthorityResolver` (no retrieval / composition).

**Source composition ports (MP-5D):** [`intergrax/contracts/context_view_source_ports.py`](../../../intergrax/contracts/context_view_source_ports.py) — consumer-owned `MemoryContextSourcePort`, `KnowledgeContextSourcePort`, `UclContextSourcePort`, `CollaborativeWorkContextSourcePort`; typed per-domain requests/results and reference-first `ContextViewSourceCandidate` variants reusing MP-5B `ContextViewEntrySourceRef` locators (no retrieval, hydration, or adapters).

**Default composition (MP-5E):** [`intergrax/contracts/context_view_composition.py`](../../../intergrax/contracts/context_view_composition.py) — `ContextViewComposer`, `ContextViewCompositionRequest`, replaceable ordering/identity strategies; default [`intergrax/collaborative_work/context_view_composition.py`](../../../intergrax/collaborative_work/context_view_composition.py) (`DefaultContextViewComposer`) consumes an approved `ContextViewPolicyDecision`, invokes injected MP-5D ports for eligible categories only, validates every candidate, dedupes/orders deterministically, and materializes reference-first `ContextView` with `effective_scope` (least-context).

**Reference-read boundary (MP-5F-B4):** Collaborative Work owns WorkItem, WorkArtifact and WorkArtifactVersion identity, lifecycle and canonical ownership relationships. The public reference-read boundary exposes scoped canonical references only (`CollaborativeWorkReferenceReadPort` in [`intergrax/collaborative_work/contracts/collaborative_work_reference_read.py`](../../../intergrax/collaborative_work/contracts/collaborative_work_reference_read.py)); default scoped catalog projection (no payload hydration): [`DefaultCollaborativeWorkReferenceReader`](../../../intergrax/collaborative_work/default_collaborative_work_reference_reader.py) over [`CollaborativeWorkScopedReferenceCatalog`](../../../intergrax/collaborative_work/repository.py). MP-5 adapters consume those references but do not own Collaborative Work semantics. **Anti-substitution:** `ContextView ≠ WorkItem`; `ContextView ≠ WorkArtifact`; MP-5 does not choose WorkArtifact current version — CW aggregate `current_version_id` is authoritative.

**Source adapters (MP-5F-B5):** replaceable MP-5D ports in [`intergrax/collaborative_work/context_view_source_adapters.py`](../../../intergrax/collaborative_work/context_view_source_adapters.py) translate only — source domains remain semantic owners. Composition root: [`wire_default_context_view_composer`](../../../intergrax/collaborative_work/context_view_source_wiring.py) (explicit DI: source read ports → default adapters → `DefaultContextViewComposer`). **Pluginability:** replace the MP-5D port implementation and/or the source-domain `*ReferenceReadPort` implementation independently. **MP-5F-B5-C1 invariants:** ContextView source adapters receive and preserve canonical `RequestIdentity` on MP-5D source requests — they never infer or fabricate `PrincipalType`. `candidate_scope` is derived only from authoritative source-domain `evaluated_scope` / ref provenance, never from unverified request fields. Custom `*ReferenceReadPort` implementations must return scope-consistent typed results; adapters validate before projection.

**MP-5G-C1 scope compatibility:** ContextView never projects source-unproven scope dimensions onto a candidate or entry. Scope compatibility semantics are replaceable through [`ContextViewScopeCompatibilityPolicy`](../../../intergrax/contracts/context_view_scope_compatibility.py); `DefaultContextViewScopeCompatibilityPolicy` implements Model B. `DefaultContextViewComposer` injects the policy contract (constructor DI; optional override at `wire_default_context_view_composer`) and does not own category-specific compatibility rules. **Collaborative Work** candidates still require exact proven work-item and operation dimensions. **Memory**, **Knowledge**, and **UCL** candidates may omit `work_item_id` when their domain did not prove work-item ownership; **Memory** also omits `operation_scope` unless Memory authority proves it. View `effective_scope` may remain work-item- and operation-scoped for admission while per-entry `entry_scope` stays source-truthful.

```text
ContextViewRequest
  → MP-1 effective authority (`CollaborativeWorkAuthorityResolver`)
  → ContextViewVisibilityPolicy
  → ContextViewPolicyDecision
  → DefaultContextViewComposer (MP-5E)
      → injected MP-5D consumer-owned source port (replaceable adapter)
      → source-domain public *ReferenceReadPort → canonical refs (reference-only)
      → truthful source candidate
      → ContextViewScopeCompatibilityPolicy (admission)
      → MP-5D structural isolation validators (category, tenant, visibility, ref integrity)
      → dedupe / order / entry + view identity strategies
  → ContextView
```

**ContextView** is principal-scoped, policy-governed, composed, read-oriented, and auditable via reused provenance contracts. **`ContextView ≠ storage`**.

**Anti-substitution:** `UCL ≠ Principal-scoped ContextView`; `Memory ≠ Principal-scoped ContextView`; `RAG result ≠ Principal-scoped ContextView`; `Context Engineering ≠ Principal-scoped ContextView`; `SharedContextView` / `DecisionContextView` / LKW conversation context ≠ platform Principal-scoped ContextView.

**Authority:** resolve membership and effective delegation before source retrieval; fail-closed when principal, scope, or policy cannot be established. **Least-context** for operations and external-agent projections (MP-8-ready seam).

Capability coordination: [`MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md) § MP-5.

---

## Collaborative Activity & Provenance (MP-6)

**MP-6 ownership — FROZEN** ([ADR-MP-007](../technical/adr/entries/2026-09-18/ADR-MP-007.md) **Accepted**). **MP-6 — ENTERPRISE CERTIFIED / CLOSED** (**MP-6H — CLOSED / CERTIFIED** — [`MP-6_FINAL_ENTERPRISE_CERTIFICATION.md`](../maintainers/qualification/MP-6_FINAL_ENTERPRISE_CERTIFICATION.md); subject to independent audit). **MP-6A — CLOSED / RECERTIFIED** (**MP-6A-C1** / **MP-6A-C1-R1** atomic append ownership). **MP-6B — CLOSED / RECERTIFIED** (**MP-6B-C1** / **MP-6B-C1-R1** policy-resolved append intent; subject to independent audit). **MP-6C — CLOSED / RECERTIFIED** (trusted publisher context, `CollaborativeActivityIngestionService`, injectable ingestion policy, append intent only after admission; subject to independent audit). **MP-6D — CLOSED / RECERTIFIED** (atomic append store: SQLite provider qualified; PostgreSQL provider qualified on live integration DB; `append_idempotent(intent)` owns idempotency, per-workspace `append_position`, `recorded_at`, and durable materialization from `effective_durability_class`; replay immutable). **MP-6E — CLOSED / RECERTIFIED** (scoped read: `CollaborativeActivityReadService` + injectable `CollaborativeActivityReadAuthorizationPolicy`; authority before provider; tenant/workspace isolation; opaque scope-bound cursor; keyset pagination by `append_position`; no authorization inside store; SQLite qualified; PostgreSQL qualified on live integration DB when configured).

**MP-6C ingestion boundary (closed / recertified — MP-6C-C1 / MP-6C-C1-R1):** publisher context is derived from a verified platform identity (`VerifiedCollaborativeActivityPublisherIdentity` from canonical `RequestIdentity` at the authenticated composition boundary) by an injected `CollaborativeActivityPublisherContextResolver` — not from caller-provided tenant/kind/namespace/workspace claims. **Publisher authority is explicit registration** (`CollaborativeActivityPublisherRegistration` via replaceable `CollaborativeActivityPublisherAuthoritySource` config snapshot): PLATFORM and PLUGIN require a per-`(tenant_id, producer_principal_id)` record; `resolve_publisher_authority` returning `None` denies (never implicit PLATFORM). **Tenant-wide workspace access requires an explicit `TENANT_WIDE` workspace grant**; missing registration or workspace authority denies (never tenant-wide from a missing grant). `SERVICE` / `ORG_SYSTEM` principal types alone do not grant PLATFORM. `CollaborativeActivityPublisherContext` carries authenticated `producer_principal_id`, distinct from semantic `publication.actor`. `CollaborativeActivityIngestionService` implements `CollaborativeActivityPublicationPort` and `CollaborativeActivityWritePort`, evaluates `CollaborativeActivityIngestionPolicy`, materializes `CollaborativeActivityAppendIntent` only on ALLOW, and calls `CollaborativeActivityAppendStore.append_idempotent` once per successful invocation. DENY, policy failure, and publisher resolution failure → zero store calls. Default policy: fail-closed namespace authorization (reserved `platform` / `intergrax` spoof denied for plugins), tenant/workspace alignment, source/type namespace rules, platform minimum durability resolution. **No direct source → append store.** MP-6 consumes verified identity; authentication itself is owned upstream. Live runtime registry integration is future configuration work (not MP-6C-C1-R1).

**MP-6C security (fail-closed):** spoofed platform namespace; cross-tenant publisher; unauthorized idempotent replay; plugin escalation; unregistered service escalation to PLATFORM; cross-tenant registration miss; missing workspace grant escalation to tenant-wide; implicit platform fallback; missing/broken policy; store bypass via raw publication.

**Capability statement:** Collaborative Activity is an immutable, typed semantic record of meaningful collaborative actions on the work plane — who acted, on what object, under which scope and authority, with what outcome, and which canonical evidence references enable reconstruction. MP-6 does **not** store WorkArtifact/Memory/UCL/prompt bodies, replace observability traces, duplicate proof receipts, or authorize mutations.

**Anti-substitution:** `Collaborative Activity != Runtime Trace`; Activity ≠ observability telemetry (retries, latency, tokens, HTTP); Activity ≠ `ProofReceipt` storage; Activity ≠ Decision/Approval authority (MP-4); Activity ≠ ContextView payload (MP-5 references `view_id` + scope only).

**Architecture flow:**

```text
Source Domain (authoritative mutation)
  → canonical successful operation result
  → replaceable MP-6F source integration adapter (translation only)
  → CollaborativeActivityPublication (neutral contract; requested durability only)
  → CollaborativeActivityPublicationPort / ingestion boundary
  → MP-6 Activity service (MP-6C policy validation — resolves effective durability)
  → CollaborativeActivityAppendIntent (policy-resolved persistence input)
  → CollaborativeActivityAppendStore.append_idempotent(intent) (replaceable;
     atomically assigns recorded_at + append_position for new keys only;
     materializes durability_class from intent.effective_durability_class)
  → configured provider

Read:
Authorized Consumer
  → CollaborativeActivityReadRequest + MP-1 effective authority
  → CollaborativeActivityReadAuthorizationPolicy (authorization before provider)
  → authorized CollaborativeActivityQuery
  → CollaborativeActivityReadPort (provider-neutral; no authorization inside store)
  → CollaborativeActivityPage (opaque scope-bound cursor; keyset by append_position)
```

**Anti-bypass (MP-6F):** source domains never import or call `CollaborativeActivityAppendStore`, activity SQLite/PostgreSQL providers, or `CollaborativeActivityAppendIntent`. Ingress is **`CollaborativeActivityPublicationPort` only**.

**MP-6F source integration (MP-6F-C1 — contract-driven):** replaceable typed mappers + decorators in [`collaborative_activity_source_mapping.py`](../../../intergrax/collaborative_work/collaborative_activity_source_mapping.py), [`collaborative_activity_source_ports.py`](../../../intergrax/collaborative_work/collaborative_activity_source_ports.py), [`collaborative_activity_source_adapters.py`](../../../intergrax/collaborative_work/collaborative_activity_source_adapters.py), [`collaborative_activity_source_wiring.py`](../../../intergrax/collaborative_work/collaborative_activity_source_wiring.py). Adapters depend on **source-operation Protocols** (`CollaborativeWorkActivityMutationPort`, `CollaborativeWorkArtifactActivityMutationPort`, `CollaborativeDecisionBindingActivitySourcePort`, `ContextViewComposer`); concrete services are composition-root choices only. Artifact publication results use structural `PublishedWorkArtifactPublicationResult` (not repository imports). **`publication.actor`** is source-proven semantic actor; **publisher identity** remains MP-6C (`VerifiedCollaborativeActivityPublisherIdentity` + explicit registration). **`source_stable_id`** is replay-stable operation identity (command `idempotency_key` for Collaborative Work mutations; deterministic `view_id` for ContextView compose).

| Source | Adapter contract | Integrated | `source_stable_id` |
|--------|------------------|------------|-------------------|
| WorkItem create / transition | `CollaborativeWorkActivityMutationPort` | Yes | request `idempotency_key` |
| WorkItem field update (`WORK_ITEM_UPDATED`) | — | **No** — no authoritative non-transition update seam | — |
| Assignment create / transition | `CollaborativeWorkActivityMutationPort` | Yes | request `idempotency_key` |
| WorkArtifact create / publish version | `CollaborativeWorkArtifactActivityMutationPort` | Yes | request `idempotency_key` |
| Collaborative decision binding create | `CollaborativeDecisionBindingActivitySourcePort` | Yes | request `idempotency_key` |
| ContextView compose | `ContextViewComposer` | Yes | composed `view_id` |
| Repository / log inference | — | **Forbidden** | — |

**Publication vs source mutation (fail-safe):** source mutation and activity publication are separate operations — **no cross-store atomicity**. On publication failure after a committed source mutation, the adapter **propagates** the publication error (no silent swallow). The caller may observe a successful source result only when publication also succeeded; if publication fails, the source mutation may already be committed and **no rollback is implied**. MP-6F does not classify final audit-critical durability: **MP-6C owns effective durability resolution**, so integration cannot safely discard failed publication. ContextView `occurred_at` in activity mapping is **composition completion time observed at the integration boundary**, not a fabricated domain event timestamp.

**MP-6E read semantics (qualified providers):** authorization runs before any `CollaborativeActivityReadPort` call; the page cursor is **not** an authorization token (scope/filter binding only). Continuation uses **keyset pagination by `append_position`** (`append_position > cursor`, `ORDER BY append_position ASC`, **no `OFFSET`**). `occurred_at` is a filter/display dimension, not a continuation watermark. Reads follow **forward live feed** semantics: appends after page *N* may appear on later pages; there is **no fixed historical snapshot** across pages without an explicit snapshot token. **SQLite** and **PostgreSQL** read providers are qualified under [`MP-6E-C1_POSTGRESQL_READ_PROVIDER_QUALIFICATION.md`](../maintainers/qualification/MP-6E-C1_POSTGRESQL_READ_PROVIDER_QUALIFICATION.md).

**Public contracts (MP-6A / MP-6A-C1 gate):** [`intergrax/contracts/collaborative_activity.py`](../../../intergrax/contracts/collaborative_activity.py) — `CollaborativeActivity`, `CollaborativeActivityTypeId`, `CollaborativeActivityBuiltinType`, `CollaborativeActivitySourceId`, `CollaborativeActivityActorRef`, `CollaborativeActivityTargetRef`, `CollaborativeActivityProvenanceRef`, `CollaborativeActivityOutcome`, `ActivityIdempotencyKey`, `CollaborativeActivityPublication`, `CollaborativeActivityAppendIntent`, `CollaborativeActivityQuery`, `CollaborativeActivityPageCursor`; ports `CollaborativeActivityPublicationPort`, `CollaborativeActivityWritePort`, `CollaborativeActivityReadPort`, `CollaborativeActivityAppendStore`.

**MP-6A-C1 — CLOSED / RECERTIFIED** (activity identity, extensibility, timeline semantics hardening; subject to independent audit). **MP-6A-C1-R1 — CLOSED** (atomic append position and materialization boundary; subject to independent audit). **MP-6B-C1-R1 — CLOSED** (AppendIntent append-store pluginability proof; subject to independent audit).

**MP-6B — CLOSED / RECERTIFIED** (**MP-6B-C1** append intent boundary; subject to independent audit):

```text
requested durability (publication)
  → policy resolution (MP-6C)
  → effective durability (append intent)
  → materialized durability_class (activity)
```

**MP-6B core DTO/runtime invariants** (subject to independent audit):

| Area | DTO / seam | Structural (contract) | Semantic (MP-6C+ policy) |
|------|------------|----------------------|----------------------------|
| Identity | `ActivityIdempotencyKey`, `mint_collaborative_activity_id` | scoped key, golden `activity-id/v1` hash | namespace emit authorization |
| Actor | `CollaborativeActivityActorRef` | tenant alignment, delegation pairing | principal legitimacy |
| Scope / target | `CollaborativeActivityScope`, `CollaborativeActivityTargetRef` | tenant/workspace/work_item alignment | type-target policy |
| Provenance | `CollaborativeActivityProvenanceRef` | reference-only, dedupe + canonical sort | legitimacy / fetch |
| Durability | publication `requested_durability_class` vs materialized `durability_class` | wire separation | effective class (no downgrade of audit-critical) |
| Publication vs record | `CollaborativeActivityPublication` vs `CollaborativeActivity` | no `recorded_at` / `append_position`; JSON roundtrip without wire `activity_type` | ingestion policy |
| Read | `CollaborativeActivityQuery`, `CollaborativeActivityPage` | time bounds, cursor scope semantics | authorization (MP-6E) |

**Serialization:** public wire DTOs are `frozen=True`, `extra="forbid"`, schema-versioned where independently transported; `CollaborativeActivityPublication.model_dump_json()` roundtrips without accepting foreign `activity_type` authority.

**Correction:** `platform.activity.correction` requires `caused_by_activity_id` or typed `collaborative_activity` target (`CollaborativeActivityRecordTargetRef`).

**Idempotency:** `ActivityIdempotencyKey(tenant_id, workspace_id, source, source_stable_id, activity_type)` with namespaced `CollaborativeActivitySourceId` and `CollaborativeActivityTypeId`; `activity_id = mint_collaborative_activity_id(...)` over versioned length-prefixed hash material (`activity-id/v1`, SHA-256 truncated to 32 hex); at-least-once delivery with deterministic deduplication; publication scope must match key tenant/workspace.

**Activity type extensibility:** namespaced plugin types via `CollaborativeActivityTypeId.for_extension(...)`; platform built-ins via `CollaborativeActivityBuiltinType`; reserved `platform` / `intergrax` namespaces — extension producers cannot claim without policy (MP-6C+).

**Ordering:** no global total order; **event-time presentation** may sort by `(occurred_at, activity_id)` for display; **forward pagination continuation** uses opaque `CollaborativeActivityPageCursor` (append/snapshot position — provider-neutral, not `occurred_at` watermark); materialized records carry monotonic unique `append_position` within `(tenant_id, workspace_id)` assigned by the configured append-store implementation under the platform append contract (not producers; duplicate idempotency keys do not allocate a new position); `recorded_at` is durable materialization time at the append boundary (producer does not set on publication); contiguous gapless numbering is not required; optional `caused_by_activity_id`; append-only (`platform.activity.correction` supersession — no in-place history rewrite). Late-arriving events (earlier `occurred_at`, later `recorded_at`) remain reachable when continuing from a cursor.

**Referential integrity:** validation at ingestion via public contracts and policy (`trust canonical producer` default for MP-6A; strict cross-domain checks via injected validators in MP-6C+ — no hardcoded repository imports in core). `caused_by_activity_id` must not cross tenant/workspace (MP-6C ingestion invariant).

**Threat model (fail-closed):** actor spoofing → publication boundary requires authoritative actor refs tied to resolved principal; tenant/workspace leakage → scope on every record + authorized query; **cross-tenant idempotency collision** → scoped key; **cross-plugin source collision** → namespaced `CollaborativeActivitySourceId` (no generic plugin bucket); **cursor late-arrival loss** → opaque append continuation vs event-time filters; **type spoofing / namespace squatting** → reserved namespaces + MP-6C policy seam; duplicate ingestion → idempotency key; forged provenance → typed refs only, validation seam; payload leakage → reference-only provenance union.

| Layer | May depend on | Must not depend on |
|-------|---------------|-------------------|
| Source domains | `intergrax/contracts/collaborative_activity.py` | MP-6 store implementations, vendors |
| MP-6 contracts | other `intergrax/contracts/*` refs | `agents/`, `applications/` |
| MP-6 implementation | contracts, MP-1 authority for validation | direct source repositories |
| Persistence providers | `CollaborativeActivityAppendStore` | collaborative semantics |
| Application / UI (MP-9) | `CollaborativeActivityReadPort` | store internals |

**Roadmap:** MP-6C publication boundary (**CLOSED**) → MP-6D store (**CLOSED**) → MP-6E scoped read (**CLOSED**) → MP-6F source integrations (**CLOSED**) → MP-6G E2E qualification (**CLOSED / CERTIFIED**) → MP-6H final enterprise certification (**CLOSED / CERTIFIED**) → **MP-6 — ENTERPRISE CERTIFIED / CLOSED**.

Capability coordination: [`MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md) § MP-6.

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
| [`DECISION_APPROVAL_GOVERNANCE.md`](DECISION_APPROVAL_GOVERNANCE.md) | MP-4 Decision / Approval / Governance ownership |
| [`MULTIPLAYER_AI.md`](../capabilities/architecture/MULTIPLAYER_AI.md) | Multi-layer feature coordination |
| [`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md) | Hosting boundary |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | Execution boundary |
| [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) | PLATFORM-INV-001…003 |
