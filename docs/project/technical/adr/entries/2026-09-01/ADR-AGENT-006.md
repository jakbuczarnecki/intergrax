# ADR-AGENT-006: Canonical runtime materialization authority (AC-3)

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture only) |
| **Date** | 2026-09-01 |
| **Deciders** | Agent Platform / Harness architecture |
| **Related** | [`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) §18–§19 · [`ADR-AGENT-004`](../2026-08-12/ADR-AGENT-004.md) · [`ADR-AGENT-005`](../2026-08-17/ADR-AGENT-005.md) · AC-3 |

## Context

AC-3 (production registry projection from canonical lifecycle authority) is blocked because `artifact_locator` is **ephemeral or caller-fed** today. Projection and activation cannot treat any durable store as the source of truth for where the immutable materialization artifact lives.

**Current nearest abstractions (insufficient as authority):**

| Abstraction | Role today | Why not canonical authority |
|-------------|------------|------------------------------|
| `MaterializationOutput.artifact_locator` | Returned by `RuntimeMaterializationService.materialize()` | Ephemeral return value; not persisted |
| `BuildRevisionResult.artifact_locator` | API/result field after build | Returned to caller only; not stored |
| `ActivateRuntimeRevisionRequest.artifact_locator` | Caller-fed activation input | Caller becomes source of truth |
| `DeploymentInstanceRecord.serving_unit_ref` | Deployment readiness / serving unit | Deployment ≠ immutable artifact identity |
| `RuntimeRevision` | Immutable runtime identity | Contains `materialization_artifact_digest` but **no** `artifact_locator` |

**Required chain for AC-3:**

```text
runtime_revision_id
  → canonical stored materialization record
  → exact artifact_locator
  → exact expected artifact digest
  → production VENV resolver
```

This ADR freezes the missing typed contract. **No implementation** in this task.

## Decision

Introduce **`RuntimeMaterializationRecord`** and **`RuntimeMaterializationStore`** as the canonical lifecycle authority for the immutable materialization artifact locator bound to one `RuntimeRevision`.

### Concept

`RuntimeMaterializationRecord` is an **immutable canonical lifecycle record** proving which exact materialized runtime artifact belongs to one `RuntimeRevision`.

It is **not**:

- deployment readiness
- active serving pointer
- registry projection
- package lock itself
- arbitrary artifact path cache
- manifest-derived state

### Record identity

| Field | Classification | Rule |
|-------|----------------|------|
| **Primary authority key** | `runtime_revision_id` | One record per revision; store keyed by this id |
| **Scope validation** | `application_id`, `application_environment_id` | Must match the bound `RuntimeRevision`; reject cross-environment misuse |

**Uniqueness model (aligned with existing `RuntimeRevision` contract):**

- `runtime_revision_id` is the **global primary key** for revision records (`InMemoryRuntimeRevisionStore` keys `revisions` by id; `AgentPlatformAdminService.build_revision` rejects the same id with a different `(application_id, application_environment_id)`).
- `application_id` + `application_environment_id` are **mandatory scope validators** on the materialization record, not a composite primary key.
- A revision always belongs to exactly one application environment ([`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) §18.1).

### Frozen field set — `RuntimeMaterializationRecord`

Pydantic contract (future implementation): `extra="forbid"`, `frozen=True`.

| Field | Classification | Required |
|-------|----------------|----------|
| `runtime_revision_id` | authority | yes |
| `application_id` | validation / scope | yes |
| `application_environment_id` | validation / scope | yes |
| `materialization_topology` | authority | yes |
| `artifact_locator` | authority | yes |
| `materialization_artifact_digest` | authority (content identity) | yes |
| `materialized_runtime_lock_id` | authority | yes |
| `materialized_runtime_lock_digest` | authority | yes |

**Explicitly excluded** (already on `RuntimeRevision` or out of scope):

- `build_input_digest` — identity field on `RuntimeRevision`, not locator authority
- `materialized_at`, materializer identity/version — no existing lifecycle convention requires them; audit events remain the observability path

### Immutability — `persist(record)` semantics

For a given `runtime_revision_id`:

| Case | Behavior |
|------|----------|
| **1** — no existing record | persist (success) |
| **2** — existing record, all **authority fields** bitwise-equivalent | idempotent success |
| **3** — existing record differs in **any** authority field | conflict / fail closed |

**Never permitted:**

- update locator in place
- update digest in place
- change topology in place
- rebind lock
- repoint the same revision to a new artifact

**New artifact ⇒ new `RuntimeRevision`.** Physical relocation of bytes for the same revision is forbidden; relocation requires a new materialization event and new revision authority, not mutation of the existing record.

### Store contract — `RuntimeMaterializationStore`

Minimum Protocol (names follow existing store verbs `get_*` / `persist_*`):

```text
get_by_revision(runtime_revision_id: str) -> RuntimeMaterializationRecord | None
persist(record: RuntimeMaterializationRecord) -> RuntimeMaterializationRecord
```

**Absent from production lifecycle contract (by design):**

- list APIs
- delete APIs
- mutable update APIs
- query-by-app APIs

### Ownership (Reference Production V1)

Extends [`ADR-AGENT-005`](../2026-08-17/ADR-AGENT-005.md):

```text
ProductionProcessComposition
  → ProductionAgentPlatformRuntime
  → AgentPlatformRuntimeStores
  → RuntimeMaterializationStore
```

- Reference Production V1 may use an **in-memory implementation** over shared `AgentDistributionStoreState` (same pattern as `InMemoryMaterializedRuntimeLockStore`, `InMemoryRuntimeRevisionStore`).
- **One OS process**, process-local state; lifecycle + projection + serving share the same composition root.
- **Restart loses state**; no multi-instance durability claim.
- **No hidden singleton.** **No builder-local store.**

### Authoritative write point

**Separation of concerns:**

- `RuntimeMaterializationService` **produces** `MaterializationOutput`; it does **not** own durable stores by design.
- **Lifecycle orchestration persists authority** after successful materialization and cross-check against the candidate revision and lock.

**Frozen sequence after `materialize()` succeeds:**

```text
RuntimeMaterializationService.materialize(...)
  → MaterializationOutput
  → validate against RuntimeRevision + MaterializedRuntimeLock
  → RuntimeMaterializationStore.persist(RuntimeMaterializationRecord)
  → only then: projection preparation / activation may continue
```

**Writer:** `AgentPlatformAdminService.build_revision` (canonical build orchestration path in `intergrax/agent_distribution/admin_service.py`), immediately after successful materialization validation and before returning `BuildRevisionResult`. The API result field `BuildRevisionResult.artifact_locator` becomes a **read-through echo** of the persisted record, not independent authority.

### Relation to `RuntimeRevision`

`RuntimeMaterializationRecord` must agree with `RuntimeRevision` on:

- `runtime_revision_id`
- `application_id`
- `application_environment_id`
- `materialization_topology`
- `materialization_artifact_digest`
- `materialized_runtime_lock_id`
- `materialized_runtime_lock_digest`

**Mandatory timing:** these fields are required on `RuntimeRevision` when `revision_state ∈ {validated, active, superseded}` (see `RuntimeRevision._validate_state_requirements`). The materialization record must exist **before** registry projection preparation and activation eligibility.

`artifact_locator` remains **only** on `RuntimeMaterializationRecord`, not on `RuntimeRevision`.

### Relation to `MaterializedRuntimeLock` — three-way trust

```text
RuntimeRevision
  ↕  (lock id + digest equality)
RuntimeMaterializationRecord
  ↕  (lock id + digest equality)
MaterializedRuntimeLockStore.get_lock(lock_id)
  ↕  (lock digest equality)
artifact embedded lock (VENV bundle)
```

Production authority lookup:

1. Load `RuntimeRevision` by id.
2. Load `RuntimeMaterializationRecord` by `runtime_revision_id`; validate app/env scope.
3. Load `MaterializedRuntimeLock` from `MaterializedRuntimeLockStore`; verify id and digest against revision and record.
4. Resolve `artifact_locator` through production VENV resolver; verify `materialization_artifact_digest` and embedded lock against canonical lock.

### Artifact locator semantics

`artifact_locator` means: **opaque canonical locator identifying the immutable materialization artifact.**

- The **record stores** the locator.
- The **production resolver interprets** supported schemes.
- The **store does not** understand filesystem paths.

| Scheme | Production authority |
|--------|---------------------|
| `file://` | supported (Reference Production V1 / VENV_BUNDLE) |
| `test://` | tests and proofs only |
| `reference://` | **forbidden** for production authority (see `resolve_production_artifact_root`) |

Contract allows opaque locators; current production resolver supports **VENV_BUNDLE only** (`OCI_IMAGE`, `SANDBOX_SIDECAR` deferred).

### Content identity vs location

- `materialization_artifact_digest` = **immutable content identity** (`sha256:<64 hex>` per `RuntimeMaterializationService`).
- `artifact_locator` = **location evidence**, not content identity.

Production must verify **both** the digest and the locator→artifact relationship. If the same immutable bytes are physically relocated, the **same `RuntimeRevision` may not be rebound**; relocation requires a new revision/materialization authority record.

### Separation from deployment and serving

| Record | Meaning |
|--------|---------|
| `RuntimeMaterializationRecord` | what immutable runtime artifact was built |
| `DeploymentInstanceRecord` | what serving/deployment instance was prepared (`serving_unit_ref`, readiness) |
| `ApplicationEnvironmentServingRecord` | which revision currently owns traffic |

```text
materialization ≠ deployment ≠ activation
```

Do not overload `DeploymentInstanceRecord` with artifact locator authority.

### Relation to registry projection

Projection preparation consumes canonical authority from:

- `RuntimeRevisionStore`
- effective roster authority (see adjacent decision)
- `MaterializedRuntimeLockStore`
- **`RuntimeMaterializationStore`**

```text
RuntimeMaterializationRecord.artifact_locator
  → VENV resolver (`resolve_production_artifact_root` / `build_production_runtime_agent_factory_resolver`)
  → RegistryProjection
```

`RuntimeRegistryProjectionStore` holds the **derived** materialized projection; it is **not** the source of `artifact_locator` truth.

### Adjacent decision — effective roster (deferred)

**Direction (not fully designed here):**

- Do **not** create `EffectiveRosterStore` in this ADR.
- Prefer **`EffectiveRosterAuthorityService`** that deterministically reconstructs roster from:
  - `AgentInstallationStore`
  - `ApplicationAgentBindingStore`
  - manifest defaults / contract metadata where legitimate
- Then verifies: `rebuilt.effective_roster_revision_id == RuntimeRevision.effective_roster_revision_id`

Rationale: avoid duplicating derived immutable state when deterministic reconstruction is already the architecture. **Separate follow-up decision**; not solved by this ADR.

### Failure model (fail closed)

Reject (no fallback) when:

- missing materialization record
- revision id mismatch
- app/env mismatch
- topology mismatch
- artifact digest mismatch
- lock id mismatch
- lock digest mismatch
- unsupported `artifact_locator` scheme at resolver boundary
- duplicate `persist` with different authority fields

**Forbidden fallbacks:**

- `ActivateRuntimeRevisionRequest.artifact_locator`
- `BuildRevisionResult.artifact_locator` (without persisted record)
- manifest paths
- environment variables
- application default paths

### Lifecycle sequence

Aligned with current `AgentPlatformAdminService.build_revision` ordering, with the **new** persistence gate before projection/activation:

| Step | Action |
|------|--------|
| 1 | canonical roster |
| 2 | canonical lock (`MaterializedRuntimeLockStore.persist_lock`) |
| 3 | candidate `RuntimeRevision` (`persist_candidate_revision`) |
| 4 | materialize artifact (`RuntimeMaterializationService.materialize`) |
| 5 | compute/verify artifact digest (in `MaterializationOutput`) |
| 6 | **`persist RuntimeMaterializationRecord`** ← new authority gate |
| 7 | finalize revision to `validated` (`mark_validated`) |
| 8 | prepare `RegistryProjection` (reads canonical record) |
| 9 | READY |
| 10 | COMMIT |
| 11 | serving pointer |

**Invariant:** projection preparation and activation cannot proceed without persisted canonical materialization authority.

### Rollback / N+1

N and N+1 each have separate immutable:

- `RuntimeRevision`
- `RuntimeMaterializationRecord`
- registry projection

READY on N+1 does not alter N. COMMIT moves serving pointer only. Rollback to N reuses N's prior materialization record and artifact identity; **no rematerialization** required if the artifact remains available at the stored locator.

### Reference Production V1 scope limits

| Limitation | Accepted for V1 |
|------------|-----------------|
| Topology | `VENV_BUNDLE` only for executable production resolver |
| Store | process-local in-memory over shared `AgentDistributionStoreState` |
| Durability | restart loses materialization authority |
| Scale | no multi-instance consistency |
| Persistence | no durable database |
| Deferred | `OCI_IMAGE`, `SANDBOX_SIDECAR` |

## Alternatives considered

| Alternative | Verdict | Reason |
|-------------|---------|--------|
| Put `artifact_locator` on `RuntimeRevision` | **Rejected** | `RuntimeRevision` is immutable runtime **identity**; produced artifact **evidence** (locator) belongs in a separate lifecycle record; preserves build vs identity boundaries |
| Reuse `DeploymentInstanceRecord` | **Rejected** | deployment instance ≠ immutable artifact identity |
| Caller supplies locator during activation | **Rejected** | caller becomes production authority (`ActivateRuntimeRevisionRequest.artifact_locator` today) |
| Derive locator from digest/path convention | **Rejected** | hidden path mapping breaks explicit lifecycle contracts |
| Store full artifact path in `RegistryProjection` only | **Rejected** | projection is downstream; cannot be upstream materialization authority |

## Consequences

### Positive

- Removes caller-supplied artifact authority from production path
- Supports exact revision rollback without rematerialization
- Makes projection input assembly deterministic
- Allows future durable store implementation without contract change
- Preserves topology and deployment separation

### Negative

- One new typed record and store Protocol
- One in-memory V1 implementation and `AgentDistributionStoreState` extension
- `AgentPlatformRuntimeStores` bundle extension
- Build orchestration persistence step in `AgentPlatformAdminService`
- Activation/prepare paths must load locator from store (follow-up implementation)

## Architecture invariants

| ID | Invariant |
|----|-----------|
| **I1** | One `RuntimeRevision` has at most one canonical `RuntimeMaterializationRecord` |
| **I2** | `RuntimeMaterializationRecord` is immutable after persistence |
| **I3** | Caller-provided `artifact_locator` is never production authority |
| **I4** | Artifact digest is content identity; locator is location evidence |
| **I5** | Revision, materialization record, lock, and artifact-embedded lock must agree |
| **I6** | Projection preparation requires persisted materialization authority |
| **I7** | Deployment/readiness records are separate from materialization authority |
| **I8** | Lifecycle and serving use stores from the same `ProductionProcessComposition` |
| **I9** | READY does not change serving authority; COMMIT does |
| **I10** | Missing canonical authority fails closed |

## Compliance

- Tier boundaries preserved: Tier-0 contracts and stores; Tier-3 hosts consume projections
- No implementation in this ADR
- Aligns with [`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) §18–§19, §21
- Extends store ownership from [`ADR-AGENT-005`](../2026-08-17/ADR-AGENT-005.md)

## Implementation plan — next tasks only

| Phase | Scope |
|-------|-------|
| **1** | `RuntimeMaterializationRecord`, `RuntimeMaterializationStore` Protocol, in-memory implementation, shared state ownership, unit tests |
| **2** | Add `materialization_store` to `AgentPlatformRuntimeStores`; persist record in canonical build path |
| **3** | Production projection authority service loads revision + lock + materialization record |
| **4** | `EffectiveRosterAuthorityService` wiring |
| **5** | Real production prepare→projection→activation proof; AC-3 closure audit |

**Do not implement in this ADR task.**
