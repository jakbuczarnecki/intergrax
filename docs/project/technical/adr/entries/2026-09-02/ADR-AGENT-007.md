# ADR-AGENT-007: Immutable historical EffectiveRoster snapshot authority (AC-3 Phase 4)

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture only) |
| **Date** | 2026-09-02 |
| **Deciders** | Agent Platform / Harness architecture |
| **Related** | [`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) §13–§18 · [`ADR-AGENT-004`](../2026-08-12/ADR-AGENT-004.md) · [`ADR-AGENT-005`](../2026-08-17/ADR-AGENT-005.md) · [`ADR-AGENT-006`](../2026-09-01/ADR-AGENT-006.md) · AC-3 · AC-3-PHASE-4-HISTORICAL-ROSTER-AUTHORITY-GAP |

## Context

`RuntimeRevision` freezes `effective_roster_revision_id` as immutable runtime identity ([`runtime_revision.py`](../../../../../../intergrax/agent_distribution/runtime_revision.py)). That value is a **content hash** produced by `EffectiveRoster.compute_revision_id()` ([`roster.py`](../../../../../../intergrax/agent_distribution/roster.py)). It proves **which** roster was used to build a revision, but it **cannot reconstruct** the exact historical roster after current desired state changes.

**Failure scenario:**

1. Revision **N** is built from package **A**, binding config **X**, factory reference **F1**.
2. Desired state later changes to package **B**, config **Y**, factory reference **F2**.
3. Revision **N** remains valid for serving and rollback.
4. Rebuilding `EffectiveRoster` from current installation/binding stores produces a **different** roster than revision **N** referenced.

Therefore old revisions cannot depend on current desired state for projection authority. A hash alone is integrity evidence, not payload authority.

[`ADR-AGENT-006`](../2026-09-01/ADR-AGENT-006.md) deferred effective roster authority and tentatively preferred deterministic reconstruction from installation/binding stores. That approach is **rejected** for historical runtime authority (see §Alternatives). This ADR supersedes that adjacent direction for AC-3 Phase 4.

**Gap closed:** AC-3-PHASE-4-HISTORICAL-ROSTER-AUTHORITY-GAP.

**Out of scope for this ADR:** snapshot model implementation, store implementation, build persistence wiring, production projection changes, `EffectiveRosterAuthorityService`, installation/binding store wiring, `RuntimeRevision` schema changes.

## Decision

Freeze **immutable historical lifecycle snapshot authority** for the exact `EffectiveRoster` used to build each `RuntimeRevision`.

### Source-of-truth model — two distinct roles

| Role | What it is | Used for |
|------|------------|----------|
| **Current desired state** | Installations, bindings, manifest defaults — mutable operator intent | Deriving **future** `EffectiveRoster` at build time via `EffectiveRosterBuilder` |
| **Historical runtime authority** | Immutable `EffectiveRoster` snapshot keyed by `effective_roster_revision_id` | Reproducing an **already-built** `RuntimeRevision` for projection, replay, and rollback |

`EffectiveRoster` remains **derived desired-state output at build time** — not a durable desired-state store of record. The snapshot is **not** current desired-state SoT. It **is** allowed and required as immutable historical lifecycle evidence for a built revision.

This resolves the apparent conflict with `EffectiveRosterEntry` docstring *"derived only, not durable SoT"* ([`roster.py`](../../../../../../intergrax/agent_distribution/roster.py) §13.4): **not** durable desired-state SoT; **is** durable historical lifecycle evidence per revision.

### Concept — `EffectiveRosterSnapshotStore`

Content-addressed immutable store for exact `EffectiveRoster` payloads.

**Primary key:** `effective_roster_revision_id`

**Methods only:**

```text
get_by_revision(
    effective_roster_revision_id: str
) -> EffectiveRoster | None

persist(
    roster: EffectiveRoster
) -> EffectiveRoster
```

**Explicitly absent from production lifecycle contract:**

- `update`, `delete`, mutable patch
- `latest` or list-as-authority APIs
- lookup by current `application_id` / `application_environment_id` as primary production authority

Historical lookup is by immutable content identity only.

### Snapshot identity

Primary identity is `effective_roster_revision_id`. It must equal `EffectiveRoster.compute_revision_id()`. Store key and payload hash must match.

**On persist (fail closed):**

1. `computed_id = roster.compute_revision_id()`
2. Require `roster.effective_roster_revision_id` is non-null
3. Require `computed_id == roster.effective_roster_revision_id`
4. Persist under key `effective_roster_revision_id`

Reject persist when revision id is null or hash mismatch.

### Immutable persist semantics

For a given `effective_roster_revision_id`:

| Case | Behavior |
|------|----------|
| **1** — no existing record | persist exact roster (success) |
| **2** — existing record, payload bitwise-equivalent | idempotent success |
| **3** — existing record, different payload under same id | conflict / fail closed |

Case 3 should be impossible under valid hashing; the store must defend against corruption or implementation bugs. **Never** silently overwrite content under the same revision id.

### Content-addressable contract

`effective_roster_revision_id` is simultaneously:

1. **Identity reference** stored in `RuntimeRevision`
2. **Integrity evidence** for snapshot payload

On read, `EffectiveRosterAuthorityService` must recompute `snapshot.compute_revision_id()` and require exact equality with:

- `snapshot.effective_roster_revision_id`
- `RuntimeRevision.effective_roster_revision_id`

No trust in store key name alone.

### `RuntimeRevision` reference — no new field

**No** new `effective_roster_snapshot_ref` (or similar) on `RuntimeRevision` V1.

Existing field is sufficient:

```text
RuntimeRevision.effective_roster_revision_id
```

This is the content-addressed reference. Reconsider a separate physical locator only if a future durable backing store requires it (analogous to `artifact_locator` on `RuntimeMaterializationRecord`). Not needed for process-local Reference Production V1.

### Build-time write order

Canonical build sequence (future implementation):

```text
desired-state stores
  → EffectiveRosterBuilder
  → EffectiveRoster with revision id
  → persist EffectiveRoster snapshot          ← new gate
  → dependency resolution
  → MaterializedRuntimeLock (persist_lock)
  → candidate RuntimeRevision (references effective_roster_revision_id)
  → materialize
  → RuntimeMaterializationRecord (persist)
  → mark VALIDATED
```

**Critical invariant:** a `RuntimeRevision` must never become canonical build output unless the roster snapshot it references **already exists**.

**Preferred ordering:** persist roster snapshot **before** `persist_candidate_revision`.

**Allowed deviation** (current `build_application_revision` persists candidate revision before materialization): roster snapshot must exist **no later than** candidate revision creation and **before** any revision can be treated as resumable or canonical. Replay and projection must fail closed if snapshot is missing even when revision record exists.

### Build replay

If an existing `RuntimeRevision` references `effective_roster_revision_id = H`, build replay must require snapshot **H** to exist.

| Condition | Behavior |
|-----------|----------|
| Snapshot **H** present and valid | replay proceeds (alongside existing materialization authority checks per ADR-AGENT-006) |
| Snapshot **H** missing | **fail closed** |
| Reconstruct **H** from current desired state | **forbidden** |
| Synthesize roster from hash | **forbidden** |

### Projection authority — `EffectiveRosterAuthorityService`

Future service (not implemented in this ADR):

1. Receive canonical `RuntimeRevision`
2. Load snapshot via `revision.effective_roster_revision_id`
3. Verify `snapshot.compute_revision_id()` equals revision reference
4. Verify scope alignment:
   - `application_id`
   - `application_environment_id`
   - `manifest_release_id` on snapshot equals `application_release_id` on revision (current build semantics — invariant **R5**)
5. Return exact historical `EffectiveRoster`

Production projection must **not** accept caller-supplied `EffectiveRoster` after Phase 4 implementation (invariant **R11**). Caller-fed roster is not production authority.

### Rollback semantics

Example: revision **N** uses roster **H_N**; revision **N+1** uses **H_N1**; current desired state may correspond to **H_N2**.

Rollback to **N** must load **H_N** from snapshot store — not **H_N1**, not **H_N2**, not a roster rebuilt from current installation/binding state. Snapshot store is essential for exact rollback without rematerialization of roster semantics.

### Desired-state mutation

Supported lifecycle:

```text
build N (snapshot H_N persisted)
  → mutate installation / binding / manifest-default state
  → build N+1 (snapshot H_N1 persisted)
  → project / serve / rollback N
```

Snapshot **H_N** remains immutable and valid. No desired-state mutation may alter an existing snapshot.

### Factory reference authority

Snapshot preserves exact `EffectiveRosterEntry` fields including:

- `package_digest`, `distribution_package_id`
- `active_installation_id`, `effective_enablement`, `effective_default_agent`
- `merged_config`, `secret_refs`, `policy_overrides`
- `factory_reference`, `application_binding_id`, `manifest_origin_ref`

Historical factory resolution for projection must originate from snapshot roster entries. **No** manifest executable fallback for historical authority.

### Manifest defaults gap (deferred, not blocking)

Manifest defaults are required to **derive** a new roster at build time. They are **not** required to reconstruct a historical roster once snapshot exists.

AC-3-PHASE-4-MANIFEST-DEFAULTS-AUTHORITY-GAP is a build-time desired-state concern, not a blocker for historical projection after snapshot persistence. Manifest-default persistence redesign is out of scope.

### Installation/binding store continuity gap (deferred, not blocking)

Installation/binding store continuity matters for deriving **new** rosters. Historical projection must **not** depend on current installation/binding state. Production projection after Phase 4 must not require installation/binding stores solely to recreate old runtime authority.

### Authority chain (separate concerns)

```text
EffectiveRosterSnapshot
        ↓ H_roster (effective_roster_revision_id)
RuntimeRevision
        ↓ lock id / digest
MaterializedRuntimeLock
        ↓
RuntimeMaterializationRecord
        ↓ artifact_locator
artifact
```

Do **not** embed roster payload into `RuntimeMaterializationRecord` or `MaterializedRuntimeLock`.

### Production V1 ownership

Extends [`ADR-AGENT-005`](../2026-08-17/ADR-AGENT-005.md) and [`ADR-AGENT-006`](../2026-09-01/ADR-AGENT-006.md):

```text
ProductionProcessComposition
  → ProductionAgentPlatformRuntime
  → AgentPlatformRuntimeStores
  → EffectiveRosterSnapshotStore
```

Reference Production V1 may use process-local in-memory implementation over shared `AgentDistributionStoreState`, alongside `revision_store`, `lock_store`, `materialization_store`, `serving_store`, `registry_projection_store`. **Restart loses state.** Multi-instance durable implementation deferred.

### Shared state and concurrency

Snapshot storage lives on shared `AgentDistributionStoreState`. Multiple store wrappers over the same state must see the same snapshots. Use shared state-level synchronization (same pattern as `InMemoryRuntimeMaterializationStore` / `_materialization_lock`).

`persist` must be atomic across wrappers sharing one state:

```text
shared state lock
  → lookup by effective_roster_revision_id
  → insert / idempotent return / conflict
```

No silent overwrite. No hidden singleton.

### Snapshot retention

Snapshots referenced by any retained `RuntimeRevision` are immutable historical evidence. Do **not** garbage-collect a snapshot while any retained revision references its `effective_roster_revision_id`. Actual GC/retention mechanism is **out of scope**; future policy may GC only when revision retention guarantees permit.

### Storage duplication (intentional)

`EffectiveRoster` is already materialized transiently during build. Persisting the exact roster payload is **intentional** — the hash alone cannot reconstruct payload. This is required historical evidence, not accidental duplication.

### Error semantics

Future implementation uses narrow typed conflict / missing-authority errors (exact class names not frozen). Behavior is frozen:

- missing snapshot → fail closed
- hash mismatch → fail closed
- scope mismatch → fail closed
- duplicate persist with different payload → fail closed

Possible names: `EffectiveRosterSnapshotConflict`, `EffectiveRosterAuthorityMissing`.

## Alternatives considered

| Alternative | Verdict | Reason |
|-------------|---------|--------|
| **Versioned desired-state history** — version every binding/installation/manifest-default and support as-of reconstruction | **Rejected for AC-3 V1** | Much broader subsystem; requires coherent cross-store historical transaction/version point; unnecessary when exact merged roster already exists at build time. May be future audit architecture. |
| **Inline roster in `RuntimeRevision`** | **Rejected** | Revision already has content-addressed roster identity; inline payload enlarges revision contract; duplicates roster across revisions sharing same roster; complicates schema migration. Separate content-addressed snapshot allows reuse. |
| **Current-state reconstruction** — `effective_roster_revision_id` + current installation/binding → rebuild and compare | **Rejected** | Works only until desired state changes; fails historical serving and rollback ([`ADR-AGENT-006`](../2026-09-01/ADR-AGENT-006.md) adjacent direction superseded). |
| **`EffectiveRosterStore` as mutable desired-state SoT** | **Rejected** | Conflicts with derived-roster architecture; conflates desired state with historical authority. |

## Consequences

### Positive

- Closes AC-3-PHASE-4-HISTORICAL-ROSTER-AUTHORITY-GAP with minimal contract surface
- Enables exact rollback and historical projection after desired-state mutation
- Reuses existing `effective_roster_revision_id` on `RuntimeRevision` — no schema migration
- Aligns with `RuntimeMaterializationRecord` immutability pattern (ADR-AGENT-006)
- Clear separation: desired state derives future rosters; snapshots authorize past revisions

### Negative

- One new store Protocol and in-memory V1 implementation
- `AgentPlatformRuntimeStores` bundle extension
- Build orchestration persistence step before candidate revision (preferred)
- `EffectiveRosterAuthorityService` and projection API tightening in follow-up phases
- Storage duplication per distinct roster hash (accepted)

## Architecture invariants

| ID | Invariant |
|----|-----------|
| **R1** | Every built `RuntimeRevision` references exactly one `effective_roster_revision_id` |
| **R2** | Every referenced `effective_roster_revision_id` resolves to an immutable `EffectiveRoster` snapshot |
| **R3** | `snapshot.compute_revision_id() == snapshot.effective_roster_revision_id == RuntimeRevision.effective_roster_revision_id` |
| **R4** | Snapshot scope (`application_id`, `application_environment_id`) equals `RuntimeRevision` scope |
| **R5** | Snapshot `manifest_release_id` equals `RuntimeRevision.application_release_id` under current build semantics |
| **R6** | Desired-state mutation cannot change an existing snapshot |
| **R7** | Production projection uses historical snapshot, never current desired-state reconstruction |
| **R8** | Rollback reuses snapshot referenced by target `RuntimeRevision` |
| **R9** | Missing or corrupt snapshot fails closed |
| **R10** | Snapshot store never silently overwrites content under the same revision id |
| **R11** | Caller-supplied `EffectiveRoster` is not production authority after Phase 4 |
| **R12** | Factory references for historical projection originate from snapshot entries |

## Non-goals

- No `EffectiveRosterStore` as mutable desired-state store
- No versioned binding subsystem
- No event sourcing
- No manifest-default persistence redesign
- No AC-4 capability matching
- No activation locator cleanup
- No distributed durable database adapter
- No snapshot GC implementation
- No `RuntimeRevision` schema V2

## Compliance

- Tier boundaries preserved: Tier-0 contracts and stores; Tier-3 hosts consume projections
- No implementation in this ADR
- Supersedes ADR-AGENT-006 adjacent effective-roster reconstruction direction for historical authority
- Aligns with [`AGENT_DISTRIBUTION.md`](../../../architecture/AGENT_DISTRIBUTION.md) §13–§18
- Extends store ownership from [`ADR-AGENT-005`](../2026-08-17/ADR-AGENT-005.md)

## Implementation plan — Phase 4 sequence

| Phase | Scope |
|-------|-------|
| **A** | `EffectiveRosterSnapshotStore` Protocol + in-memory shared-state implementation + atomic persist tests |
| **B** | Persist snapshot in `build_application_revision()` + replay fail-closed if snapshot missing |
| **C** | `EffectiveRosterAuthorityService` — revision → snapshot lookup + validation |
| **D** | Remove `EffectiveRoster` from public production projection API (caller-fed roster no longer authority) |
| **E** | Historical proof: build N → mutate desired state → build N+1 → project N → exact roster N; then Phase 4 closure |

**Do not implement in this ADR task.**
