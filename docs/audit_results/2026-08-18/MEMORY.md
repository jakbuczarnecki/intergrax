# MEMORY - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** MEMORY
- **Constituent domains:** MEMORY (user profile LTM · episodic index · task memory · MemoryView · vector secondary indexes)
- **Tier(s):** Tier-0 `intergrax/memory/` · Tier-1 `intergrax/runtime/task_memory/` · Tier-3 host wiring `intergrax/applications/_shared/memory_vector_wiring.py`
- **audited_sha:** `628e24130de34f291a416cb1cff9397a2b327dec`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 7 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/MEMORY.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/MEMORY.md`
- **Scope in:**
  - `UserProfileManager` LTM retrieval paths (direct vectorstore vs `RetrievalService`)
  - `VectorSessionTurnIndexStore` tenant-bound episodic index operations
  - `UserProfileManager` forget/delete lifecycle vs derived LTM vectors
  - `UserProfileStore` / `UserProfile` optimistic concurrency contract
  - `PolicyScopedMemoryView` scope authority, retention, and mutation semantics
  - `RuntimeExecutionContext` identity fields consumed by MemoryView
  - Tier-3 `memory_vector_wiring` tenant requirement (positive control)
  - Memory / RAG / Context Engineering separation (positive control)
  - historical MEM / MEM-DEPTH / MEM-VEC **Done** delivery states (positive control - not re-audited as failures)
- **Scope out:**
  - remediation implementation
  - second Memory subsystem or duplicate LTM store
  - universal re-qualification of all vector providers beyond documented bounds
  - RAG or Context Engineering domain re-audit beyond Memory touchpoints
  - silent runtime fixes in production source
- **Prior audit reference(s):** Protocol v2 [`RAG`](RAG.md) (`RAG-SCOPE-CONTRACT-INTEGRITY` - MEMORY-01 is downstream evidence); Protocol v2 [`IDENTITY_TRUST`](IDENTITY_TRUST.md) (`IDT-FIX-*` - coordinate MEMORY-05); historical MEM / MEM-DEPTH / MEM-VEC **Done** rows remain valid delivery facts
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `e1c3aa3f55dbc7231d1315ae8b8aa40bbd9914fe`

## Executive summary

**Verdict: FAIL.** Seven accepted findings (5 HIGH, 2 MEDIUM) show Memory-owned wiring defects on LTM `RetrievalService` recall (missing canonical `VectorStoreScope`), tenant-bound episodic index identity (per-call tenant override), primary/secondary index lifecycle on forget/delete, blind `UserProfile` aggregate overwrite without versioned concurrency, `PolicyScopedMemoryView` scope authority derived from independently writable constructor arguments, inconsistent retention filtering between `read` and `list`, and fail-late `update_memory_entry` when entry identity is unknown. Positive controls: Memory / RAG / Context Engineering separation is sound; vector indexes are correctly secondary to primary stores; direct LTM vector retrieval is scoped; single-entry soft delete removes its vector; retrieved LTM ids reconcile to active primary entries; `MemoryProfile` is typed with basic validation; architecture honestly claims P2/E3 not production qualification; procedural memory remains minimal; findings do not require a second Memory subsystem. Protocol v2 residual contract defects are distinct from MEM / MEM-DEPTH / MEM-VEC delivery completion - remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** - 0 CRITICAL / 5 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-MEMORY-01

- **Severity:** HIGH
- **Category:** SECURITY / TENANT ISOLATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-SCOPE-AUTHORITY-INTEGRITY
- **Claim falsified:** Every LTM retrieval path, including `RetrievalService`-backed recall, carries the same canonical tenant/namespace/workspace `VectorStoreScope` as indexing and direct vector querying.
- **Substance:** `UserProfileManager` owns a canonical `_vector_scope()` containing tenant, namespace, and workspace. The direct vectorstore LTM search path correctly supplies this scope. However `_search_longterm_via_retrieval_service()` constructs `RetrievalRequest` with query/top-k/threshold/metadata_filter and omits `scope=self._vector_scope()`. Thus the actual Memory consumer invokes canonical `RetrievalService` unscoped. Metadata filters for `user_id`/`deleted`/`index_domain`/`collection_name` do not replace system-owned `VectorStoreScope`. This is concrete downstream evidence of accepted RAG-01, but it is also a Memory-owned wiring defect.
- **Evidence:**
  - `intergrax/memory/user_profile_manager.py` - `_vector_scope()`; `_search_longterm_via_retrieval_service()` `RetrievalRequest` without `scope`
  - `intergrax/memory/user_profile_manager.py` - direct vectorstore search path supplies scope (positive contrast)
- **Confidence:** HIGH - direct code path; Memory consumer omits scope on RetrievalService path.
- **Target invariant:** Every LTM retrieval, including RetrievalService-backed recall, carries the same canonical tenant/namespace/workspace `VectorStoreScope` as indexing and direct vector querying. Coordinate with `RAG-SCOPE-CONTRACT-INTEGRITY`. Do not build a second retrieval path.

### AUDIT-20260818-MEMORY-02

- **Severity:** HIGH
- **Category:** SECURITY / TENANT BOUNDARY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-SCOPE-AUTHORITY-INTEGRITY
- **Claim falsified:** A materialized tenant-bound memory index cannot switch tenants.
- **Substance:** `VectorSessionTurnIndexStore` is constructed with `_tenant_id`, suggesting a tenant-bound episodic index dependency. However public upsert/search/tombstone operations accept `tenant_id`, and `_scope` uses `tenant_id or self._tenant_id`. A store instance materialized for tenant A can therefore perform vector operations for tenant B simply by receiving tenant B per call.
- **Evidence:**
  - `intergrax/memory/session_turn_index_service.py` - `_tenant_id` at construction; per-call `tenant_id` parameter; `_scope(tenant_id or self._tenant_id)`
- **Confidence:** HIGH - explicit per-call tenant override on tenant-bound instance.
- **Target invariant:** Either tenant identity is fixed at construction and per-call identity must match it, or the component is explicitly an unbound multi-tenant service with a trusted canonical tenant authority. Do not keep an ambiguous hybrid.

### AUDIT-20260818-MEMORY-03

- **Severity:** HIGH
- **Category:** PRIVACY / RETENTION / DATA-LIFECYCLE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-DURABILITY-LIFECYCLE-INTEGRITY
- **Claim falsified:** Primary Memory lifecycle and secondary retrieval index lifecycle are coordinated - forget/delete produce deterministic tombstones/removal for every derived memory index.
- **Substance:** `remove_memory_entry()` soft-deletes the primary entry and deletes the vector. But `clear_memory()` marks all active entries deleted and persists the profile, but does not delete/tombstone their vector records. `delete_profile()` deletes only the primary profile and does not remove LTM vectors. Therefore user-level forget/account-deletion flows can leave original memory content physically present in the secondary vector index.
- **Evidence:**
  - `intergrax/memory/user_profile_manager.py` - `remove_memory_entry()` vector delete (positive contrast); `clear_memory()` no vector tombstone; `delete_profile()` no LTM vector removal
- **Confidence:** HIGH - lifecycle asymmetry on bulk forget and profile delete.
- **Target invariant:** Primary store remains source of truth, but "not returned" is not equivalent to privacy deletion. Forget/delete must produce deterministic tombstones/removal for every derived memory index, with retry/reconciliation semantics for partial failure. Do not claim distributed transaction/exactly-once if unavailable.

### AUDIT-20260818-MEMORY-04

- **Severity:** HIGH
- **Category:** CONCURRENCY / CONSISTENCY ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-DURABILITY-LIFECYCLE-INTEGRITY
- **Claim falsified:** Durable user-profile Memory mutation requires optimistic concurrency or another canonical atomic mutation contract.
- **Substance:** `UserProfileStore` exposes only `get_profile`/`save_profile`/`delete_profile`. `save_profile` has blind whole-aggregate overwrite semantics. The contract has no `expected_revision`/CAS/conflict mechanism. `UserProfile` itself carries `version`, but the store contract does not use it. `UserProfileManager` performs read-modify-write aggregate operations for long-term memory entries and system instructions. Concurrent workers reading the same profile revision can therefore overwrite one another and lose accepted memory updates.
- **Evidence:**
  - `intergrax/memory/user_profile_store.py` - `save_profile` whole-aggregate overwrite; no expected revision
  - `intergrax/memory/user_profile_memory.py` - `UserProfile.version` field present but unused by store contract
  - `intergrax/memory/user_profile_manager.py` - read-modify-write on LTM entries and system instructions
- **Confidence:** HIGH - version field exists but store contract ignores it.
- **Target invariant:** Use one revision authority. Concurrent conflicting writes fail explicitly or retry through a deterministic merge policy. Do not implement provider-specific locking as the platform contract.

### AUDIT-20260818-MEMORY-05

- **Severity:** HIGH
- **Category:** IDENTITY / AUTHORIZATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-SCOPE-AUTHORITY-INTEGRITY
- **Claim falsified:** MemoryView authority is derived from canonical trusted execution identity and cannot be independently rebound by constructor arguments or mutable request metadata.
- **Substance:** `PolicyScopedMemoryView` accepts `RuntimeExecutionContext`, `tenant_id`, and `task_id` as independent values. It does not validate `task_id == exec_ctx.task_id`. `RuntimeExecutionContext` has no canonical `tenant_id` field. `_guard_scope_boundary()` compares `memory_scope_tenant_id` metadata to `self._tenant_id`, defaulting the expected value to `self._tenant_id` itself. The guard is called on write only; read/list/delete do not call it. Current UAEP wiring passes request tenant and current task consistently, which is a positive control, but the canonical reusable MemoryView boundary itself does not prove ownership.
- **Evidence:**
  - `intergrax/runtime/task_memory/memory_view.py` - independent `tenant_id`/`task_id`; `_guard_scope_boundary()` write-only; metadata default to `self._tenant_id`
  - `intergrax/contracts/runtime_execution_context.py` - no canonical `tenant_id` field
  - `intergrax/agents/uaep.py` - consistent tenant/task wiring (positive contrast)
- **Confidence:** HIGH - reusable boundary does not enforce identity closure.
- **Target invariant:** All read/write/list/delete operations preserve the same scope boundary. Do not duplicate tenant identity in several independently writable fields. Coordinate with IDENTITY_TRUST remediation.

### AUDIT-20260818-MEMORY-06

- **Severity:** MEDIUM
- **Category:** RETENTION / CONTRACT CONSISTENCY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-READ-MUTATION-CONSISTENCY
- **Claim falsified:** Retention semantics apply uniformly to every Memory read surface.
- **Substance:** `PolicyScopedMemoryView.read()` applies `should_forget_stm_record()` and hides an expired record. `PolicyScopedMemoryView.list()` returns `TaskMemoryCoordinator.list_namespace()` results without applying the same retention rule. The same expired record can therefore be invisible through direct read but visible through list.
- **Evidence:**
  - `intergrax/runtime/task_memory/memory_view.py` - `read()` applies `should_forget_stm_record()`; `list()` does not
- **Confidence:** HIGH - explicit asymmetry between read and list surfaces.
- **Target invariant:** Expired records cannot be exposed through list/search merely because point-read correctly hides them. Prefer one canonical retention-filtering boundary rather than copying policy logic across callers.

### AUDIT-20260818-MEMORY-07

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION DEFECT / FAIL-LATE
- **Status at publication:** ACCEPTED
- **Remediation block:** MEMORY-READ-MUTATION-CONSISTENCY
- **Claim falsified:** Unknown memory entry identity has explicit deterministic NOT_FOUND semantics.
- **Substance:** `UserProfileManager.update_memory_entry()` iterates `entry` but has no found/not found state. If requested `entry_id` is absent: with a non-empty profile, `entry` remains bound to the last unrelated entry and may be re-indexed when content is supplied; with an empty profile, `entry` may be undefined and cause `UnboundLocalError`. The profile is also saved despite no matching entry.
- **Evidence:**
  - `intergrax/memory/user_profile_manager.py` - `update_memory_entry()` loop without found flag; save despite no match
- **Confidence:** HIGH - loop-variable leakage and missing NOT_FOUND path.
- **Target invariant:** Never mutate/reindex an unrelated entry and never rely on loop-variable leakage.

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Memory / RAG / Context Engineering separation | NOT falsified - sound |
| Vector indexes are secondary indexes, not primary source of truth | NOT falsified |
| Canonical Tier-3 memory vector wiring requires explicit non-empty tenant | NOT falsified |
| `KnowledgeDocument` + `VectorStoreScope` + `VectorStoreRecord` on LTM/episodic vector write paths | NOT falsified |
| Direct `UserProfileManager` vector retrieval is scoped | NOT falsified |
| Single-entry soft delete removes its vector | NOT falsified |
| Retrieved LTM ids reconciled back to active primary `UserProfile` entries | NOT falsified |
| `MemoryProfile` typed, extra-forbid, basic validation for `retention_days`/`session_index_top_k` | NOT falsified |
| Architecture honestly claims P2/E3 rather than production qualification | NOT falsified |
| Procedural memory remains explicitly minimal | NOT falsified |
| Findings do not require a second Memory subsystem | NOT falsified |

## Relationship to accepted RAG / IDENTITY findings

- **MEMORY-01** is concrete downstream evidence of accepted **RAG-01** (`RAG-SCOPE-CONTRACT-INTEGRITY`) on the Memory-owned `RetrievalService` consumer path. RAG owns canonical `RetrievalService` scope authority; Memory owns supplying scope on every LTM recall invocation. Remediation coordinates - do not duplicate ownership.
- **MEMORY-05** intersects accepted **IDENTITY_TRUST** findings (`IDT-FIX-A`, `IDT-FIX-D`) on execution identity closure. Memory owns `MemoryView` boundary enforcement; Identity owns canonical trusted execution identity spine. Remediation coordinates - do not duplicate tenant identity fields.

## Root-cause remediation grouping

### MEMORY-SCOPE-AUTHORITY-INTEGRITY - one canonical tenant/task/workspace authority

**Findings:** `AUDIT-20260818-MEMORY-01`, `AUDIT-20260818-MEMORY-02`, `AUDIT-20260818-MEMORY-05`

One canonical tenant/task/workspace authority across LTM, episodic, and TaskMemory access. Coordinate with `RAG-SCOPE-CONTRACT-INTEGRITY` and IDENTITY_TRUST remediation.

### MEMORY-DURABILITY-LIFECYCLE-INTEGRITY - concurrent mutation and index lifecycle

**Findings:** `AUDIT-20260818-MEMORY-03`, `AUDIT-20260818-MEMORY-04`

Safe concurrent primary-memory mutation and deterministic secondary-index forget/delete reconciliation. Do not claim universal distributed transactions.

### MEMORY-READ-MUTATION-CONSISTENCY - uniform retention and deterministic updates

**Findings:** `AUDIT-20260818-MEMORY-06`, `AUDIT-20260818-MEMORY-07`

Uniform retention visibility across read/list surfaces and deterministic update/not-found semantics.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `628e24130de34f291a416cb1cff9397a2b327dec`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical MEM / MEM-DEPTH / MEM-VEC plan **Done** rows remain valid delivery facts - not rewritten.

## Open questions / blocked items

- 01: exact coordination surface with `RAG-SCOPE-CONTRACT-INTEGRITY` when both RAG service and Memory consumer change - operator decision deferred to remediation.
- 05: MemoryView identity closure coordinates with `IDT-FIX-A` / `IDT-FIX-D` - no parallel identity spine.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 7 (`AUDIT-20260818-MEMORY-01` … `AUDIT-20260818-MEMORY-07`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
