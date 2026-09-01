# Unified Context Lifecycle - Plan

**Status:** **CTX-UCL-6** **ACCEPTED / CLOSED** through **6D**; **CTX-UCL-CLOSEOUT-1** **ACCEPTED / CLOSED**; **TOKEN-10E-1…4** and **TOKEN-10E** **ACCEPTED / CLOSED**; **TOKEN-10E-CLOSEOUT-1** **READY_FOR_REVIEW**
**Architecture (1:1):** [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](../../architecture/UNIFIED_CONTEXT_LIFECYCLE.md)
**ADR:** [`ADR-UCL-001`](../../technical/adr/entries/2026-08-01/ADR-UCL-001.md) (**Accepted**)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Related plans:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`MEMORY.md`](MEMORY.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)

## Cursor read scope (token budget)

Open `## 6` / `### 6.1*` maintenance queues - **P0/P1** rows with Status ≠ Done only; skip closed/complete registers unless re-validating a cited gap.

---

## Current status

| Item | Status |
|------|--------|
| **CTX-UCL-ARCH-1** | **ACCEPTED / CLOSED** through R4-R1 |
| **CTX-UCL-ARCH-1-R1** | **Correction delivered** |
| **CTX-UCL-ARCH-1-R2** | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R3** | **Closed through R4** |
| **CTX-UCL-ARCH-1-R4** | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R4-R1** | **Accepted / Closed** |
| **CTX-UCL-1** | **ACCEPTED / CLOSED** through **CTX-UCL-1-R1** and **CTX-UCL-1-R2** |
| **CTX-UCL-1-R1** | **ACCEPTED / CLOSED** |
| **CTX-UCL-1-R2** | **ACCEPTED / CLOSED** |
| **CTX-UCL-2** | **ACCEPTED / CLOSED through R1** - `OptimizationArtifactRepository` port + `InMemoryOptimizationArtifactRepository` reference adapter |
| **CTX-UCL-2-R1** | **ACCEPTED / CLOSED** - monotonic bounded wait, deterministic wake proofs, provider lifecycle correction |
| **CTX-UCL-3** | **ACCEPTED / CLOSED** through R1/R2/R3 - `ContextPlan`, `SessionHistorySnapshot`, deterministic lookup inputs, canonical session provider (no last-N slicing), CE global budget |
| **CTX-UCL-4** | **ACCEPTED / CLOSED through R1** - non-recursive `MessageSequenceArtifactExecutor` on `CREATE_ARTIFACT` + `ACQUIRED` only |
| **CTX-UCL-5** | **ACCEPTED / CLOSED** through R1/R2/R3 |
| **CTX-UCL-6** | **ACCEPTED / CLOSED** through **6D** |
| **CTX-UCL-6A** | **ACCEPTED / CLOSED** through R1 |
| **CTX-UCL-6B** | **ACCEPTED / CLOSED** |
| **CTX-UCL-6C** | **ACCEPTED / CLOSED** |
| **CTX-UCL-6D** | **ACCEPTED / CLOSED** |
| **CTX-UCL-CLOSEOUT-1** | **ACCEPTED / CLOSED** - cross-domain runtime truth, documentation sync, closure proof |
| **TOKEN-10A** | Accepted / Closed |
| **TOKEN-10B** | Accepted / Closed |
| **TOKEN-10C** | Accepted / Closed |
| **TOKEN-10D** | Accepted / Closed |
| **TOKEN-10E-ARCH-1** | **Correction required / superseded** by UCL + ADR-UCL-001 |
| **TOKEN-10E-1** | Durable compaction policy, identity, eligibility and activation safety contracts - **ACCEPTED / CLOSED** |
| **TOKEN-10E-2** | Durable candidate flow using existing lookup/reservation semantics - **ACCEPTED / CLOSED** |
| **TOKEN-10E-3** | Durable receipts and rollback metadata - **ACCEPTED / CLOSED** |
| **TOKEN-10E-4** | First durable production `OptimizationArtifactRepository` adapter and durable `SessionContextRevision` activation integration - **ACCEPTED / CLOSED** |
| **TOKEN-10E** | Phase integration - policy → candidate → validation → durable storage → CAS activation - **ACCEPTED / CLOSED** |
| **TOKEN-10E-CLOSEOUT-1** | Public package-root contract freeze - **READY_FOR_REVIEW** |
| **TOKEN-10F / G / H** | See [`TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md) |

---

## Dependencies

| Dependency | Role |
|------------|------|
| **TOKEN-10A–10D** | Closed - pipeline, cache-stable assembly, cache-aware orchestration provide executor and timing gate |
| **ADR-UCL-001** | Cross-domain ownership, flows, validation ordering, reusable artifact lifecycle, internal-call boundary, single-flight creation - **Accepted** |
| **ADR-MEM-001** | Context Compiler budget semantics - superseded where UCL conflicts |
| **MEMORY** domain | `ConversationLedger`, `SessionContextRevision`, `OptimizationArtifactRepository`, `InMemoryOptimizationArtifactRepository` (CTX-UCL-2), retention |
| **CONTEXT_ENGINEERING** domain | `ContextPlan`, collection, compilation, final integrity validation, preflight, internal-call budget classification (CTX-UCL-3) |
| **TOKEN_OPTIMIZATION** feature | Typed artifact executors, receipts, protected regions, artifact creation on `CREATE_ARTIFACT` - not repository owner |
| **NEXUS_EXECUTION_FLOW** | Lifecycle coordination, lookup-before-create orchestration, reservation coordination |
| **APPLICATION_HOSTING** | Profile normalization, authorization, UX |

**Hard gate (historical - satisfied):** **TOKEN-10E-1** MUST NOT begin until **CTX-UCL-CLOSEOUT-1** is **accepted/closed**. Gate satisfied; **TOKEN-10E-1…4** and **TOKEN-10E** are **ACCEPTED / CLOSED**.

**Canonical sequence:**

```text
CTX-UCL-ARCH-1 → accepted/closed
CTX-UCL-1 → scope/guard/reservation contracts (ready for review)
CTX-UCL-2 → OptimizationArtifactRepository + InMemoryOptimizationArtifactRepository
CTX-UCL-3 → ContextPlan artifact requirements
CTX-UCL-4 → non-recursive internal executor behavior
CTX-UCL-5 → runtime single-flight integration proof
CTX-UCL-6 → legacy migration
CTX-UCL-CLOSEOUT-1 → accepted/closed
TOKEN-10E-1 … TOKEN-10E-4 → accepted/closed
TOKEN-10E → accepted/closed
TOKEN-10E-CLOSEOUT-1 → ready for review
```

---

## Task sequence

### Phase 0 - Architecture

| ID | Deliverable | Status |
|----|-------------|--------|
| **CTX-UCL-ARCH-1** | Cross-domain architecture freeze; audit table (19 mechanisms); ownership; two-mode model | **Delivered through R1/R2/R3/R4** |
| **CTX-UCL-ARCH-1-R1** | Ownership reconciliation, canonical flows, validation order, ADR-UCL-001, guardrails | **Correction delivered** |
| **CTX-UCL-ARCH-1-R2** | Document integrity and audit accuracy | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R3** | Reusable artifact lifecycle, reuse-before-create, decision outcomes, roadmap sync | **Correction delivered through R4** |
| **CTX-UCL-ARCH-1-R4** | Internal model-call boundary, single-flight artifact creation, repository delivery ownership | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R4-R1** | ADR BOM regression guard | **Accepted / Closed** |

### Phase 1 - Contracts

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-1** | `ModelCallExecutionScope`; `OptimizationExecutionGuard`; `ContextOptimizationDecision`; `ArtifactLookupKey`; `ReusableOptimizationArtifact`; `ArtifactCompatibilityResult`; `ArtifactCreationCoordinationStatus`; `ArtifactCreationReservation`; policy fields; reason codes; safe serialization | Typed contracts; deterministic equality/identity semantics; **no repository implementation; no LLM calls** |
| **CTX-UCL-2** | `OptimizationArtifactRepository` neutral interface; **`InMemoryOptimizationArtifactRepository`** reference implementation; atomic lookup; tenant-scoped artifact keys; `try_acquire_creation_reservation`; bounded lease/expiry; atomic or observably ordered validated store; reservation release/failure handling; artifact invalidation and retirement; artifact reference resolution; `SessionContextRevision` artifact references; deterministic concurrency tests | Exact compatible lookup hit; source-hash miss; tenant isolation; policy/strategy/validation incompatibility; atomic single-flight reservation (same key: exactly one reservation owner; different keys: independent reservations); lease expiry recovery; failed creation releases reservation; successful store becomes reusable; no raw content in repository telemetry |

### Phase 2 - Runtime spine

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-3** | `ContextPlan` exposes `optimization_required`, source target ranges/groups, requested artifact type, target token or budget class, allowed strategies, minimum preservation requirements; internal-call budget classification where CE participates; deterministic `ArtifactLookupKey` inputs; structured `SessionHistoryProvider`; CE as sole global budget | CE supplies deterministic lookup inputs; does not perform catalog lookup; provider delivers refs not `[-N:]` slices |
| **CTX-UCL-4** | `MessageSequenceArtifactExecutor` only on `CREATE_ARTIFACT`; internal summarizer marked `INTERNAL_OPTIMIZATION_CALL`; `OptimizationExecutionGuard` enforced; no recursive optimization of same source; no executor on `REUSE_ARTIFACT` or when reservation is `ALREADY_IN_PROGRESS`; receipt and validation tied to parent operation and lookup key | Internal summarizer does not re-enter full UCL; same source/strategy recursion blocked; `optimization_depth` violation rejected; valid internal call still runs preflight |
| **CTX-UCL-5** | Canonical Nexus UCL flow: `PRIMARY_MODEL_CALL` → CE `ContextPlan` → artifact lookup → reservation coordination → `REUSE_ARTIFACT` or `CREATE_ARTIFACT` → bounded internal call on create → final CE compile; inject `OptimizationArtifactRepository`; use `InMemoryOptimizationArtifactRepository` in reference tests | Two sequential unchanged calls: first creates, second reuses; two concurrent unchanged calls: one summarizer invocation; reservation waiter/defer behavior observable; no recursive UCL entry from internal summarizer |

### Phase 3 - Legacy migration

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-6** | Disable independent `HistoryLayer` summarizer; remove provider-level duplicate summarization; remove application-local caches; remove direct summarizer calls bypassing reservation | Verify all history-summary creation uses canonical repository and execution-scope boundary |
| **CTX-UCL-6A** | Disable HistoryLayer summarization/truncation authority; OFF remains raw compatibility load; legacy reduction strategies fail closed. | **ACCEPTED / CLOSED** |
| **CTX-UCL-6B** | Canonical SessionHistorySnapshot-only provider path; raw messages require stable revision; legacy slicing/flattening disabled. | **ACCEPTED / CLOSED** |
| **CTX-UCL-6C** | Legacy compression profile migration to canonical `ContextOptimizationPolicy` or fail-closed. | **ACCEPTED / CLOSED** |
| **CTX-UCL-6D** | Remove history summary prompt builder/YAML, application-local caches, and direct summarizer bypasses. | **ACCEPTED / CLOSED** |

### Phase 4 - Closeout

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-CLOSEOUT-1** | Cross-domain runtime + documentation sync | **ACCEPTED / CLOSED** - one canonical optimization decision point; one canonical summary creation path; internal-call recursion blocked; single-flight same-key creation proven; different-key concurrency preserved; reference repository wired; no competing summary caches |

### Phase 5 - Durable compaction (after UCL closeout)

| ID | Deliverable | Status |
|----|-------------|--------|
| **TOKEN-10E-1** | Durable policy, source identity, eligibility, and activation safety contracts extending UCL (reuses UCL repository and reservation contracts; no second repository) | **ACCEPTED / CLOSED** |
| **TOKEN-10E-2** | Durable candidate flow using existing lookup/reservation semantics | **ACCEPTED / CLOSED** |
| **TOKEN-10E-3** | Durable receipts and rollback metadata | **ACCEPTED / CLOSED** |
| **TOKEN-10E-4** | First durable production `OptimizationArtifactRepository` adapter and durable `SessionContextRevision` activation integration (implementation may live in Memory/Session packages) | **ACCEPTED / CLOSED** |
| **TOKEN-10E** | Phase integration - policy → candidate → validation → durable storage → CAS activation | **ACCEPTED / CLOSED** |
| **TOKEN-10E-CLOSEOUT-1** | Public package-root contract freeze | **READY_FOR_REVIEW** |

---

## Acceptance gates

### CTX-UCL-ARCH-1-R4 (documentation)

- [x] `PRIMARY_MODEL_CALL` and `INTERNAL_OPTIMIZATION_CALL` explicitly distinct
- [x] Every primary model call traverses full UCL
- [x] Internal optimization calls do not recursively optimize same target
- [x] `OptimizationExecutionGuard` defined with depth and ancestry invariants
- [x] Single-flight same-key creation normative via `ArtifactCreationReservation`
- [x] Content-addressed deduplication explicitly insufficient alone
- [x] Reservation acquisition and already-in-progress paths defined
- [x] Non-owner callers do not invoke summarizer
- [x] Lease expiry and failure recovery defined
- [x] CTX-UCL-2 owns `InMemoryOptimizationArtifactRepository` reference delivery
- [x] TOKEN-10E-4 owns first durable production repository adapter delivery
- [x] ADR-UCL-001 synchronized
- [x] Documentation guardrails extended
- [ ] **Human review and acceptance**

### TOKEN-10E-1 gate (satisfied)

- [x] CTX-UCL-CLOSEOUT-1 accepted/closed
- [x] Typed contracts, single-budget path, reuse-before-create, single-flight creation, MessageSequence executor, ephemeral integration, and legacy migration proven coherent
- [x] **TOKEN-10E-1…4** and **TOKEN-10E** accepted/closed

---

## Migration sequence

1. **Document** internal-call boundary and single-flight creation (CTX-UCL-ARCH-1-R4).
2. **Introduce contracts** (CTX-UCL-1).
3. **Deliver reference repository** (CTX-UCL-2).
4. **Wire ContextPlan artifact requirements** + structured session provider + CE budget (CTX-UCL-3).
5. **Add non-recursive MessageSequence executor on create only** (CTX-UCL-4).
6. **Switch single-flight integration** (CTX-UCL-5).
7. **Profile and legacy migration** (CTX-UCL-6).
8. **Closeout** (CTX-UCL-CLOSEOUT-1).
9. **TOKEN-10E** durable compaction on UCL foundation - **ACCEPTED / CLOSED** (rollback execution remains out of scope).

---

## Closeout gate

**CTX-UCL-CLOSEOUT-1** requires:

- [x] One canonical optimization decision point on every `PRIMARY_MODEL_CALL` consuming UCL-managed context
- [x] One canonical summary creation path (`MessageSequenceArtifactExecutor`, `message_sequence_summarization.v1`, `INTERNAL_OPTIMIZATION_CALL`)
- [x] Internal-call recursion blocked (`OptimizationExecutionGuard`, `OPTIMIZATION_RECURSION_BLOCKED`)
- [x] Single-flight same-key creation proven (`test_concurrent_same_key_single_flight`)
- [x] Different-key concurrency preserved (`test_different_key_concurrency_allows_two_model_calls`)
- [x] Sequential reuse-before-create proven (`test_lookup_hit_reuses_without_executor`)
- [x] Failed creation releases reservation (`test_executor_failure_releases_reservation`)
- [x] Reference `InMemoryOptimizationArtifactRepository` wired
- [x] No competing summary generator or application-local caches (cross-domain source guard)
- [x] Runtime call graph matches 19-mechanism closeout register
- [x] Documentation hub lists UCL domain pair and ADR-UCL-001
- [x] TOKEN_OPTIMIZATION §8.10 references UCL as sole lifecycle source
- [x] Public claims guardrails pass
- [ ] Independent final review and acceptance

---

<a id="protocol-v2-unified-context-lifecycle-remediation-2026-08-18"></a>

## Protocol v2 - Unified Context Lifecycle remediation (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md`](../../audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-UNIFIED-CONTEXT-LIFECYCLE-PERSIST.

> **Historical delivery boundary:** CTX-UCL-1…6D, CTX-UCL-CLOSEOUT-1, TOKEN-10E-1…4, and TOKEN-10E remain **ACCEPTED / CLOSED** historical facts. TOKEN-10E-CLOSEOUT-1 remains **READY_FOR_REVIEW**. Protocol-v2 remediation rows below address **residual defects** only - they do not reopen closed delivery rows.

<a id="ucl-governed-review-integrity-2026-08-18"></a>

### UCL-GOVERNED-REVIEW-INTEGRITY - authoritative human review for ephemeral use and durable activation

**Priority:** P0
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-01`](../../audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md), [`AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-02`](../../audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md)

**Outcome (planning only):**

- Human review governs permitted lifecycle transition - not merely repository persistence - for ephemeral `PERSIST_AFTER_HUMAN_REVIEW` and durable activation where policy requires it.
- Fail closed when review bridge unavailable; cross-link canonical Governance/UER approval authority ([`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md), [`UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md)) - no caller-controlled approval booleans.
- `MANUAL_REVIEW_THEN_COMPARE_AND_SWAP` enforced at `SessionContextRevisionActivationService` with scoped approval evidence before CAS.

<a id="ucl-durable-validation-integrity-2026-08-18"></a>

### UCL-DURABLE-VALIDATION-INTEGRITY - executable validation levels and correct artifact lifecycle ordering

**Priority:** P0/P1
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-03`](../../audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md), [`AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-05`](../../audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md)

**Outcome (planning only):**

- `minimum_validation_requirement` levels map to explicit deterministic validation stages - remove unsupported enum values if only one level is implementable.
- Repository lifecycle distinguishes executor-valid candidate from durable-policy-valid reusable artifact; rejected durable candidates are invalidated/retired with deterministic retry/replacement semantics.
- No premature `store_validated_artifact()` publication before all required durable validation passes.

<a id="ucl-revision-genesis-integrity-2026-08-18"></a>

### UCL-REVISION-GENESIS-INTEGRITY - consistent revision-zero bootstrap model

**Priority:** P1
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-04`](../../audit_results/2026-08-18/UNIFIED_CONTEXT_LIFECYCLE.md)

**Outcome (planning only):**

- One genesis revision model across `DurableCompactionSourceIdentity`, validation eligibility, `SQLiteSessionContextRevisionStore`, and CAS activation - either legal `0 → 1` first transition or explicit baseline bootstrap at session creation.

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- Historical **ACCEPTED / CLOSED** CTX-UCL and TOKEN-10E rows remain historical facts - not rewritten as remediation completion.
- **TOKEN-10E-CLOSEOUT-1** remains **READY_FOR_REVIEW** - not marked implemented/verified/closed by this remediation block.

**Recommended remediation order (prioritization, not dependency graph):** UCL-GOVERNED-REVIEW-INTEGRITY → UCL-DURABLE-VALIDATION-INTEGRITY → UCL-REVISION-GENESIS-INTEGRITY

---

## Deferred work

| Item | Notes |
|------|-------|
| Rollback execution | Memory/Session owns `ActiveContextRevisionPointer` restore - out of TOKEN-10E scope |
| Human-review UX | Application host responsibility - out of TOKEN-10E scope |
| Production/customer durable enablement | Explicit/default-off; automatic production enablement out of TOKEN-10E scope |
| Append-only ledger durable storage backend (beyond SQLite path) | Future Memory/Session backend work |
| Universal distributed single-flight guarantee | Beyond evidenced SQLite + in-memory reference adapters |
| LKW integration | After TOKEN-10 platform proof |
| Provider-specific cache mutation | **Explicitly rejected** |

---

## Next step

Independent audit of **TOKEN-10E-CLOSEOUT-1** (public contract freeze). **TOKEN-10E-1…4** and **TOKEN-10E** are **ACCEPTED / CLOSED**; durable activation is implemented; rollback execution remains outside scope.
