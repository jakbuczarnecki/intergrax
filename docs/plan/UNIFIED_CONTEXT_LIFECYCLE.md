# Unified Context Lifecycle — Plan

**Status:** **CTX-UCL-3** correction delivered through CTX-UCL-3-R1; **CTX-UCL-2** accepted/closed through R1; **CTX-UCL-1** accepted/closed through R1/R2
**Architecture (1:1):** [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](../architecture/UNIFIED_CONTEXT_LIFECYCLE.md)
**ADR:** [`ADR-UCL-001`](../adr/entries/2026-08-01/ADR-UCL-001.md) (**Accepted**)
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)
**Related plans:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`MEMORY.md`](MEMORY.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)

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
| **CTX-UCL-2** | **ACCEPTED / CLOSED through R1** — `OptimizationArtifactRepository` port + `InMemoryOptimizationArtifactRepository` reference adapter |
| **CTX-UCL-2-R1** | **ACCEPTED / CLOSED** — monotonic bounded wait, deterministic wake proofs, provider lifecycle correction |
| **CTX-UCL-3** | **Correction delivered through CTX-UCL-3-R1** — **READY_FOR_REVIEW** — `ContextPlan`, `SessionHistorySnapshot`, deterministic lookup inputs, canonical session provider (no last-N slicing), CE global budget |
| **CTX-UCL-4 … CTX-UCL-6** | Not started / blocked pending **CTX-UCL-3** acceptance |
| **CTX-UCL-CLOSEOUT-1** | Not started |
| **TOKEN-10A** | Accepted / Closed |
| **TOKEN-10B** | Accepted / Closed |
| **TOKEN-10C** | Accepted / Closed |
| **TOKEN-10D** | Accepted / Closed |
| **TOKEN-10E-ARCH-1** | **Correction required / superseded** by UCL + ADR-UCL-001 |
| **TOKEN-10E-1 … TOKEN-10E-4** | **Blocked** pending **CTX-UCL-CLOSEOUT-1** accepted/closed |
| **TOKEN-10F / G / H** | Planned |

---

## Dependencies

| Dependency | Role |
|------------|------|
| **TOKEN-10A–10D** | Closed — pipeline, cache-stable assembly, cache-aware orchestration provide executor and timing gate |
| **ADR-UCL-001** | Cross-domain ownership, flows, validation ordering, reusable artifact lifecycle, internal-call boundary, single-flight creation — **Accepted** |
| **ADR-MEM-001** | Context Compiler budget semantics — superseded where UCL conflicts |
| **MEMORY** domain | `ConversationLedger`, `SessionContextRevision`, `OptimizationArtifactRepository`, `InMemoryOptimizationArtifactRepository` (CTX-UCL-2), retention |
| **CONTEXT_ENGINEERING** domain | `ContextPlan`, collection, compilation, final integrity validation, preflight, internal-call budget classification (CTX-UCL-3) |
| **TOKEN_OPTIMIZATION** feature | Typed artifact executors, receipts, protected regions, artifact creation on `CREATE_ARTIFACT` — not repository owner |
| **NEXUS_EXECUTION_FLOW** | Lifecycle coordination, lookup-before-create orchestration, reservation coordination |
| **APPLICATION_HOSTING** | Profile normalization, authorization, UX |

**Hard gate:** **TOKEN-10E-1** MUST NOT begin until **CTX-UCL-CLOSEOUT-1** is **accepted/closed**.

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
TOKEN-10E-1 → may begin
```

---

## Task sequence

### Phase 0 — Architecture

| ID | Deliverable | Status |
|----|-------------|--------|
| **CTX-UCL-ARCH-1** | Cross-domain architecture freeze; audit table (19 mechanisms); ownership; two-mode model | **Delivered through R1/R2/R3/R4** |
| **CTX-UCL-ARCH-1-R1** | Ownership reconciliation, canonical flows, validation order, ADR-UCL-001, guardrails | **Correction delivered** |
| **CTX-UCL-ARCH-1-R2** | Document integrity and audit accuracy | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R3** | Reusable artifact lifecycle, reuse-before-create, decision outcomes, roadmap sync | **Correction delivered through R4** |
| **CTX-UCL-ARCH-1-R4** | Internal model-call boundary, single-flight artifact creation, repository delivery ownership | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R4-R1** | ADR BOM regression guard | **Accepted / Closed** |

### Phase 1 — Contracts

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-1** | `ModelCallExecutionScope`; `OptimizationExecutionGuard`; `ContextOptimizationDecision`; `ArtifactLookupKey`; `ReusableOptimizationArtifact`; `ArtifactCompatibilityResult`; `ArtifactCreationCoordinationStatus`; `ArtifactCreationReservation`; policy fields; reason codes; safe serialization | Typed contracts; deterministic equality/identity semantics; **no repository implementation; no LLM calls** |
| **CTX-UCL-2** | `OptimizationArtifactRepository` neutral interface; **`InMemoryOptimizationArtifactRepository`** reference implementation; atomic lookup; tenant-scoped artifact keys; `try_acquire_creation_reservation`; bounded lease/expiry; atomic or observably ordered validated store; reservation release/failure handling; artifact invalidation and retirement; artifact reference resolution; `SessionContextRevision` artifact references; deterministic concurrency tests | Exact compatible lookup hit; source-hash miss; tenant isolation; policy/strategy/validation incompatibility; atomic single-flight reservation (same key: exactly one reservation owner; different keys: independent reservations); lease expiry recovery; failed creation releases reservation; successful store becomes reusable; no raw content in repository telemetry |

### Phase 2 — Runtime spine

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-3** | `ContextPlan` exposes `optimization_required`, source target ranges/groups, requested artifact type, target token or budget class, allowed strategies, minimum preservation requirements; internal-call budget classification where CE participates; deterministic `ArtifactLookupKey` inputs; structured `SessionHistoryProvider`; CE as sole global budget | CE supplies deterministic lookup inputs; does not perform catalog lookup; provider delivers refs not `[-N:]` slices |
| **CTX-UCL-4** | `MessageSequenceArtifactExecutor` only on `CREATE_ARTIFACT`; internal summarizer marked `INTERNAL_OPTIMIZATION_CALL`; `OptimizationExecutionGuard` enforced; no recursive optimization of same source; no executor on `REUSE_ARTIFACT` or when reservation is `ALREADY_IN_PROGRESS`; receipt and validation tied to parent operation and lookup key | Internal summarizer does not re-enter full UCL; same source/strategy recursion blocked; `optimization_depth` violation rejected; valid internal call still runs preflight |
| **CTX-UCL-5** | Canonical Nexus UCL flow: `PRIMARY_MODEL_CALL` → CE `ContextPlan` → artifact lookup → reservation coordination → `REUSE_ARTIFACT` or `CREATE_ARTIFACT` → bounded internal call on create → final CE compile; inject `OptimizationArtifactRepository`; use `InMemoryOptimizationArtifactRepository` in reference tests | Two sequential unchanged calls: first creates, second reuses; two concurrent unchanged calls: one summarizer invocation; reservation waiter/defer behavior observable; no recursive UCL entry from internal summarizer |

### Phase 3 — Legacy migration

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-6** | Disable independent `HistoryLayer` summarizer; remove provider-level duplicate summarization; remove application-local caches; remove direct summarizer calls bypassing reservation | Verify all history-summary creation uses canonical repository and execution-scope boundary |

### Phase 4 — Closeout

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-CLOSEOUT-1** | Cross-domain runtime + documentation sync | One canonical optimization decision point; one canonical summary creation path; internal-call recursion blocked; single-flight same-key creation proven; different-key concurrency preserved; reference repository wired; no ambiguous delivery item; no competing summary caches; identical sequential source reuses; identical concurrent source invokes summarizer once |

### Phase 5 — Durable compaction (after UCL closeout)

| ID | Deliverable | Blocked by |
|----|-------------|------------|
| **TOKEN-10E-1** | Durable policies and contracts extending UCL (reuses UCL repository and reservation contracts; no second repository) | **CTX-UCL-CLOSEOUT-1** accepted/closed |
| **TOKEN-10E-2** | Durable candidate flow using existing lookup/reservation semantics | CTX-UCL-4, TOKEN-10E-1 |
| **TOKEN-10E-3** | Durable receipts and rollback metadata | TOKEN-10E-1 |
| **TOKEN-10E-4** | First durable production `OptimizationArtifactRepository` adapter and durable `SessionContextRevision` activation integration (implementation may live in Memory/Session packages) | CTX-UCL-2, TOKEN-10E-3 |
| **TOKEN-10E-CLOSEOUT-1** | Public package-root contract freeze | TOKEN-10E-4 |

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

### TOKEN-10E-1 gate (future)

- CTX-UCL-CLOSEOUT-1 accepted/closed
- Typed contracts, single-budget path, reuse-before-create, single-flight creation, MessageSequence executor, ephemeral integration, and legacy migration proven coherent

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
9. **Begin TOKEN-10E** durable compaction on UCL foundation.

---

## Closeout gate

**CTX-UCL-CLOSEOUT-1** requires:

- One canonical optimization decision point on every model call
- One canonical summary creation path
- Internal-call recursion blocked
- Single-flight same-key creation proven in runtime tests
- Different-key concurrency preserved
- Reference `InMemoryOptimizationArtifactRepository` wired
- No ambiguous `CTX-UCL-2+` delivery item
- No competing summary generator or application-local caches
- Identical sequential source reuses artifact
- Identical concurrent source invokes summarizer once
- Runtime call graph matches architecture audit classifications (19 mechanisms)
- Documentation hub lists UCL domain pair and ADR-UCL-001
- TOKEN_OPTIMIZATION §8.10 references UCL as sole lifecycle source
- Public claims guardrails pass

---

## Deferred work

| Item | Notes |
|------|-------|
| Append-only ledger durable storage backend | **TOKEN-10E-4** (durable production adapter) |
| Durable production Optimization Artifact Catalog backend | **TOKEN-10E-4** (delivery coordination; Memory/Session owns contracts) |
| CAS revision store | **CTX-UCL-2** (reference in-memory) / **TOKEN-10E-4** (durable) |
| LKW integration | After TOKEN-10 platform proof |
| Provider-specific cache mutation | **Explicitly rejected** |

---

## Next step

**Independent review of CTX-UCL-2** (configurable `OptimizationArtifactRepository` port and `InMemoryOptimizationArtifactRepository` reference adapter). After acceptance: **CTX-UCL-3** — ContextPlan artifact requirements and deterministic lookup inputs.

**Repository boundary (CTX-UCL-2):** `OptimizationArtifactRepository` is the configurable port. Application host selects and injects an adapter. `InMemoryOptimizationArtifactRepository` is reference-only, process-local, and non-durable. There is no implicit in-memory production fallback. **CTX-UCL-5** will inject the repository into Nexus/UCL. **TOKEN-10E-4** will deliver the first durable production adapter.
