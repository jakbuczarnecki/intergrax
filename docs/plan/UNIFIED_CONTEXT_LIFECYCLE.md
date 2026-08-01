# Unified Context Lifecycle — Plan

**Status:** Ready for review (**CTX-UCL-ARCH-1-R3**)
**Architecture (1:1):** [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](../architecture/UNIFIED_CONTEXT_LIFECYCLE.md)
**ADR:** [`ADR-UCL-001`](../adr/entries/2026-08-01/ADR-UCL-001.md) (Proposed / Ready for Review)
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)
**Related plans:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`MEMORY.md`](MEMORY.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)

---

## Current status

| Item | Status |
|------|--------|
| **CTX-UCL-ARCH-1** | **Architecture delivered through R1/R2/R3** |
| **CTX-UCL-ARCH-1-R1** | **Correction delivered** |
| **CTX-UCL-ARCH-1-R2** | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R3** | **Ready for review** |
| **CTX-UCL-1 … CTX-UCL-6** | Not started |
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
| **ADR-UCL-001** | Cross-domain ownership, flows, validation ordering, reusable artifact lifecycle — Proposed / Ready for Review |
| **ADR-MEM-001** | Context Compiler budget semantics — superseded where UCL conflicts |
| **MEMORY** domain | `ConversationLedger`, `SessionContextRevision`, Optimization Artifact Catalog, retention |
| **CONTEXT_ENGINEERING** domain | `ContextPlan`, collection, compilation, final integrity validation, preflight |
| **TOKEN_OPTIMIZATION** feature | Typed artifact executors, receipts, protected regions, artifact creation on `CREATE_ARTIFACT` |
| **NEXUS_EXECUTION_FLOW** | Lifecycle coordination, lookup-before-create orchestration |
| **APPLICATION_HOSTING** | Profile normalization, authorization, UX |

**Hard gate:** **TOKEN-10E-1** MUST NOT begin until **CTX-UCL-CLOSEOUT-1** is **accepted/closed**.

**Canonical sequence:**

```text
CTX-UCL-ARCH-1-R3 → accepted
CTX-UCL-1 → canonical policy, decision and artifact identity contracts
CTX-UCL-2 → artifact catalog and revision reference contracts
CTX-UCL-3 → ContextPlan artifact requirements
CTX-UCL-4 → lookup-miss artifact creation
CTX-UCL-5 → runtime reuse-before-create integration
CTX-UCL-6 → legacy migration
CTX-UCL-CLOSEOUT-1 → accepted/closed
TOKEN-10E-1 → may begin
```

---

## Task sequence

### Phase 0 — Architecture

| ID | Deliverable | Status |
|----|-------------|--------|
| **CTX-UCL-ARCH-1** | Cross-domain architecture freeze; audit table (19 mechanisms); ownership; two-mode model | **Delivered through R1/R2/R3** |
| **CTX-UCL-ARCH-1-R1** | Ownership reconciliation, canonical flows, validation order, ADR-UCL-001, guardrails | **Correction delivered** |
| **CTX-UCL-ARCH-1-R2** | Document integrity and audit accuracy | **Accepted / Closed** |
| **CTX-UCL-ARCH-1-R3** | Reusable artifact lifecycle, reuse-before-create, decision outcomes, roadmap sync | **Ready for review** |

### Phase 1 — Contracts

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-1** | `ContextOptimizationDecision` enum (`NO_OP`, `SELECT_ONLY`, `REUSE_ARTIFACT`, `CREATE_ARTIFACT`, `POLICY_BLOCKED`, `FAIL_CLOSED`); `ArtifactLookupKey`; `ReusableOptimizationArtifact` metadata contract; artifact compatibility result contract; policy fields controlling artifact reuse, persistence, LLM summarization, lossy strategies, administrative refresh | Typed contracts; deterministic equality/identity semantics; safe serialization; no storage backend; no LLM calls |
| **CTX-UCL-2** | `OptimizationArtifactRepository` or equivalent neutral interface; lookup by `ArtifactLookupKey`; store validated reusable artifact; mark invalidated or retired; resolve artifact reference; `SessionContextRevision` references reusable artifact IDs/hashes; source-range and source-hash lineage; tenant/session isolation | Contract tests: exact compatible hit; source-hash miss; policy-version incompatibility; validation-version incompatibility; tenant isolation; revision artifact references; no storage backend unless explicitly in scope |

### Phase 2 — Runtime spine

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-3** | `ContextPlan` exposes `optimization_required`, source target ranges/groups, requested artifact type, target token or budget class, allowed strategies, minimum preservation requirements; structured `SessionHistoryProvider`; CE as sole global budget | CE supplies deterministic lookup inputs; does not perform catalog lookup; provider delivers refs not `[-N:]` slices |
| **CTX-UCL-4** | `MessageSequenceArtifactExecutor` invoked only after `CREATE_ARTIFACT` decision or explicit approved refresh; no executor invocation on `REUSE_ARTIFACT`; creation emits compatibility metadata, validation result and receipt | Tests prove identical source does not trigger duplicate LLM summarization when valid artifact supplied |
| **CTX-UCL-5** | Canonical Nexus UCL flow: `ContextPlan` → `ArtifactLookupKey` → lookup → `REUSE_ARTIFACT` or `CREATE_ARTIFACT` → final CE compile; every canonical model call traverses decision point; valid lookup hit bypasses TO transformation; ephemeral artifact persistence follows policy; observability distinguishes lookup hit, lookup miss, transform execution | Two consecutive model calls over unchanged source: first may `CREATE_ARTIFACT`; second must `REUSE_ARTIFACT`; second must not invoke summarizer |

### Phase 3 — Legacy migration

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-6** | Remove or disable independent `HistoryLayer` summarization; remove provider-level summary regeneration; remove application-local summary caches; map legacy strategy config to canonical policy | Verify no duplicate summarizer call path remains |

### Phase 4 — Closeout

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-CLOSEOUT-1** | Cross-domain runtime + documentation sync | One canonical optimization decision point; no competing summary generator; reuse-before-create proven; identical source does not regenerate summary; artifact invalidation proven; tenant isolation proven; artifact vs provider cache terminology verified; runtime call graph matches architecture |

### Phase 5 — Durable compaction (after UCL closeout)

| ID | Deliverable | Blocked by |
|----|-------------|------------|
| **TOKEN-10E-1** | Durable compaction contracts over UCL (extends UCL artifact/revision contracts; no duplicate repository) | **CTX-UCL-CLOSEOUT-1** accepted/closed |
| **TOKEN-10E-2** | Durable candidate flow — artifact lookup first; reuse existing `MessageSequenceArtifact`; create only on miss; revision references selected artifact | CTX-UCL-4, TOKEN-10E-1 |
| **TOKEN-10E-3** | Receipts indicate reused/created/invalidated artifact; no LLM on reuse | TOKEN-10E-1 |
| **TOKEN-10E-4** | Activation on revision references; CAS activation and rollback reuse immutable artifact references; no artifact regeneration on activation | CTX-UCL-2, TOKEN-10E-3 |
| **TOKEN-10E-CLOSEOUT-1** | Public package-root contract freeze | TOKEN-10E-4 |

---

## Acceptance gates

### CTX-UCL-ARCH-1-R3 (documentation)

- [x] Reuse-before-create normative in UCL architecture
- [x] `ContextOptimizationDecision` outcomes defined (`NO_OP` … `FAIL_CLOSED`)
- [x] `ArtifactLookupKey` and `ReusableOptimizationArtifact` contracts defined
- [x] Artifact invalidation rules and closed-range stability
- [x] Ephemeral artifact persistence distinction
- [x] Durable compaction artifact lookup first
- [x] CTX-UCL-1…6 responsibilities synchronized
- [x] TOKEN-10E-1…4 responsibilities synchronized
- [x] ADR-UCL-001 reusable-artifact decision section
- [x] Documentation guardrails for reuse-before-create
- [ ] **Human review and acceptance**

### TOKEN-10E-1 gate (future)

- CTX-UCL-CLOSEOUT-1 accepted/closed
- Typed contracts, single-budget path, reuse-before-create, MessageSequence executor, ephemeral integration, and legacy migration proven coherent

---

## Migration sequence

1. **Document** retention vs optimization and reusable artifact lifecycle (CTX-UCL-ARCH-1-R3).
2. **Introduce contracts** (CTX-UCL-1, CTX-UCL-2).
3. **Wire ContextPlan artifact requirements** + structured session provider + CE budget (CTX-UCL-3).
4. **Add MessageSequence executor on create only** (CTX-UCL-4).
5. **Switch reuse-before-create integration** (CTX-UCL-5).
6. **Profile and legacy migration** (CTX-UCL-6).
7. **Closeout** (CTX-UCL-CLOSEOUT-1).
8. **Begin TOKEN-10E** durable compaction on UCL foundation.

---

## Closeout gate

**CTX-UCL-CLOSEOUT-1** requires:

- One canonical optimization decision point on every model call
- No competing summary generator
- Reuse-before-create proven in runtime tests
- Identical compatible source does not regenerate summary
- Artifact invalidation and tenant isolation proven
- Artifact catalog terminology distinct from provider cache
- Runtime call graph matches architecture audit classifications (19 mechanisms)
- Documentation hub lists UCL domain pair and ADR-UCL-001
- TOKEN_OPTIMIZATION §8.10 references UCL as sole lifecycle source
- Public claims guardrails pass

---

## Deferred work

| Item | Notes |
|------|-------|
| Append-only ledger storage backend | CTX-UCL-2+ implementation |
| Optimization Artifact Catalog storage backend | CTX-UCL-2+ implementation |
| CAS revision store | CTX-UCL-2 / TOKEN-10E-4 |
| LKW integration | After TOKEN-10 platform proof |
| Provider-specific cache mutation | **Explicitly rejected** |

---

## Next step

**Review and accept CTX-UCL-ARCH-1-R3**, then begin **CTX-UCL-1** (canonical policy, decision and artifact identity contracts).
