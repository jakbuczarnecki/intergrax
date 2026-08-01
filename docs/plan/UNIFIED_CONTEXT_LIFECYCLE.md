# Unified Context Lifecycle — Plan

**Status:** Architecture defined / ready for review (**CTX-UCL-ARCH-1**)  
**Architecture (1:1):** [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](../architecture/UNIFIED_CONTEXT_LIFECYCLE.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Related plans:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`MEMORY.md`](MEMORY.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)

---

## Current status

| Item | Status |
|------|--------|
| **CTX-UCL-ARCH-1** | **Architecture defined / ready for review** |
| **CTX-UCL-1 … CTX-UCL-6** | Not started |
| **CTX-UCL-CLOSEOUT-1** | Not started |
| **TOKEN-10A** | Accepted / Closed |
| **TOKEN-10B** | Accepted / Closed |
| **TOKEN-10C** | Accepted / Closed |
| **TOKEN-10D** | Accepted / Closed |
| **TOKEN-10E-ARCH-1** | **Correction required / superseded** by cross-domain UCL architecture |
| **TOKEN-10E implementation** | **Blocked** pending accepted UCL architecture and CTX-UCL foundation |
| **TOKEN-10F / G / H** | Planned |

---

## Dependencies

| Dependency | Role |
|------------|------|
| **TOKEN-10A–10D** | Closed — pipeline, cache-stable assembly, cache-aware orchestration provide executor and timing gate |
| **ADR-MEM-001** | Context Compiler budget semantics — extended by UCL single-budget authority |
| **MEMORY** domain | Session ledger, revision storage, retention |
| **CONTEXT_ENGINEERING** domain | `ContextPlan`, collection, compilation, preflight |
| **TOKEN_OPTIMIZATION** feature | Transformation executor, receipts, protected regions |
| **NEXUS_EXECUTION_FLOW** | Lifecycle coordination wiring |
| **APPLICATION_HOSTING** | Profile normalization, authorization, UX |

**Hard gate:** **TOKEN-10E-1** must not begin until **CTX-UCL-ARCH-1** is accepted and **CTX-UCL-1** contracts land.

---

## Task sequence

### Phase 0 — Architecture (this task)

| ID | Deliverable | Status |
|----|-------------|--------|
| **CTX-UCL-ARCH-1** | Cross-domain architecture freeze; audit table; ownership; two-mode model; roadmap decomposition | **Ready for review** |

### Phase 1 — Contracts

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-1** | `ContextOptimizationPolicy`, `OptimizationArtifact` union, `OptimizationCandidate`, policy normalization from profiles | Typed contracts in `intergrax/contracts/` or domain-appropriate package; no runtime wiring |
| **CTX-UCL-2** | `ConversationSnapshot`, `ConversationMessageRef`, `MessageGroup`, `SessionRevisionActivationRequest`; ledger/revision contracts | Contract tests; no storage backend |

### Phase 2 — Runtime spine

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-3** | Structured `SessionHistoryProvider`; CE as sole global budget; `ContextPlan` emission | Provider delivers refs not `[-N:]` slices; single budget in CE compile path |
| **CTX-UCL-4** | `MessageSequenceArtifact` pipeline + structural validators | Validators cover tool groups, IDs, recent tail; integrates with TO receipts |
| **CTX-UCL-5** | Ephemeral assembly on canonical Nexus paths | `EPHEMERAL_ASSEMBLY` replaces dormant `HistoryLayer` path; no durable mutation |

### Phase 3 — Legacy migration

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-6** | HistoryLayer, profile flags, provider slicing, bridge dedup | `semantic_compression_enabled` fail-fast or default-off; HistoryLayer removed from canonical construction |

### Phase 4 — Closeout

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-CLOSEOUT-1** | Cross-domain runtime + documentation sync | Audit table matches runtime; hub + feature docs aligned |

### Phase 5 — Durable compaction (after UCL foundation)

| ID | Deliverable | Blocked by |
|----|-------------|------------|
| **TOKEN-10E-1** | Durable compaction contracts over UCL | CTX-UCL-1, CTX-UCL-2 accepted |
| **TOKEN-10E-2** | Candidate construction over `MessageSequenceArtifact` | CTX-UCL-4 |
| **TOKEN-10E-3** | Receipts, rollback metadata, validation | TOKEN-10E-1 |
| **TOKEN-10E-4** | Revision activation + cache-lineage transition | CTX-UCL-2, TOKEN-10E-3 |
| **TOKEN-10E-CLOSEOUT-1** | Public package-root contract freeze | TOKEN-10E-4 |

---

## Acceptance gates

### CTX-UCL-ARCH-1 (documentation)

- [x] Current-state audit table with call-site verification
- [x] Domain ownership frozen
- [x] Two-mode model (`EPHEMERAL_ASSEMBLY`, `DURABLE_COMPACTION`)
- [x] Canonical data model specified (contracts not implemented)
- [x] Fail-closed reason codes frozen
- [x] Legacy migration decisions documented
- [x] TOKEN-10E blocked pending UCL
- [ ] **Human review and acceptance**

### CTX-UCL-1 gate

- Typed `ContextOptimizationPolicy` replaces scattered flags
- `OptimizationArtifact` union defined
- Profile → policy normalization specified and tested

### CTX-UCL-3 gate

- Exactly one global budget resolver per model call
- No provider pre-slicing before `ContextPlan`

### CTX-UCL-5 gate

- Ephemeral path produces model-facing context without revision mutation
- `HistoryLayer` not on canonical hot path

### TOKEN-10E-1 gate (future)

- CTX-UCL-ARCH-1 accepted
- CTX-UCL-1 and CTX-UCL-2 complete

---

## Migration sequence

1. **Document** retention vs optimization distinction (CTX-UCL-ARCH-1 — done).
2. **Introduce contracts** without changing runtime behavior (CTX-UCL-1, CTX-UCL-2).
3. **Wire structured session provider** + CE budget (CTX-UCL-3) — deprecate `messages[-N:]` slicing.
4. **Add MessageSequence validators** (CTX-UCL-4).
5. **Switch ephemeral assembly** to UCL flow (CTX-UCL-5) — retire HistoryLayer from construction.
6. **Profile migration** — `semantic_compression_enabled` fail-fast/default-off (CTX-UCL-6).
7. **Begin TOKEN-10E** durable compaction on UCL foundation.

---

## Closeout gate

**CTX-UCL-CLOSEOUT-1** requires:

- Runtime call graph matches architecture audit classifications
- No duplicate global budget authorities
- Documentation hub lists UCL domain pair
- TOKEN_OPTIMIZATION feature docs reference UCL for durable compaction
- Public claims guardrails pass

---

## Deferred work

| Item | Notes |
|------|-------|
| Append-only ledger storage backend | CTX-UCL-2+ implementation |
| CAS revision store | CTX-UCL-2 / TOKEN-10E-4 |
| LKW integration | After TOKEN-10 platform proof |
| Adaptive ranking integration | ADAPTIVE_HARNESS_INTELLIGENCE |
| Provider-specific cache mutation | **Explicitly rejected** |

---

## Next step

**Review and accept CTX-UCL-ARCH-1**, then begin **CTX-UCL-1** (canonical policy and typed artifact contracts).
