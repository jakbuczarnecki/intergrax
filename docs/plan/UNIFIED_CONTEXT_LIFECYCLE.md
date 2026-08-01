# Unified Context Lifecycle — Plan

**Status:** Correction delivered / ready for review (**CTX-UCL-ARCH-1-R1**)
**Architecture (1:1):** [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](../architecture/UNIFIED_CONTEXT_LIFECYCLE.md)
**ADR:** [`ADR-UCL-001`](../adr/entries/2026-08-01/ADR-UCL-001.md) (Proposed / Ready for Review)
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)
**Related plans:** [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`MEMORY.md`](MEMORY.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)

---

## Current status

| Item | Status |
|------|--------|
| **CTX-UCL-ARCH-1** | **Correction delivered through CTX-UCL-ARCH-1-R1** |
| **CTX-UCL-ARCH-1-R1** | **Ready for review** |
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
| **ADR-UCL-001** | Cross-domain ownership, flows, validation ordering — Proposed / Ready for Review |
| **ADR-MEM-001** | Context Compiler budget semantics — superseded where UCL conflicts |
| **MEMORY** domain | `ConversationLedger`, `SessionContextRevision`, retention |
| **CONTEXT_ENGINEERING** domain | `ContextPlan`, collection, compilation, final integrity validation, preflight |
| **TOKEN_OPTIMIZATION** feature | Typed artifact executors, receipts, protected regions |
| **NEXUS_EXECUTION_FLOW** | Lifecycle coordination wiring |
| **APPLICATION_HOSTING** | Profile normalization, authorization, UX |

**Hard gate:** **TOKEN-10E-1** MUST NOT begin until **CTX-UCL-CLOSEOUT-1** is **accepted/closed**.

**Canonical sequence:**

```text
CTX-UCL-ARCH-1-R1 → accepted
CTX-UCL-1 → canonical policy and typed artifact contracts
CTX-UCL-2 → ConversationLedger, ConversationSnapshot, SessionContextRevision, activation contracts
CTX-UCL-3 → structured SessionHistoryProvider and single CE budget
CTX-UCL-4 → MessageSequenceArtifact executor and structural validators
CTX-UCL-5 → canonical EPHEMERAL_ASSEMBLY runtime integration
CTX-UCL-6 → legacy migration and profile/bridge cleanup
CTX-UCL-CLOSEOUT-1 → accepted/closed
TOKEN-10E-1 → may begin
```

---

## Task sequence

### Phase 0 — Architecture

| ID | Deliverable | Status |
|----|-------------|--------|
| **CTX-UCL-ARCH-1** | Cross-domain architecture freeze; audit table (19 mechanisms); ownership; two-mode model | **Correction delivered through R1** |
| **CTX-UCL-ARCH-1-R1** | Ownership reconciliation, canonical flows, validation order, ADR-UCL-001, guardrails | **Ready for review** |

### Phase 1 — Contracts

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-1** | `ContextOptimizationPolicy`, `OptimizationArtifact` union, `OptimizationCandidate`, policy normalization from profiles | Typed contracts; no runtime wiring |
| **CTX-UCL-2** | `ConversationLedger`, `ConversationSnapshot`, `ConversationMessageRef`, `MessageGroup`, `SessionContextRevisionActivationRequest`; revision contracts | Contract tests; no storage backend |

### Phase 2 — Runtime spine

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-3** | Structured `SessionHistoryProvider`; CE as sole global budget; `ContextPlan` emission | Provider delivers refs not `[-N:]` slices |
| **CTX-UCL-4** | `MessageSequenceArtifactExecutor` + structural validators | Validators cover tool groups, IDs, recent tail |
| **CTX-UCL-5** | Ephemeral assembly on canonical Nexus paths | `EPHEMERAL_ASSEMBLY`; no durable mutation |

### Phase 3 — Legacy migration

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-6** | HistoryLayer, profile flags, provider slicing, bridge dedup | `semantic_compression_enabled` fail-fast or default-off |

### Phase 4 — Closeout

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| **CTX-UCL-CLOSEOUT-1** | Cross-domain runtime + documentation sync | Audit table matches runtime; hub aligned |

### Phase 5 — Durable compaction (after UCL closeout)

| ID | Deliverable | Blocked by |
|----|-------------|------------|
| **TOKEN-10E-1** | Durable compaction contracts over UCL | **CTX-UCL-CLOSEOUT-1** accepted/closed |
| **TOKEN-10E-2** | Candidate construction over `MessageSequenceArtifact` | CTX-UCL-4, TOKEN-10E-1 |
| **TOKEN-10E-3** | Receipts, rollback metadata, validation | TOKEN-10E-1 |
| **TOKEN-10E-4** | Revision activation request + cache-lineage (Memory/Session CAS) | CTX-UCL-2, TOKEN-10E-3 |
| **TOKEN-10E-CLOSEOUT-1** | Public package-root contract freeze | TOKEN-10E-4 |

---

## Acceptance gates

### CTX-UCL-ARCH-1-R1 (documentation)

- [x] Canonical ownership identical across UCL, TOKEN_OPTIMIZATION §8.10, ADR-UCL-001
- [x] One canonical ephemeral and durable flow
- [x] Candidate vs final model-facing validation ordering
- [x] `ConversationLedger` vs `SessionContextRevision` terminology
- [x] Audit mechanism count synchronized (19 rows)
- [x] TOKEN-10E-1 gated on CTX-UCL-CLOSEOUT-1
- [x] Documentation guardrails for ownership regression
- [ ] **Human review and acceptance**

### TOKEN-10E-1 gate (future)

- CTX-UCL-CLOSEOUT-1 accepted/closed
- Typed contracts, single-budget path, MessageSequence executor, ephemeral integration, and legacy migration proven coherent

---

## Migration sequence

1. **Document** retention vs optimization distinction (CTX-UCL-ARCH-1 — done; R1 reconciles ownership).
2. **Introduce contracts** (CTX-UCL-1, CTX-UCL-2).
3. **Wire structured session provider** + CE budget (CTX-UCL-3).
4. **Add MessageSequence executor** (CTX-UCL-4).
5. **Switch ephemeral assembly** (CTX-UCL-5).
6. **Profile migration** (CTX-UCL-6).
7. **Closeout** (CTX-UCL-CLOSEOUT-1).
8. **Begin TOKEN-10E** durable compaction on UCL foundation.

---

## Closeout gate

**CTX-UCL-CLOSEOUT-1** requires:

- Runtime call graph matches architecture audit classifications (19 mechanisms)
- No duplicate global budget authorities
- Documentation hub lists UCL domain pair and ADR-UCL-001
- TOKEN_OPTIMIZATION §8.10 references UCL as sole lifecycle source
- Public claims guardrails pass

---

## Deferred work

| Item | Notes |
|------|-------|
| Append-only ledger storage backend | CTX-UCL-2+ implementation |
| CAS revision store | CTX-UCL-2 / TOKEN-10E-4 |
| LKW integration | After TOKEN-10 platform proof |
| Provider-specific cache mutation | **Explicitly rejected** |

---

## Next step

**Review and accept CTX-UCL-ARCH-1-R1**, then begin **CTX-UCL-1** (canonical policy and typed artifact contracts).
