# Memory — Implementation Plan

**Architecture (1:1):** [`architecture/MEMORY.md`](../../architecture/MEMORY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

**Cross-plan — Agent layer (ACP):** Per-agent `memory_view` and `memory_scope` (user vs org §30.9) resolve in `merge_environment` — [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 2** (`ACP-DX-2`). Agent session state (`AcpSessionState`) is separate from LTM namespaces; do not store secrets in `acp.state.v1` (architecture §25.2).

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). MEMORY owns persistent summary compression only where staging, validation, compression receipts, and rollback metadata exist.

**Last updated:** 2026-08-05 — **LCI-4D READY_FOR_REVIEW** (session-turn and user-profile vector indexing use `KnowledgeDocument`, explicit scope and native vector-store records).

**LCI-4D decision:** Memory indexing preserves stable identity, tenant, namespace, workspace, provenance and user metadata without using user metadata as system scope transport. Session and profile vector writes use `VectorStoreRecord` with `VectorStoreScope`; LCI-5A remains planned for the next native text-loader stream.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (MEMORY plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/MEMORY_appendices.md`](plan/satellites/MEMORY_appendices.md) · [`plan/satellites/MEMORY_audit_history.md`](plan/satellites/MEMORY_audit_history.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + row `TOKEN-MEM-1`; inspect only memory summary/consolidation/write paths required for staging/rollback.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/MEMORY.md`](../../architecture/MEMORY.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/MEMORY.md`](../../technical/guides/audit_slices/MEMORY.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/MEMORY_appendices.md`](plan/satellites/MEMORY_appendices.md) | appendices |
| [`plan/satellites/MEMORY_audit_history.md`](plan/satellites/MEMORY_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-MEM — MemorySummaryCompressor (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)  
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)  
**Priority:** P2 after TOKEN-UER-1 and preferably after TOKEN-CE-1 receipt path  
**Delivery rule:** one `TOKEN-MEM-*` row per PR; no live overwrite before validation.

**First implementation slice:** **TOKEN-5A** (feature plan §TOKEN-5A) — helper-only `MemorySummaryCompressor`. TOKEN-MEM-1 remains helper-only unless explicitly expanded in a later row. Live memory-store overwrite, compaction jobs, and runtime wiring are out of scope for TOKEN-MEM-1 / TOKEN-5A.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-MEM-1** | Code | P2 | **Done / Closed** | `intergrax/memory/summary_compressor.py` — conservative helper-only `MemorySummaryCompressor` with staged candidate/result model, protected-region validation, compression receipt, rollback metadata, optional `token_counter`, optional `semantic_validation_hook` interface, deterministic light/structural compression only, and benchmark-ready result fields aligned with TOKEN-5A / future TOKEN-6B / LKW-PF6 | Live source never overwritten before validation; failed compression cannot corrupt persistent memory; rollback metadata required on every result; original/compressed hashes stored; memory compression opt-in by policy/profile; no user facts/dates/IDs/policy text silently lost; benchmark-ready result fields (`source_type`, `strategy`, `original_hash`, `optimized_hash`, `original_tokens`, `optimized_tokens`, `saved_tokens`, `saved_ratio`, `validation_status`, `fallback_status`, `receipt`/`receipt_ref`, `rollback_metadata`, `semantic_validation_status` when hook used); `uv run pytest tests/unit/memory/ -q` — **closeout:** helper-only compressor, staged result/rollback metadata, protected-region validation, compression receipts, optional `token_counter` and `semantic_validation_hook`, benchmark-ready result shape; no live memory-store overwrite, vector index mutation, embedding regeneration, LLM rewriting, HOS/runtime wiring, or LKW proof execution |

**Safety rules (TOKEN-MEM-1 / TOKEN-5A):**

- live source must never be overwritten before validation,
- failed compression must not corrupt persistent memory,
- rollback metadata is required on every compression result,
- memory compression is opt-in by policy/profile,
- user facts, dates, IDs, policy text, and protected terms must not be silently lost,
- no compression of primary memory store records in the helper-only slice,
- no vector index mutation without primary-store source of truth,
- no lossy compression of legal/security/policy text unless explicitly allowed by policy.

**Refinement TOKEN-5A-R — unsafe lossy truncation guard:**

- `max_summary_chars` is treated as **lossy** compression
- no truncation under default `allow_lossy=False` policy
- lossy truncation requires explicit `allow_lossy=True` **and** `semantic_validation_hook` acceptance
- no LLM-as-a-Judge implementation was added in TOKEN-5A-R
- no live memory-store wiring was added in TOKEN-5A-R

**Explicit exclusions:** no live memory-store overwrite, no automatic memory compaction job, no vector index mutation, no embedding regeneration, no LLM/model-based semantic rewriting, no full LLM-as-a-Judge eval engine (belongs to TOKEN-6B / regression/evals), no runtime hot-path wiring, no HOS emission, no observability exporter wiring, no token regression benchmark runner, no LKW proof execution in TOKEN-MEM-1.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §7, §16 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** — memory-routed rows 14.1–14.2, 15.1–15.3, 16.1–16.2 closed in master register (16.x implemented in CONTEXT_ENGINEERING)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-14.1 | §14 RAG | Graph RAG as default production retrieval profile | P1 | **Done** |
| AUDIT-IDEAL-14.2 | §14 RAG | Retrieval poisoning defense live on product hosts | P1 | **Done** |
| AUDIT-IDEAL-15.1 | §15 Memory | Org memory 2.5 (organizational LTM scope) | **P0** | **Done** |
| AUDIT-IDEAL-15.2 | §15 Memory | Episodic / semantic / procedural taxonomy (`MemoryKind` uplift) | P1 | **Done** |
| AUDIT-IDEAL-15.3 | §15 Memory | Entity graph memory ship (MEM-DEPTH-5.1 beyond RFC) | P2 | **Done** |
| AUDIT-IDEAL-16.1 | §16 Context | Online context drift monitoring + alerts | P1 | **Done** — owner [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §11 |
| AUDIT-IDEAL-16.2 | §16 Context | Semantic compression in production profiles | P2 | **Done** — owner CE §11 (`semantic_compression_enabled`) |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---
