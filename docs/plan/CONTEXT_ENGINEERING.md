## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/CONTEXT_ENGINEERING_audit_history.md`](plan/satellites/CONTEXT_ENGINEERING_audit_history.md) | audit history |
| [`plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md`](plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


# Context Engineering — Implementation Plan

**Architecture (1:1):** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**ADR:** [`ADR-CTX-001`](../adr/entries/2026-06-12/ADR-CTX-001.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

> **CTX-UCL-3 (2026-08-02):** `ContextPlan` contracts, `SessionHistorySnapshot`, deterministic `ContextArtifactLookupInputs`, canonical session provider without pre-plan last-N slicing — **READY_FOR_REVIEW**; no repository lookup, artifact executor, or LLM wiring yet.

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). CE owns `ContextPackOptimizer`, source-aware context compression, post-compression token recalculation, receipt references in provenance/metadata, and fallback to original fragments on validation failure.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CONTEXT_ENGINEERING plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites/) satellites on demand. **On demand (one max):** [`plan/satellites/CONTEXT_ENGINEERING_audit_history.md`](plan/satellites/CONTEXT_ENGINEERING_audit_history.md) · [`plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md`](plan/satellites/CONTEXT_ENGINEERING_embedded_detail.md). §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + rows `TOKEN-CE-1` / `TOKEN-CE-2`; inspect existing `ContextCompiler`, `DefaultNexusContextEngine`, `ContextBudgetPolicy`, `DegradationLadder`, and adapter-token preflight only as needed.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/CONTEXT_ENGINEERING.md`](../guides/audit_slices/CONTEXT_ENGINEERING.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Status summary (2026-06-17)

| Phase | Scope | Status |
|-------|-------|--------|
| **CTX** (control plane closeout) | `context_runtime_bridge`, `context_wiring`, Appendix L | **Done** (2026-06-02) |
| **R-Context** | Budget API, `CONTEXT_*` events (graph path) | **Done** |
| **MEM-DEPTH-1.*** | `ContextCompiler`, `DegradationLadder`, preflight **modules** | **Done** — library + tests; **hot-path wiring = CE-3.9** (post ACP-CLOSE) |
| **CE-DOC** | Domain split + architecture + plan + FAUDIT refresh | **Done** (CE-DOC.7 closes 2026-06-12 audit) |
| **CE-EXT** | Plugin engine + hot-path compiler + step-aware + codebase preset | **Done** (S0–S12, 2026-06-12) |
| **CE-DOC.8** | Architecture ↔ implementation sync post CE-EXT | **Done** (2026-06-12) |
| **CE-ALIGN** | Post-audit implementation alignment (GAP-CTX-15..19) | **Done** (A0–A6, 2026-06-12) |
| **CE-PROV-WIRE** | Builtin stub providers → legacy collectors on `assemble()` path | **Done** (B1–B4, 2026-06-14) |
| **CE-DOC.9** | FAUDIT 2026-06-12 deep audit — GAP-CTX-15..19 + CE-ALIGN sprint register | **Done** (2026-06-12) |
| **CE-DOC.10** | CE-ALIGN closeout audit + architecture sync | **Done** (2026-06-12) |
| **CE-HANDLE-FILL** | RuntimeState → provider metadata sync on nexus context steps | **Done** (2026-06-14) |
| **P2-ARCH-05** | Add context path unification rules (approved / disallowed / transitional paths + Cursor checklist) | **Done** (2026-06-20) |

**As-built maturity:** L3+ engine / L3 control plane — CE-PROV-WIRE closed GAP-CTX-20; Layer Completion iteration III (2026-06-17) confirms **Architecturally Mature** — no P0/P1; **Full Harness LC** (2026-06-17); see architecture §3.

**Delivery rule:** One **CE-*** ID per PR → update master table + gap register → `pytest -m gate` + domain CI scripts green.

---

## Phase TOKEN-CE — ContextPackOptimizer for token optimization

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)  
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)  
**Priority:** P1 after `TOKEN-UER-1` contracts and receipts  
**Delivery rule:** one `TOKEN-CE-*` row per PR; extend existing CE compiler/engine, do not build a second context compiler.

**TOKEN-4A note:** helper-only `ContextPackOptimizer` lives in `intergrax/runtime/token_optimization/context_pack.py`. It does not wire into `ContextCompiler`, `DefaultNexusContextEngine`, RAG retrieval, prompt assembly, or context runtime behavior yet. Runtime wiring remains future **TOKEN-CE-1B** work.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-CE-1A** | Code | P1 | Done / Closed | `intergrax/runtime/token_optimization/context_pack.py` — helper-only `ContextPackOptimizer` with deterministic light/structural compaction, protected-region validation, receipts, and optional `token_counter` measurement | Mandatory/policy fragments preserved; fragment order/IDs/source/provenance preserved; validation failure falls back to original pack; no tokenizer/model/runtime wiring; `uv run pytest tests/unit/runtime/token_optimization/test_context_pack.py -q` |
| **TOKEN-CE-1B** | Code | P1 | Planned | Runtime wiring into `ContextCompiler` / `DefaultNexusContextEngine` / context assembly with source-aware light/structural compression stage after ranking/budgeting and before format/preflight | Uses existing adapter token counter path; receipt refs attach to provenance/metadata; `uv run pytest tests/unit/runtime/nexus/context/ -q`; `uv run python scripts/check_compression_receipts.py` |
| **TOKEN-CE-2** | Test/Gate | P1 | Planned | Context token regression fixtures proving savings without quality/provenance loss | Benchmark fixture shows lower assembled tokens for repeated/verbose fragments; context quality gate remains green; no semantic compression enabled yet; `uv run python scripts/maintenance/check_context_preflight_uses_adapter_tokens.py`; `uv run python scripts/check_token_regression_benchmarks.py` |

**Explicit exclusions:** no semantic compression in first CE slice, no replacement of `ContextCompiler`, no direct agent prompt assembly, no bypass of `ContextBudgetPolicy` / `DegradationLadder` / adapter-token preflight.

---
