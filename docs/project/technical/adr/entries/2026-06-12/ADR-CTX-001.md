# ADR-CTX-001: Context Engineering as first-class domain and plugin engine

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-12 |
| **Deciders** | Harness platform architecture |
| **Related** | [`architecture/CONTEXT_ENGINEERING.md`](../../architecture/CONTEXT_ENGINEERING.md) · [`plan/CONTEXT_ENGINEERING.md`](../../plan/CONTEXT_ENGINEERING.md) · [`ADR-MEM-001`](../2026-06-08/ADR-MEM-001.md) · audit map §16 |

## Context

Context assembly was historically documented under [`architecture/MEMORY.md`](../../architecture/MEMORY.md) as “Layer C - Context Compiler”. Implementation lives in `intergrax/runtime/nexus/context` with partial closeout (Phase CTX, MEM-DEPTH, R-Context).

**Problem:** Context Engineering is not a memory store. It is a **cross-source orchestration engine** that consumes Memory, RAG, Tools, Orchestration priors, policies, and runtime state to produce a bounded LLM window per step. Authors need a **plugin catalog** (providers, rankers, formatters) comparable to Tools/Skills/Integrations - not only YAML profiles.

Alternatives considered:

1. **Keep CE inside MEMORY canon** - status quo; conflates persistence with compilation; blocks clear plugin roadmap.
2. **Tier-0 `intergrax/context` package only** - wrong tier; assembly is Nexus execution-critical (Tier-1).
3. **Separate domain pair `CONTEXT_ENGINEERING` + Tier-1 engine + optional Tier-0 shared contracts (chosen)** - aligns with IDEAL §16, audit layer 16, and operator requirement for Cursor-class extensibility.

## Decision

1. Introduce **22nd domain pair:** `architecture/CONTEXT_ENGINEERING.md` ↔ `plan/CONTEXT_ENGINEERING.md`.
2. **MEMORY** canon owns Layer A/B only (stores + lifecycle). Layer C canonical spec moves to **CONTEXT_ENGINEERING**.
3. Define **Context Engineering Engine** as Tier-1 Nexus subsystem with **plugin catalog** (`ContextSourceProvider`, `ContextRanker`, `ContextBudgetAllocator`, `ContextFormatter`, `ContextValidator`, `ContextEngine`).
4. Shipped default: **`DefaultNexusContextEngine`** wrapping existing `ContextCompiler`, runtime steps, and `ContextManager`.
5. **Observability** is mandatory on the engine spine: `CONTEXT_ASSEMBLED`, `CONTEXT_TRIMMED`, `CONTEXT_CANDIDATE_*`, OTel spans, structured logs - registered in OBS gates.
6. **ADR-MEM-001** remains valid for Context Compiler semantics; scope ownership for new work is **CE-\*** plan IDs.

**Rejected:**

- Merging CE into RAG (RAG = retrieval only, not global window assembly).
- Agent-side prompt concatenation as extension mechanism (Tier violation).

## Consequences

### Positive

- Clear boundary for plugin authors and audit layer 16.
- Step-aware, codebase-scale presets have a single home.
- MEMORY doc shrinks to persistence concerns - less drift.

### Negative

- Documentation migration cost; cross-links must stay current.
- Two execution paths (`ContextManager` vs turn pipeline) remain until CE-3 refactor unifies under `ContextEngine`.

### Follow-up

Execute [`plan/CONTEXT_ENGINEERING.md`](../../plan/CONTEXT_ENGINEERING.md) Phase CE-EXT waves CE-1 → CE-12.
