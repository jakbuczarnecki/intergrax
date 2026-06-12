# ADR-MEM-001: Context Compiler — global budget allocator and degradation ladder

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-08 |
| **Deciders** | Harness platform architecture |
| **Related** | [`architecture/MEMORY.md`](../architecture/MEMORY.md) · Phase MEM-DEPTH · audit map §16 |

## Context

Phase MEM closed platform memory stores (task KV, session, user LTM, wiring, hooks, observability). Context assembly remained **fragmented**: `HistoryLayer`, `UserLongtermMemoryStep`, `rag.retrieve` (catalog), and `ContextBudgetPolicy` each apply **local** limits without a single global token budget or ordered degradation trace.

Alternatives considered:

1. **Per-step budget only** — status quo; cannot guarantee never-overflow invariant.
2. **LLM-side truncation** — provider-dependent; no Harness audit trail.
3. **Unified Context Compiler (chosen)** — single compile pass before agent LLM step (`on_next_step`) with degradation ladder and pre-flight invariant.

## Decision

Introduce **`ContextCompiler`** in Tier-1 (`runtime/nexus/context/`) that:

1. Collects `messages_for_llm` as scored **`ContextCandidate`** fragments after all injection steps.
2. Enforces **`ContextDecisionProfile`** from `RuntimeConfig.context_decision_profile`.
3. Allocates a **global input token budget** derived from `llm_adapter.context_window_tokens` minus reserved output margin.
4. Applies **`DegradationLadder`** in normative order (MEMORY canon §8.2) until within budget.
5. Emits **`CONTEXT_TRIMMED`** / diagnostics with `degradation_step` per apply.
6. Runs **`verify_context_preflight`** immediately before every core LLM call.

**Rejected:**

- **Char-cut only** — retained as last-resort step inside ladder, not happy path.
- **Compiler inside Tier-0** — violates Tier-1 ownership of context assembly.

## Consequences

### Positive

- Never-overflow acceptance gate (MEM-DEPTH-1.6) becomes testable.
- `ContextDecisionProfile` enforced end-to-end.
- Memory Layer audit can reach **L3+** without Mem0 SaaS product scope.

### Negative

- Additional pipeline step (`CompileContextStep`) on every chat turn.
- Token estimates use adapter window + optional tokenizer — not byte-perfect for all providers.

## Compliance

- Tier boundaries preserved — compiler in Tier-1; agents unchanged.
- Provenance via existing `CONTEXT_*` runtime events.
- Linked from [`architecture/MEMORY.md`](../architecture/MEMORY.md) §7–§8 and [`plan/MEMORY.md`](../plan/MEMORY.md) Phase MEM-DEPTH.

## Implementation notes

- `runtime/nexus/context/context_compiler.py`, `degradation_ladder.py`, `context_preflight.py`
- Context assembly via CE providers + `context_preflight.py` before LLM calls in `on_next_step`
- `context_budget.py` — tokenizer-aware trim helper
- Verification: `pytest -m gate -q`; `tests/unit/runtime/nexus/context/test_context_compiler.py`
