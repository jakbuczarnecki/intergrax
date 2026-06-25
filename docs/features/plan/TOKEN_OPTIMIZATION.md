<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Multi-layer Feature Plan

**Status:** Planned  
**Feature architecture (1:1):** [`../architecture/TOKEN_OPTIMIZATION.md`](../architecture/TOKEN_OPTIMIZATION.md)  
**Source audit instruction:** [`../../audit/TOKEN_OPTIMIZATION.md`](../../audit/TOKEN_OPTIMIZATION.md)  
**Primary anchor domain:** `CONTEXT_ENGINEERING`  
**Related domains:** `LLM_ADAPTERS`, `TOOLS`, `MEMORY`, `RAG`, `OBSERVABILITY`, `UNIFIED_EXECUTION_RUNTIME`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `ADAPTIVE_HARNESS_INTELLIGENCE`

---

## Cursor read scope (token budget)

Do not read the whole repository.

Default read scope for Token Optimization work:

1. `docs/features/architecture/TOKEN_OPTIMIZATION.md`
2. `docs/features/plan/TOKEN_OPTIMIZATION.md`
3. The affected domain architecture/plan pair for the current TOKEN slice.
4. The minimal source files required by that domain plan item.

Do not create `docs/plan/TOKEN_OPTIMIZATION.md`. This is a multi-layer feature plan, not a domain-layer plan.

---

## Planning model

This file coordinates cross-layer delivery. Concrete implementation rows must still be added to the owning domain plan files when a phase becomes actionable.

Examples:

| TOKEN phase | Owning plan file |
|-------------|------------------|
| `TOKEN-2` OutputPolicy runtime | `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` and/or `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` |
| `TOKEN-3` ToolSchemaOptimizer | `docs/plan/TOOLS.md` |
| `TOKEN-4` ContextPackOptimizer | `docs/plan/CONTEXT_ENGINEERING.md` |
| `TOKEN-5` MemorySummaryCompressor | `docs/plan/MEMORY.md` |
| `TOKEN-6` telemetry and gates | `docs/plan/OBSERVABILITY.md` plus affected domain plans |
| `TOKEN-7` adaptive optimization | `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` |

---

## Phase TOKEN-1 — Feature architecture and domain sync

**Goal:** Establish Token Optimization as a documented multi-layer feature without breaking the domain architecture/plan 1:1 rule.

**Owner layer:** `CONTEXT_ENGINEERING` as anchor; documentation structure owned by platform docs.

**Dependencies:** none.

**Deliverables:**

- `docs/features/architecture/TOKEN_OPTIMIZATION.md`
- `docs/features/plan/TOKEN_OPTIMIZATION.md`
- updated platform documentation navigation,
- updated Cursor docs-sync instruction,
- domain architecture sync plan for affected layers,
- ADR queue.

**Acceptance criteria:**

- no `docs/plan/TOKEN_OPTIMIZATION.md` exists unless a matching `docs/architecture/TOKEN_OPTIMIZATION.md` is intentionally added,
- feature architecture and feature plan link to each other,
- affected domain docs know where cross-layer feature coordination lives,
- docs checks understand feature pairs or explicitly ignore them safely.

**Required checks:**

```bash
uv run python scripts/check_docs_domain_pairs.py
```

**Status:** Planned.

---

## Phase TOKEN-2 — OutputPolicy runtime

**Goal:** Replace prompt-only verbosity control with runtime output policy.

**Owner layer:** `UNIFIED_EXECUTION_RUNTIME`; possible contract touchpoint in `AGENT_CONTRACTS_AND_ASSEMBLY`.

**Dependencies:** TOKEN-1.

**Deliverables:**

- `OutputPolicy` contract,
- `OutputPolicyResolver`,
- output profiles: `minimal`, `terse`, `standard`, `full`, `audit`, `machine_receipt`, `debug_verbose`,
- runtime safety bypass rules,
- model call integration where output budget/profile is resolved.

**Acceptance criteria:**

- output profile is selected by runtime policy, not ad-hoc prompt wording,
- high-risk contexts can force standard/full clarity,
- terse mode is available for operator updates,
- audit/full mode remains explicit.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/ -q
uv run python scripts/check_output_policy_wiring.py
```

**Status:** Planned.

---

## Phase TOKEN-3 — ToolSchemaOptimizer

**Goal:** Reduce recurring tool catalog token cost without changing tool schema semantics.

**Owner layer:** `TOOLS`.

**Dependencies:** TOKEN-1; may use TOKEN-2 output/profile contracts if shared.

**Deliverables:**

- compact tool description presentation,
- natural-language example compression where safe,
- protected schema validation,
- savings telemetry for tool catalog injection.

**Acceptance criteria:**

- tool names, parameter names, enum values, required fields, and JSON schema semantics are unchanged,
- tool call payloads are not compressed by default,
- compact catalog can be enabled by policy/profile,
- schema preservation tests pass.

**Required tests/checks:**

```bash
uv run pytest tests/unit/tools/ -q
uv run python scripts/check_tool_schema_optimizer.py
```

**Status:** Planned.

---

## Phase TOKEN-4 — ContextPackOptimizer

**Goal:** Optimize selected context fragments after ranking/budgeting and before final formatting/preflight.

**Owner layer:** `CONTEXT_ENGINEERING`.

**Dependencies:** TOKEN-1; consumes LLM adapter token counters.

**Deliverables:**

- `ContextPackOptimizer`,
- source-aware compression strategy,
- protected-region handling,
- compression receipts attached to context provenance,
- post-compression token recalculation,
- fallback to original fragments on validation failure.

**Acceptance criteria:**

- ranking happens before lossy compression,
- mandatory/policy fragments are preserved,
- total assembled tokens decrease in benchmark cases,
- context quality gate remains green,
- provenance contains compression receipt references where applicable.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/nexus/context/ -q
uv run python scripts/check_context_preflight_uses_adapter_tokens.py
uv run python scripts/check_compression_receipts.py
```

**Status:** Planned.

---

## Phase TOKEN-5 — MemorySummaryCompressor

**Goal:** Safely compress persistent natural-language memory summaries and documentation-derived memory blocks.

**Owner layer:** `MEMORY`.

**Dependencies:** TOKEN-1; may reuse receipt contracts from TOKEN-4.

**Deliverables:**

- staging write flow,
- protected-region validator,
- semantic validation for lossy summaries,
- receipt storage,
- rollback metadata.

**Acceptance criteria:**

- live source is never overwritten before validation,
- failed compression cannot corrupt persistent memory,
- original and compressed hashes are stored,
- rollback path is documented and tested.

**Required tests/checks:**

```bash
uv run pytest tests/unit/memory/ -q
uv run python scripts/check_memory_compression_receipts.py
```

**Status:** Planned.

---

## Phase TOKEN-6 — Telemetry and regression gates

**Goal:** Make token savings measurable and safe across runs, steps, models, and sources.

**Owner layer:** `OBSERVABILITY`; affected implementation owners per source domain.

**Dependencies:** TOKEN-2, TOKEN-3, TOKEN-4, or TOKEN-5 depending on first telemetry source.

**Deliverables:**

- token optimization events,
- counters,
- spans,
- savings attribution model,
- token-vs-quality regression benchmark tasks,
- CI-safe regression checks.

**Acceptance criteria:**

- optimized model calls report raw/after/saved token counts,
- savings are attributable by run, step, source, model, provider, and strategy,
- regressions can fail CI when token growth is uncontrolled or quality drops.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/observability/ -q
uv run python scripts/check_token_optimization_contracts.py
uv run python scripts/check_token_regression_benchmarks.py
```

**Status:** Planned.

---

## Phase TOKEN-7 — Adaptive optimization

**Goal:** Use historical telemetry to recommend or select budgets and compression strategies.

**Owner layer:** `ADAPTIVE_HARNESS_INTELLIGENCE`.

**Dependencies:** TOKEN-6.

**Deliverables:**

- adaptive budget recommendation inputs,
- compact/full profile recommendation by task/step type,
- quality-drop escalation rules,
- operator override support.

**Acceptance criteria:**

- adaptive optimization remains policy-governed,
- runtime can escalate to fuller context when quality drops,
- recommendations are observable and reversible,
- no autonomous compression is applied without configured policy.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/adaptive/ -q
```

**Status:** Planned.

---

## ADR queue

| ADR | Scope | Status |
|-----|-------|--------|
| `ADR-TOKEN-001` | Multi-layer feature boundary and ownership | Planned |
| `ADR-TOKEN-002` | Compression receipts and protected-region validation | Planned |
| `ADR-TOKEN-003` | Tool schema optimization safety model | Planned |

---

## Delivery rules

- One TOKEN phase or one domain-owned subset per PR.
- Update feature plan and affected domain plan together when a TOKEN phase becomes active.
- Do not implement runtime code in docs-sync PRs.
- Do not duplicate existing Context Engineering budget/preflight mechanisms.
- Do not report token savings without quality/safety validation.
- Preserve architecture/plan 1:1 domain pairs.
- Preserve feature architecture/plan 1:1 feature pairs.

---

## Explicit exclusions

Token Optimization does not:

- compress private chain-of-thought,
- mutate executable code,
- rewrite strict JSON schema semantics,
- compress tool call payloads by default,
- remove required audit evidence,
- replace RAG ranking,
- replace memory lifecycle management,
- replace LLM adapter token counting,
- replace model routing.
