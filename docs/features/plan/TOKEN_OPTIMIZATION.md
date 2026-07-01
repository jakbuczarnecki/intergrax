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

## Satellite registers (read on demand)

Large cross-domain sync registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) | domain plan cross-reference map, TOKEN row checklist, phase → plan mapping |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Cursor read scope (token budget)

Do not read the whole repository.

Default read scope for Token Optimization work:

1. `docs/features/architecture/TOKEN_OPTIMIZATION.md` (read-scope block only)
2. `docs/features/plan/TOKEN_OPTIMIZATION.md` (read-scope block + active TOKEN phase only)
3. The affected domain architecture/plan pair for the current TOKEN slice.
4. The minimal source files required by that domain plan item.

**On demand (one max):** [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) when syncing domain plan rows or cross-references.

Do not create `docs/plan/TOKEN_OPTIMIZATION.md`. This is a multi-layer feature plan, not a domain-layer plan.

**Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Planning model

This file coordinates cross-layer delivery. Concrete implementation rows must still be added to the owning domain plan files when a phase becomes actionable.

| TOKEN phase | Owning plan file |
|-------------|------------------|
| `TOKEN-ARCH-0` engine lifecycle, mechanisms, strategy taxonomy, config, plugins, claims | feature architecture + feature plan (docs-only) |
| `TOKEN-1` shared contracts, receipts, protected regions | feature plan + `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` |
| `TOKEN-2` OutputPolicy runtime | `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` and optional `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` |
| `TOKEN-3` ToolSchemaOptimizer | `docs/plan/TOOLS.md` |
| `TOKEN-4` ContextPackOptimizer | `docs/plan/CONTEXT_ENGINEERING.md` |
| `TOKEN-5` MemorySummaryCompressor | `docs/plan/MEMORY.md` |
| `TOKEN-6` telemetry and regression gates | `docs/plan/OBSERVABILITY.md` plus affected domain plans |
| `TOKEN-7` adaptive optimization | `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` |

**LKW proof workload:** LKW is the primary proof workload for Token Optimization. Token Optimization is **not** a local LKW feature — it is a cross-layer platform capability owned by runtime and domain plans. LKW proof must show measurable token savings, quality/regression safety, compression receipts, protected-region preservation, and observability attribution through the Harness Observability Spine. **LKW-PF6-0** proof design is **Done / Closed** (§LKW-PF6-0 below); **TOKEN-ARCH-0** engine architecture is **Done / Closed** (§TOKEN-ARCH-0 below); **TOKEN-1A** shared contracts is **Done / Closed** (§TOKEN-1A below). TOKEN implementation order remains **TOKEN-ARCH-0** → **TOKEN-1**..**TOKEN-7** below; LKW proof ordering introduces proof-design and baseline-measurement tasks around those phases (see [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PF — Recommended execution order).

---

## LKW-PF6-0 — Token Optimization proof design

**Status:** **Done / Closed** (docs-only).

**Maturity level:** proof design only — does not close `LKW-PF6` platform proof.

**Purpose:** Define exactly what the LKW Token Optimization proof must demonstrate before **TOKEN-1A** code starts. This section is the canonical source; [`applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md`](../../applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md) §10 and [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PF6-0 closeout mirror it for LKW scheduling.

**Narrative:** Intergrax proves that agent applications can be built as configurable, observable, cost-aware runtime systems — not hand-wired demos.

**Out of scope for LKW-PF6-0:** contracts, runtime behavior, optimizers, telemetry payloads, validators, benchmarks, fixtures, scripts, and any `TOKEN-*` implementation.

### Representative LKW workflows

All workflows use the existing LKW product proof shape:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

| Workflow ID | Description | Proof intent |
|-------------|-------------|--------------|
| **LKW-TOK-W1** | Small workspace indexing + search + synthesis | Minimal tenant-scoped baseline; compact corpus per-step token categories. |
| **LKW-TOK-W2** | Medium workspace search + synthesis with evidence | RAG/evidence/context-pack attribution under realistic retrieval load. |
| **LKW-TOK-W3** | Repeated synthesis with similar tool/catalog/context exposure | Recurring tool-catalog and context-pack savings across stable-exposure runs. |
| **LKW-TOK-W4** | Failure/safety-preserving run — exact regions must not be compressed | Optimization rejection/fallback when protected regions or safety boundaries would be violated. |

### Baseline measurement shape

Measured **before** optimization; must be reproducible enough to compare against optimized runs.

Required fields per measured step/run scope:

- `input_context_tokens`
- `tool_catalog_tokens`
- `retrieved_evidence_context_pack_tokens`
- `output_tokens`
- `total_tokens`
- `model`
- `provider`
- `runtime_profile`
- `workflow_id` (`LKW-TOK-W1` … `LKW-TOK-W4`)
- `run_id`
- `step_id`

### Optimized measurement shape

Later optimized proof runs must report:

- `baseline_token_usage` (per category)
- `optimized_token_usage` (per category)
- `saved_tokens`
- `saved_ratio`
- `optimization_strategy`
- `affected_source_category`
- `fallback_status`
- `validation_status`

### Token categories

Canonical categories:

| Category | Notes |
|----------|-------|
| input/context tokens | Assembled prompt/context before optimization |
| tool catalog tokens | LLM-facing tool schema/catalog view |
| RAG/evidence/context pack tokens | Retrieved fragments included in context |
| memory tokens | Memory summaries/blocks when in scope |
| output tokens | Model completion for the step |
| system/policy tokens | Where measurable separately |
| total tokens | Aggregate for the measured scope |

Attribution dimensions (required for later telemetry and public proof): `run`, `step`, `source`, `model`, `provider`, `strategy`, `output_profile`.

### Quality and regression criteria

Optimized run **fails** the proof if token savings break any of:

- tenant-scoped evidence
- evidence references
- synthesized answer integrity
- shadow artifact behavior
- safety boundaries
- exact protected regions
- platform abstraction boundaries

**Behavioral equivalence rule:** baseline and optimized results must remain behaviorally equivalent for the proof workload; only allowed formatting or verbosity differences are permitted.

### Protected-region requirements

Token Optimization must never lose or rewrite:

- code blocks, inline code, paths, URLs, env vars, enum values, hashes, dates, exact error strings, policy text, IDs, tenant identifiers required for correctness, evidence references

**TOKEN-1B** protected-region parser/validator is **Done / Closed** (§TOKEN-1B below). **LKW-PF6-0** defines proof requirements only.

### Compression receipt expectations

Future receipts ( **TOKEN-1C** ) must prove:

- original hash, optimized hash
- original token count, optimized token count
- saved tokens, saved ratio
- strategy
- protected-region validation status
- fallback reason when optimization is rejected

No receipt implementation in LKW-PF6-0.

### Observability visibility

Token savings must be visible through the **Harness Observability Spine** or an **approved domain-signal path**. No private Token Optimization telemetry bus.

Later proof attribution fields: `run_id`, `step_id`, `workflow_id`, `model`, `provider`, `profile`, `source/category`, `strategy`, `baseline_tokens`, `optimized_tokens`, `saved_tokens`, `saved_ratio`, `validation_status`, `fallback_status`.

Owner plan: [`docs/plan/OBSERVABILITY.md`](../../plan/OBSERVABILITY.md) Phase TOKEN-OBS; early slice **TOKEN-6A-lite** defines telemetry shape only.

### Public proof format (LKW-PF6-C target)

Later public-grade proof must include:

- representative workflow description
- baseline and optimized token usage
- saved tokens and saved ratio
- receipt references
- protected-region validation result
- quality/regression result
- observability attribution
- known limitations

**Redaction — must not expose:** raw prompts, raw documents, raw RAG chunks, raw synthesized content, tool args, secrets, tokens/secrets, absolute file paths, large raw artifacts.

### LKW-PF6-0 closure rule

Done / Closed when:

- [x] §Representative workflows, baseline/optimized shapes, categories, quality criteria, protected regions, receipts, observability, and public proof format are defined above.
- [x] **TOKEN-1A** remains not started.
- [x] No code/runtime/test/CI/dependency files changed.

**Next step:** **TOKEN-1A** — shared contracts + package skeleton (Phase TOKEN-1 below). Preceded by **TOKEN-ARCH-0** (§TOKEN-ARCH-0 below).

### LKW proof phase map (post-design)

| Phase | Scope | Depends on |
|-------|-------|------------|
| **LKW-PF6-A** | Baseline token measurement for §workflows | TOKEN-1A contracts (minimal) |
| **LKW-PF6-B** | First measurable savings proof | TOKEN-2/3 and/or TOKEN-4 light, TOKEN-1C receipts |
| **LKW-PF6-C** | Public-grade proof artifact | LKW-PF6-B + TOKEN-6A-lite/6B attribution |

---

## TOKEN-ARCH-0 — Token Optimization Engine architecture and mechanism strategy

**Status:** **Done / Closed** (docs-only).

**Purpose:** Define the Token Optimization Engine lifecycle, mechanism catalog, strategy taxonomy, configuration model, plugin/extensibility model, benchmark claim model, and first public proof mechanism selection before shared contracts are implemented.

**Canonical architecture:** [`../architecture/TOKEN_OPTIMIZATION.md`](../architecture/TOKEN_OPTIMIZATION.md) §8 Token Optimization Engine lifecycle, mechanisms, and extensibility.

**Out of scope for TOKEN-ARCH-0:** shared contract implementation, runtime behavior, optimizers, telemetry payloads, validators, benchmarks, fixtures, scripts, and any `TOKEN-1*` code.

### Acceptance

Done / Closed when:

- [x] engine lifecycle documented
- [x] mechanism catalog documented
- [x] strategy taxonomy documented
- [x] configuration model documented
- [x] plugin/extensibility model documented
- [x] benchmark claim model documented
- [x] first public proof candidate mechanisms documented
- [x] **TOKEN-1A** shared contracts — Done / Closed (§TOKEN-1A below)
- [x] no runtime/code/test/CI/dependency changes (TOKEN-ARCH-0 docs-only scope)

**Next step:** **TOKEN-1C** — compression receipts + validation helpers (Phase TOKEN-1 below).

---

## TOKEN-1B — Protected region parser/validator

**Status:** **Done / Closed**.

**Purpose:** Add deterministic protected-region detection and validation helpers as the first safety gate for Token Optimization.

**Deliverables:**

- `intergrax/runtime/token_optimization/protected_regions.py`
- `tests/unit/runtime/token_optimization/test_protected_regions.py`

**Closeout:**

- protected-region detection helper added
- protected-region validation helper added
- uses TOKEN-1A contracts
- no optimization behavior added
- no receipts added
- no telemetry wiring added

**Next step:** **TOKEN-1C** — compression receipts + validation helpers.

---

## TOKEN-1A — Shared contracts + package skeleton

**Status:** **Done / Closed**.

**Purpose:** Add shared Token Optimization contract vocabulary and a minimal runtime package skeleton for later phases (TOKEN-1B..TOKEN-7).

**Deliverables:**

- `intergrax/runtime/token_optimization/__init__.py`
- `intergrax/runtime/token_optimization/contracts.py`
- `tests/unit/runtime/token_optimization/test_contracts.py`

**Closeout:**

- shared package skeleton added
- shared contracts added (profiles, policy, attribution, mechanisms, strategies, plugin descriptors, measurements, protected regions, receipt refs, request/result)
- plugin descriptor contracts added
- no runtime optimization behavior added
- no telemetry wiring added

**Next step:** **TOKEN-1C** — compression receipts + validation helpers.

---

## Implementation Blueprint

### Target runtime component layout

```text
intergrax/runtime/token_optimization/
  __init__.py
  contracts.py                 # shared DTOs, enums, policies
  output_policy.py             # OutputPolicyResolver and output profiles
  protected_regions.py         # protected region parser + validator
  receipts.py                  # CompressionReceipt builders/validators
  optimizer.py                 # TokenOptimizer orchestrator
  telemetry.py                 # HOS/domain-signal/metric emission helpers
  regression.py                # token-vs-quality benchmark helpers

intergrax/runtime/nexus/context/
  context_pack_optimizer.py    # CE integration after rank/budget, before format/preflight

intergrax/runtime/nexus/tools/
  tool_schema_optimizer.py     # compact LLM-facing tool catalog view

intergrax/memory/
  summary_compressor.py        # staging + validation + receipt + rollback for memory summaries
```

### Shared contracts to implement first

The first implementation slice must add contracts only, without wiring behavior into hot paths.

Required contracts:

- `OutputProfile`
- `CompressionLevel`
- `TokenOptimizationBypassReason`
- `TokenOptimizationSourceType`
- `ProtectedRegionKind`
- `TokenOptimizationPolicy`
- `OutputPolicy`
- `CompressionReceipt`
- `ProtectedRegion`
- `ProtectedRegionValidationResult`
- `TokenOptimizationRequest`
- `TokenOptimizationResult`
- `TokenOptimizationTelemetry`

Rules:

- Prefer frozen dataclasses with `slots=True` unless an existing domain requires Pydantic.
- Every runtime example using `RuntimeState` must explicitly pass `run_id`.
- Trace event calls must use `TraceLevel` enum where applicable.
- New Python files must start with the Intergrax copyright header.

### Implementation order

```text
LKW-PF6-0   Token Optimization proof design — Done / Closed
TOKEN-ARCH-0  Token Optimization Engine architecture and mechanism strategy — Done / Closed
TOKEN-1A    shared contracts + package skeleton — Done / Closed
TOKEN-1B    protected region parser/validator — Done / Closed
TOKEN-1C    compression receipts + validation helpers
TOKEN-2     OutputPolicy runtime resolver
TOKEN-3     ToolSchemaOptimizer compact catalog view
TOKEN-4     ContextPackOptimizer light/structural compression only
TOKEN-6A    telemetry payloads/counters for TOKEN-2..4
TOKEN-5     MemorySummaryCompressor with staging/rollback
TOKEN-6B    token regression benchmark runner + CI scripts
TOKEN-7     adaptive recommendations from telemetry, no auto-apply by default
```

Semantic compression is deliberately delayed until protected-region validation, receipts, telemetry, and regression gates exist.

---

## Phase TOKEN-1 — Shared contracts, receipts, and protected regions

**Goal:** Establish the safe foundation used by all later Token Optimization slices.

**Owner layer:** `UNIFIED_EXECUTION_RUNTIME` for runtime policy placement; feature plan for shared contracts; `OBSERVABILITY` consulted for receipt/telemetry shape.

**Dependencies:** feature architecture accepted; **TOKEN-ARCH-0** closed.

**Deliverables:**

- `intergrax/runtime/token_optimization/__init__.py`
- `intergrax/runtime/token_optimization/contracts.py`
- `intergrax/runtime/token_optimization/protected_regions.py`
- `intergrax/runtime/token_optimization/receipts.py`
- unit tests for contracts, protected-region validation, and receipt hashing,
- lightweight CI script `scripts/check_token_optimization_contracts.py`.

**Acceptance criteria:**

- contract imports do not import CE/TOOLS/MEMORY hot-path modules,
- protected-region validator detects and preserves code, inline code, paths, URLs, env vars, enum values, hashes, dates, and exact error strings,
- receipt contains original hash, optimized hash, original tokens, optimized tokens, saved tokens, saved ratio, validation status, fallback flag,
- failed validation produces a fallback result rather than optimized content,
- no hot-path runtime behavior changes yet.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run python scripts/check_token_optimization_contracts.py
```

**Domain plan rows:** `TOKEN-UER-1` in `docs/plan/UNIFIED_EXECUTION_RUNTIME.md`.

**Status:** Planned.

---

## Phase TOKEN-2 — OutputPolicy runtime

**Goal:** Replace prompt-only verbosity control with runtime output policy.

**Owner layer:** `UNIFIED_EXECUTION_RUNTIME`; optional contract hints in `AGENT_CONTRACTS_AND_ASSEMBLY` later.

**Dependencies:** TOKEN-1A contracts.

**Deliverables:**

- `intergrax/runtime/token_optimization/output_policy.py`,
- `OutputPolicyResolver`,
- output profiles: `minimal`, `terse`, `standard`, `full`, `audit`, `machine_receipt`, `debug_verbose`,
- runtime safety bypass rules,
- integration point where LLM call max-output budget/profile is resolved,
- lightweight CI script `scripts/check_output_policy_wiring.py`.

**Acceptance criteria:**

- output profile is selected by runtime policy, not ad-hoc prompt wording,
- high-risk contexts can force standard/full clarity,
- terse mode is available for operator updates,
- audit/full mode remains explicit,
- structured output calls are not shortened unless schema explicitly allows it,
- no model-specific prompt hack is required.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run pytest tests/unit/runtime/ -q
uv run python scripts/check_output_policy_wiring.py
```

**Domain plan rows:** `TOKEN-UER-2` in `docs/plan/UNIFIED_EXECUTION_RUNTIME.md`.

**Status:** Planned.

---

## Phase TOKEN-3 — ToolSchemaOptimizer

**Goal:** Reduce recurring tool catalog token cost without changing tool schema semantics.

**Owner layer:** `TOOLS`.

**Dependencies:** TOKEN-1 contracts and protected-region validator; TOKEN-6 telemetry can be added after compact catalog works.

**Deliverables:**

- `intergrax/runtime/nexus/tools/tool_schema_optimizer.py`,
- compact LLM-facing tool catalog view,
- schema-preservation validator,
- optional cache key for compact catalog view,
- savings telemetry hook placeholder,
- lightweight CI script `scripts/check_tool_schema_optimizer.py`.

**Integration target:** `ToolPlanningService`, `CatalogToolPlanner`, `tool_planner_input`, or the schema export path used before `generate_with_tools`.

**Acceptance criteria:**

- canonical `ToolContract` registry is not mutated,
- tool names, parameter names, enum values, required fields, and JSON schema semantics are unchanged,
- tool call payloads and tool result JSON are not compressed by default,
- compact catalog can be enabled by policy/profile,
- schema preservation tests pass,
- token count of LLM-facing catalog decreases on a representative catalog fixture.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/nexus/tools/ -q
uv run pytest tests/unit/tools/ -q
uv run python scripts/check_tool_schema_optimizer.py
```

**Domain plan rows:** `TOKEN-TOOLS-1` in `docs/plan/TOOLS.md`.

**Status:** Planned.

---

## Phase TOKEN-4 — ContextPackOptimizer

**Goal:** Optimize selected context fragments after ranking/budgeting and before final formatting/preflight.

**Owner layer:** `CONTEXT_ENGINEERING`.

**Dependencies:** TOKEN-1 contracts/receipts/protected regions; consumes LLM adapter token counters.

**Deliverables:**

- `intergrax/runtime/nexus/context/context_pack_optimizer.py`,
- source-aware compression strategy,
- protected-region handling,
- compression receipts attached to context provenance/metadata,
- post-compression token recalculation,
- fallback to original fragments on validation failure,
- light/structural compression only in first slice.

**Integration target:** existing CE pipeline; extend `ContextCompiler` / `DefaultNexusContextEngine` rather than building a second compiler.

**Acceptance criteria:**

- ranking happens before lossy compression,
- mandatory/policy fragments are preserved,
- total assembled tokens decrease in benchmark cases,
- context quality gate remains green,
- provenance contains compression receipt references where applicable,
- hard budget and adapter-token preflight still use the existing adapter token path,
- semantic compression remains disabled until regression gates exist.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/nexus/context/ -q
uv run python scripts/maintenance/check_context_preflight_uses_adapter_tokens.py
uv run python scripts/check_compression_receipts.py
```

**Domain plan rows:** `TOKEN-CE-1` and `TOKEN-CE-2` in `docs/plan/CONTEXT_ENGINEERING.md`.

**Status:** Planned.

---

## Phase TOKEN-5 — MemorySummaryCompressor

**Goal:** Safely compress persistent natural-language memory summaries and documentation-derived memory blocks.

**Owner layer:** `MEMORY`.

**Dependencies:** TOKEN-1 contracts/receipts/protected regions; recommended after TOKEN-4 proves runtime receipts.

**Deliverables:**

- `intergrax/memory/summary_compressor.py`,
- staging write flow,
- protected-region validator reuse,
- semantic validation hook for lossy summaries,
- receipt storage metadata,
- rollback metadata,
- lightweight CI script `scripts/check_memory_compression_receipts.py`.

**Acceptance criteria:**

- live source is never overwritten before validation,
- failed compression cannot corrupt persistent memory,
- original and compressed hashes are stored,
- rollback path is documented and tested,
- memory compression is opt-in by profile/policy,
- no user facts, dates, IDs, or policy text are silently lost.

**Required tests/checks:**

```bash
uv run pytest tests/unit/memory/ -q
uv run python scripts/check_memory_compression_receipts.py
```

**Domain plan rows:** `TOKEN-MEM-1` in `docs/plan/MEMORY.md`.

**Status:** Planned.

---

## Phase TOKEN-6 — Telemetry and regression gates

**Goal:** Make token savings measurable and safe across runs, steps, models, and sources.

**Owner layer:** `OBSERVABILITY`; affected implementation owners per source domain.

**Dependencies:** TOKEN-1 contracts; can start with telemetry for TOKEN-2/TOKEN-3 before TOKEN-4/TOKEN-5 exist.

**Deliverables:**

- `intergrax/runtime/token_optimization/telemetry.py`,
- typed optimization summary payload,
- receipt payload shape,
- counters/spans emitted through HOS or approved domain-signal path,
- savings attribution model,
- token-vs-quality benchmark fixtures,
- `scripts/check_compression_receipts.py`,
- `scripts/check_token_regression_benchmarks.py`.

**Acceptance criteria:**

- optimized model calls report raw/after/saved token counts,
- savings are attributable by run, step, source, model, provider, strategy, and output profile,
- regression checks can fail CI when token growth is uncontrolled or quality drops,
- telemetry does not create a private event bus,
- event naming respects the Observability event ownership model.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/observability/ -q
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run python scripts/check_compression_receipts.py
uv run python scripts/check_token_regression_benchmarks.py
```

**Domain plan rows:** `TOKEN-OBS-1` and `TOKEN-OBS-2` in `docs/plan/OBSERVABILITY.md`.

**Status:** Planned.

---

## Phase TOKEN-7 — Adaptive optimization

**Goal:** Use historical telemetry to recommend budgets and compression strategies.

**Owner layer:** `ADAPTIVE_HARNESS_INTELLIGENCE`.

**Dependencies:** TOKEN-6 telemetry and regression gates.

**Deliverables:**

- adaptive budget recommendation inputs,
- compact/full profile recommendation by task/step/source type,
- quality-drop escalation rules,
- operator override support,
- no autonomous production auto-apply until governance permits it.

**Acceptance criteria:**

- adaptive optimization remains policy-governed,
- runtime can escalate to fuller context when quality drops,
- recommendations are observable and reversible,
- no autonomous compression is applied without configured policy,
- AHI uses Token Optimization telemetry as input rather than duplicating token accounting.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/adaptive/ -q
```

**Domain plan rows:** `TOKEN-AHI-1` in `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`.

**Status:** Planned / Frozen until TOKEN-6 ships.

---

## ADR queue

| ADR | Scope | Status |
|-----|-------|--------|
| `ADR-TOKEN-001` | Multi-layer feature boundary and runtime component placement | Planned |
| `ADR-TOKEN-002` | Protected-region validation and compression receipts | Planned |
| `ADR-TOKEN-003` | Tool schema optimization safety model | Planned |
| `ADR-TOKEN-004` | Token telemetry and regression gate semantics | Planned |

---

## Domain plan row checklist

See [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) for the canonical domain plan row checklist, TOKEN phase → owning plan mapping, and sync instructions.

---

## First implementation prompt

Use this only after the domain plan rows above exist.

```text
Pracujemy na repozytorium `jakbuczarnecki/intergrax`, branch `development`.

Cel sesji:
Zaimplementuj TOKEN-1A/TOKEN-1B/TOKEN-1C — shared Token Optimization contracts, protected-region validator, and compression receipts.

Read scope:
- docs/features/architecture/TOKEN_OPTIMIZATION.md
- docs/features/plan/TOKEN_OPTIMIZATION.md
- docs/plan/UNIFIED_EXECUTION_RUNTIME.md rows TOKEN-UER-1/TOKEN-UER-2
- existing runtime token/cost/context budget modules only as needed

Edit scope:
- intergrax/runtime/token_optimization/__init__.py
- intergrax/runtime/token_optimization/contracts.py
- intergrax/runtime/token_optimization/protected_regions.py
- intergrax/runtime/token_optimization/receipts.py
- tests/unit/runtime/token_optimization/
- scripts/check_token_optimization_contracts.py

Do not wire behavior into LLM call path yet.
Do not implement ToolSchemaOptimizer yet.
Do not implement ContextPackOptimizer yet.
Do not implement MemorySummaryCompressor yet.
Do not create docs/plan/TOKEN_OPTIMIZATION.md.

Acceptance:
- contracts import cleanly,
- protected regions are detected and validated,
- receipts hash original/optimized content and record token savings,
- failed protected-region validation forces fallback,
- tests pass.

Run:
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run python scripts/check_token_optimization_contracts.py
uv run python scripts/audit/check_docs_domain_pairs.py

Commit:
feat: add token optimization contracts and receipts
```

---

## Delivery rules

- One TOKEN phase or one domain-owned subset per PR.
- Update feature plan and affected domain plan together when a TOKEN phase becomes active.
- Do not implement runtime code in docs-sync PRs.
- Do not duplicate existing Context Engineering budget/preflight mechanisms.
- Do not duplicate LLM adapter token counting.
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
