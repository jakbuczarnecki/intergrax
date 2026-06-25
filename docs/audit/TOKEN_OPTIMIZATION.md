<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Architecture Adoption Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain:** `TOKEN_OPTIMIZATION`  
**Type:** proposed cross-cutting capability / Context Engineering extension  
**Primary domain owner:** `CONTEXT_ENGINEERING`  
**Related domains:** `LLM_ADAPTERS`, `TOOLS`, `MEMORY`, `RAG`, `OBSERVABILITY`, `UNIFIED_EXECUTION_RUNTIME`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `ADAPTIVE_HARNESS_INTELLIGENCE`  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new Cursor / agent chat with full repository access.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only if needed.
4. The agent must perform architecture/plan adoption analysis first — not runtime implementation.
5. The expected output is a precise set of architecture, plan, ADR, and implementation steps that can later be executed in small PRs.

This document exists because Token Optimization is a potential Intergrax market advantage and must be integrated deliberately instead of being added as a prompt-only brevity trick.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: TOKEN_OPTIMIZATION
mode: architecture-adoption-audit
focus:

# mode: architecture-adoption-audit | docs-sync-plan | implementation-plan
# focus: optional narrow slice — e.g. "tool schema optimization only", "output policy only", "context compression only"

# ═══ END USER CONFIG ═══

# TASK: Token Optimization architecture adoption audit for Intergrax

You are an **implementation audit and architecture planning agent** for the Intergrax Harness AI platform.

Your mission is to design how Intergrax should adopt a **Token Optimization** capability inspired by the market success of `JuliusBrussee/caveman`, but implemented as a professional, policy-governed, observable runtime architecture — not as a meme-style prompt mode.

Do **not** implement runtime code in this task.

Do **not** perform a repo-wide exploratory audit.

Do **not** modify unrelated layers.

Your output must let a future Cursor session update the correct architecture and plan files, then implement the work in small production-grade slices.

---

## 0. Context budget rules

Follow Intergrax token-budget discipline.

- One domain/adoption topic per chat.
- Use grep/path filters before opening large files.
- Read only scoped sections.
- Do not load full architecture hubs unless explicitly necessary.
- Do not load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` unless a compact slice or read-scope block requires it.
- Do not read `docs/audit_results/` unless a RESUME note explicitly points there.
- Do not use subagents unless the operator asks.
- If more than the listed files are needed, stop and ask.

---

## 1. Mandatory reference concept

Treat `JuliusBrussee/caveman` as a reference for token-efficiency mechanisms and market demand, not as a direct dependency.

Important concept extraction:

```text
Caveman = prompt-level brevity plugin.
Intergrax target = architecture-level token economy engine.
```

Mechanisms worth considering:

| Reference mechanism | Intergrax interpretation |
|---------------------|--------------------------|
| Terse output mode | Runtime `OutputPolicy` profiles. |
| Memory/instruction compression | Validated memory/document compression with receipts. |
| MCP tool description shrinker | `ToolSchemaOptimizer` for tool catalogs and descriptions. |
| Token savings stats | Token telemetry, dashboard counters, receipts. |
| Safety exceptions | Policy-governed compression bypass rules. |
| Preserve code/URLs/paths/errors | Protected-region validation. |

Key warning:

Token optimization must not only make replies shorter. It must reduce waste across:

```text
input → context assembly → memory → RAG → tool schemas → model routing → output → telemetry
```

---

## 2. Canonical reads

Read these in order.

### 2.1 Existing audit/checklist context

1. `docs/audit/README.md` — shared production Harness checklist.
2. `.cursor/rules/intergrax-token-budget.mdc` — existing token budget operator rules, if present.

### 2.2 Primary architecture domains

Read only read-scope blocks and relevant sections:

1. `docs/architecture/CONTEXT_ENGINEERING.md`
2. `docs/architecture/LLM_ADAPTERS.md`
3. `docs/architecture/TOOLS.md`
4. `docs/architecture/MEMORY.md`
5. `docs/architecture/OBSERVABILITY.md`
6. `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md`

### 2.3 Primary plan domains

Read only active/open plan sections and read-scope blocks:

1. `docs/plan/CONTEXT_ENGINEERING.md`
2. `docs/plan/LLM_ADAPTERS.md`
3. `docs/plan/TOOLS.md`
4. `docs/plan/MEMORY.md`
5. `docs/plan/OBSERVABILITY.md`
6. `docs/plan/UNIFIED_EXECUTION_RUNTIME.md`

### 2.4 Optional only if required

Open only if the adoption decision requires it:

- `docs/architecture/RAG.md`
- `docs/plan/RAG.md`
- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/intergrax_runtime_architecture.md`
- `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` read-scope only

---

## 3. Audit mission

Determine how Token Optimization should become a formal Intergrax capability.

Answer these questions with evidence from the repository:

1. Should Token Optimization be a standalone domain, a Context Engineering subdomain, or a cross-cutting capability anchored in Context Engineering?
2. Which existing layers already contain partial token optimization mechanisms?
3. Which mechanisms are already production-ready and should not be duplicated?
4. Which missing mechanisms should be added?
5. Which architecture docs must be updated?
6. Which plan docs must receive implementation rows?
7. Which ADRs are required before implementation?
8. What is the safest implementation order?
9. What tests and gates are required to prevent token savings from damaging quality or safety?
10. How should token savings become visible in Observability?

---

## 4. Target architecture concept

Design Token Optimization as a policy-governed runtime capability.

Target statement:

```text
Intergrax provides runtime-controlled token economy for governed AI agents.
```

The capability should minimize unnecessary tokens while preserving:

- correctness,
- safety,
- provenance,
- auditability,
- protected technical details,
- developer control,
- model-call quality.

Token Optimization should cover:

- input token budgeting,
- context assembly efficiency,
- semantic and structural context compression,
- memory summary compression,
- tool description/schema compression,
- output verbosity control,
- token-aware model routing signals,
- token savings telemetry,
- compression receipts,
- regression benchmarks for tokens vs quality,
- policy-governed compression safety.

Token Optimization should not:

- alter private reasoning tokens,
- compress private chain-of-thought,
- mutate executable code,
- rewrite JSON schema semantics,
- remove required audit evidence,
- replace RAG ranking,
- replace memory lifecycle management,
- replace LLM adapter token counting,
- replace model selection logic.

---

## 5. Proposed capability model

Evaluate and refine this model against the existing Intergrax architecture:

```text
TokenOptimizationPolicy
  ├─ InputBudgetPolicy
  ├─ OutputPolicy
  ├─ CompressionPolicy
  ├─ ToolCatalogOptimizationPolicy
  ├─ MemoryCompressionPolicy
  ├─ ProtectedRegionPolicy
  ├─ ValidationPolicy
  └─ TelemetryPolicy
```

Potential runtime components:

```text
TokenOptimizer
OutputPolicyResolver
ToolSchemaOptimizer
MemorySummaryCompressor
ContextPackOptimizer
CompressionReceiptValidator
TokenOptimizationTelemetryEmitter
TokenRegressionBenchmarkRunner
```

Potential data contracts:

```python
@dataclass(frozen=True, slots=True)
class CompressionReceipt:
    receipt_id: str
    run_id: str | None
    step_id: str | None
    source_type: str
    strategy: str
    lossy: bool
    original_hash: str
    compressed_hash: str
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    saved_ratio: float
    protected_regions_count: int
    validation_status: Literal["passed", "failed", "bypassed"]
    quality_score: float | None = None
    fallback_used: bool = False
```

```python
@dataclass(frozen=True, slots=True)
class OutputPolicy:
    profile: Literal[
        "minimal",
        "terse",
        "standard",
        "full",
        "audit",
        "machine_receipt",
        "debug_verbose",
    ]
    max_output_tokens: int | None
    require_sections: tuple[str, ...] = ()
    forbid_sections: tuple[str, ...] = ()
    preserve_exact: tuple[str, ...] = ()
```

---

## 6. Required safety model

Token Optimization must preserve exact text for protected regions:

- fenced code blocks,
- inline code,
- file paths,
- URLs,
- API names,
- class names,
- function names,
- commands,
- environment variables,
- identifiers,
- enum values,
- hashes,
- dates,
- version numbers,
- exact error strings,
- security warnings,
- irreversible action confirmations,
- legal/compliance text unless explicit lossy summarization is allowed.

Compression should be disabled or downgraded for:

- destructive operations,
- security warnings,
- legal/compliance analysis,
- migration steps where sequence ambiguity matters,
- strict structured output schemas,
- evidence-heavy audit output,
- citation-sensitive answers,
- user clarification requests,
- previous misunderstanding caused by compression.

Persistent compression must follow this rule:

```text
Never overwrite live source before validation.
Use staging output → validate → receipt → atomic replace → rollback metadata.
```

---

## 7. Required integration analysis

For each domain below, determine exact adoption points and whether updates are needed.

### 7.1 `CONTEXT_ENGINEERING`

Analyze whether Token Optimization should extend the context lifecycle:

```text
COLLECT → NORMALIZE → SCORE → FILTER → RANK → BUDGET → TOKEN_OPTIMIZE → FORMAT → TOKEN_PREFLIGHT → VALIDATE → EMIT
```

Check existing support for:

- context budgets,
- degradation ladder,
- tokenizer-aware preflight,
- provenance,
- step-aware ranking,
- candidate exclusion reasons,
- quality scoring,
- context assembly events.

Expected output:

- exact architecture sections to update,
- exact plan rows to add,
- whether `ContextPackOptimizer` belongs here.

### 7.2 `LLM_ADAPTERS`

Analyze adoption points for:

- adapter-native token counting,
- context window metadata,
- output token budget resolution,
- model cost metadata,
- routing inputs for cost/quality/latency.

Expected output:

- whether Token Optimization needs new adapter contract methods,
- whether existing token counters are enough,
- required plan rows.

### 7.3 `TOOLS`

Analyze adoption points for `ToolSchemaOptimizer`.

Rules:

- may compress tool `description` fields,
- may compress natural-language examples,
- must preserve tool names,
- must preserve parameter names,
- must preserve enum values,
- must preserve required fields,
- must preserve JSON schema semantics,
- must not compress tool call payloads by default,
- must not compress tool results unless result type allows it.

Expected output:

- where tool catalog injection happens,
- where description compression can be inserted,
- tests needed to prove schema safety.

### 7.4 `MEMORY`

Analyze adoption points for `MemorySummaryCompressor`.

Expected output:

- how memory summaries are currently built/stored,
- where compression receipts should live,
- rollback requirements,
- validation requirements.

### 7.5 `RAG`

Analyze whether retrieved chunks should be compressed.

Rule:

```text
RAG compression must happen after retrieval/ranking, not before source relevance is known.
```

Expected output:

- whether RAG chunk compression is in scope for first implementation,
- how citations and source spans are preserved,
- quality tests required.

### 7.6 `OBSERVABILITY`

Analyze required events and counters.

Candidate events:

```text
TOKEN_OPTIMIZATION_STARTED
TOKEN_OPTIMIZATION_APPLIED
TOKEN_OPTIMIZATION_BYPASSED
TOKEN_OPTIMIZATION_FAILED
TOKEN_OPTIMIZATION_RECEIPT_CREATED
TOKEN_BUDGET_EXCEEDED
TOKEN_REGRESSION_DETECTED
```

Candidate metrics:

```text
input_tokens_raw_total
input_tokens_after_dedup_total
input_tokens_after_compression_total
output_tokens_budgeted_total
output_tokens_actual_total
saved_tokens_total
saved_cost_estimate_total
compression_attempts_total
compression_failures_total
compression_fallbacks_total
token_budget_overflow_total
token_regression_failures_total
```

Expected output:

- which event spine files require updates,
- which counters/spans should be added,
- how savings should be attributed per run/step/source/model.

### 7.7 `UNIFIED_EXECUTION_RUNTIME`

Analyze adoption points for runtime policy enforcement.

Expected output:

- where Token Optimization policy should be resolved,
- how safety bypass rules are enforced,
- how runtime profiles select output mode and compression level.

---

## 8. Required implementation phases

Produce a concrete phased plan. Use this as the baseline and refine it based on repository evidence.

### TOKEN-1 — Architecture and contracts

Goal: establish canonical docs, ADR, and contract boundaries.

Expected deliverables:

- final domain decision: standalone vs CE extension,
- `docs/plan/TOKEN_OPTIMIZATION.md` if standalone/cross-cutting plan hub is justified,
- ADR for domain boundary,
- contract stubs only if implementation begins.

### TOKEN-2 — OutputPolicy runtime

Goal: replace prompt-only verbosity control with runtime policy.

Expected deliverables:

- `OutputPolicy`,
- `OutputPolicyResolver`,
- output profiles,
- safety bypass,
- tests.

### TOKEN-3 — ToolSchemaOptimizer

Goal: reduce recurring tool catalog token cost.

Expected deliverables:

- compact tool descriptions,
- schema preservation tests,
- telemetry for catalog savings.

### TOKEN-4 — ContextPackOptimizer

Goal: optimize selected context fragments before final formatting.

Expected deliverables:

- source-aware compression,
- provenance metadata,
- post-compression token recalculation,
- fallback on validation failure.

### TOKEN-5 — MemorySummaryCompressor

Goal: safely compress persistent memory/document summaries.

Expected deliverables:

- staging write,
- validation,
- receipt,
- rollback metadata,
- protected-region checks.

### TOKEN-6 — Telemetry and regression gates

Goal: make savings measurable and safe.

Expected deliverables:

- events,
- counters,
- benchmark tasks,
- token regression gates.

### TOKEN-7 — Adaptive optimization

Goal: use historical telemetry to select budgets and strategies.

Expected deliverables:

- AHI integration,
- adaptive budget recommendations,
- automatic compact/full escalation based on quality.

---

## 9. Required output format

Return the audit in this structure:

```text
# Token Optimization Adoption Audit

## 1. Verdict
- standalone domain vs CE extension vs cross-cutting capability
- maturity target
- highest ROI slice

## 2. Existing Intergrax capabilities already covering this
- file/path evidence
- what not to duplicate

## 3. Gaps
- P0/P1/P2/P3 table
- owner domain
- required doc update
- required implementation slice

## 4. Architecture update plan
- exact files
- exact sections to add/change
- dependency order

## 5. Plan update plan
- exact plan files
- proposed TOKEN-* rows
- acceptance criteria

## 6. ADR plan
- ADR IDs
- decision scope
- alternatives considered

## 7. Implementation roadmap
- PR order
- per PR read scope
- per PR edit scope
- tests/gates

## 8. Risk analysis
- quality risk
- safety risk
- schema break risk
- observability risk
- cost-vs-savings risk

## 9. Cursor handoff prompts
- prompt for docs sync
- prompt for TOKEN-2 implementation
- prompt for TOKEN-3 implementation
```

---

## 10. Non-negotiable constraints

- Do not implement runtime code during architecture-adoption-audit mode.
- Do not create large unscoped PRs.
- Do not duplicate existing Context Engineering budget/preflight mechanisms.
- Do not compress executable code.
- Do not compress tool call payloads by default.
- Do not compress strict JSON schema semantics.
- Do not report token savings without quality/safety validation.
- Do not overwrite persistent files before compression validation.
- Preserve Intergrax tier boundaries.
- Preserve existing commit-per-focused-task discipline.

---

## 11. Recommended first execution result

The ideal first Cursor result should be a docs-only adoption plan, not code.

It should answer:

```text
Where should Token Optimization live?
Which existing layers change?
Which plan rows are required?
Which ADR comes first?
Which implementation slice has highest ROI?
```

Expected likely conclusion:

```text
Accept Token Optimization as a cross-cutting Intergrax capability anchored in CONTEXT_ENGINEERING, with first implementation slice focused on OutputPolicy and ToolSchemaOptimizer before deeper semantic context compression.
```

---END PROMPT---
