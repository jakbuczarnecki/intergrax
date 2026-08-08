<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Architecture Adoption Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Feature:** `TOKEN_OPTIMIZATION`  
**Type:** multi-layer feature / cross-domain capability anchored in `CONTEXT_ENGINEERING`  
**Feature architecture target:** [`../features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)  
**Feature plan target:** [`../features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)  
**Related domains:** `CONTEXT_ENGINEERING`, `LLM_ADAPTERS`, `TOOLS`, `MEMORY`, `RAG`, `OBSERVABILITY`, `UNIFIED_EXECUTION_RUNTIME`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `ADAPTIVE_HARNESS_INTELLIGENCE`  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

**Main engine guide:** [../features/token_optimization/README.md](../../capabilities/token_optimization/README.md)

---

## How to use

1. Open a new Cursor / agent chat with the repository available, but do not perform broad repository exploration. Read only the files listed in Context budget / Canonical reads, use path-filtered grep before opening files, and do not use semantic search, subagents, or full-repo scans unless the operator explicitly approves.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only if needed.
4. The agent must perform feature/domain adoption analysis first — not runtime implementation.
5. The expected output is a precise set of feature docs, domain docs, ADR, and implementation steps that can later be executed in small PRs.

This document exists because Token Optimization is a potential Intergrax market advantage and must be integrated deliberately instead of being added as a prompt-only brevity trick.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

feature: TOKEN_OPTIMIZATION
mode: architecture-adoption-audit
focus:

# mode: architecture-adoption-audit | feature-docs-sync-plan | implementation-plan
# focus: optional narrow slice — e.g. "tool schema optimization only", "output policy only", "context compression only"

# ═══ END USER CONFIG ═══

# TASK: Token Optimization architecture adoption audit for Intergrax

You are an **implementation audit and architecture planning agent** for the Intergrax Harness AI platform.

Your mission is to design how Intergrax should adopt a **Token Optimization** capability inspired by the market success of `JuliusBrussee/caveman`, but implemented as a professional, policy-governed, observable runtime architecture — not as a meme-style prompt mode.

Token Optimization is a **multi-layer feature**, not a new domain pair by default.

Use the multi-layer feature structure:

```text
docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md
docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md
```

Do **not** create `docs/project/maintainers/plans/TOKEN_OPTIMIZATION.md` unless the operator explicitly promotes Token Optimization into a full architecture domain with a matching `docs/project/architecture/TOKEN_OPTIMIZATION.md`.

Do **not** implement runtime code in this task.
Do **not** perform a repo-wide exploratory audit.
Do **not** modify unrelated layers.

Your output must let a future Cursor session update the correct feature docs and affected domain architecture/plan files, then implement the work in small production-grade slices.

---

## 0. Context budget rules

Follow Intergrax token-budget discipline.

- One feature/adoption topic per chat.
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

### 2.1 Existing feature/audit/checklist context

1. `docs/project/maintainers/audit/README.md` — shared production Harness checklist and feature prompt guidance.
2. `docs/project/capabilities/README.md` — multi-layer feature documentation structure.
3. `.cursor/rules/intergrax-token-budget.mdc` — existing token budget operator rules, if present.

### 2.2 Feature docs

Read or create/update:

1. `docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md`
2. `docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md`

### 2.3 Primary architecture domains

Read only read-scope blocks and relevant sections:

1. `docs/project/architecture/CONTEXT_ENGINEERING.md`
2. `docs/project/architecture/LLM_ADAPTERS.md`
3. `docs/project/architecture/TOOLS.md`
4. `docs/project/architecture/MEMORY.md`
5. `docs/project/architecture/OBSERVABILITY.md`
6. `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`

### 2.4 Primary plan domains

Read only active/open plan sections and read-scope blocks:

1. `docs/project/maintainers/plans/CONTEXT_ENGINEERING.md`
2. `docs/project/maintainers/plans/LLM_ADAPTERS.md`
3. `docs/project/maintainers/plans/TOOLS.md`
4. `docs/project/maintainers/plans/MEMORY.md`
5. `docs/project/maintainers/plans/OBSERVABILITY.md`
6. `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`

### 2.5 Optional only if required

Open only if the adoption decision requires it:

- `docs/project/architecture/RAG.md`
- `docs/project/maintainers/plans/RAG.md`
- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/project/architecture/intergrax_runtime_architecture.md`
- `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` read-scope only

---

## 3. Audit mission

Determine how Token Optimization should become a formal Intergrax multi-layer feature.

Answer these questions with evidence from the repository:

1. Which existing layers already contain partial token optimization mechanisms?
2. Which mechanisms are already production-ready and should not be duplicated?
3. Which missing mechanisms should be added?
4. Which feature architecture sections must be updated?
5. Which feature plan phases must be added or refined?
6. Which domain architecture docs must receive cross-references?
7. Which domain plan docs must receive implementation rows?
8. Which ADRs are required before implementation?
9. What is the safest implementation order?
10. What tests and gates are required to prevent token savings from damaging quality or safety?
11. How should token savings become visible in Observability?

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

Expected output:

- exact architecture sections to update,
- exact domain plan rows to add,
- whether `ContextPackOptimizer` belongs here.

### 7.2 `LLM_ADAPTERS`

Analyze adoption points for adapter-native token counting, context window metadata, output budget resolution, model cost metadata, and routing inputs.

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

### 7.4 `MEMORY`

Analyze adoption points for `MemorySummaryCompressor`, receipts, validation, and rollback metadata.

### 7.5 `RAG`

Analyze whether retrieved chunks should be compressed.

Rule:

```text
RAG compression must happen after retrieval/ranking, not before source relevance is known.
```

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

### 7.7 `UNIFIED_EXECUTION_RUNTIME`

Analyze adoption points for runtime policy selection, output profile selection, compression level selection, safety bypass enforcement, and environment/profile-level overrides.

---

## 8. Required implementation phases

Produce a concrete phased plan. Use this baseline and refine it based on repository evidence.

| Phase | Goal | Likely owner plan |
|-------|------|-------------------|
| `TOKEN-1` | Feature architecture and domain sync | `docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md` + affected domain docs |
| `TOKEN-2` | OutputPolicy runtime | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` / `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` |
| `TOKEN-3` | ToolSchemaOptimizer | `docs/project/maintainers/plans/TOOLS.md` |
| `TOKEN-4` | ContextPackOptimizer | `docs/project/maintainers/plans/CONTEXT_ENGINEERING.md` |
| `TOKEN-5` | MemorySummaryCompressor | `docs/project/maintainers/plans/MEMORY.md` |
| `TOKEN-6` | Telemetry and regression gates | `docs/project/maintainers/plans/OBSERVABILITY.md` + affected domain plans |
| `TOKEN-7` | Adaptive optimization | `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md` |

---

## 9. Required output format

Return the audit in this structure:

```text
# Token Optimization Adoption Audit

## 1. Verdict
- multi-layer feature placement
- maturity target
- highest ROI slice

## 2. Existing Intergrax capabilities already covering this
- file/path evidence
- what not to duplicate

## 3. Gaps
- P0/P1/P2/P3 table
- owner domain
- required feature doc update
- required domain doc update
- required implementation slice

## 4. Feature docs update plan
- docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md sections
- docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md sections

## 5. Domain architecture update plan
- exact files
- exact sections to add/change
- dependency order

## 6. Domain plan update plan
- exact plan files
- proposed TOKEN-* rows
- acceptance criteria

## 7. ADR plan
- ADR IDs
- decision scope
- alternatives considered

## 8. Implementation roadmap
- PR order
- per PR read scope
- per PR edit scope
- tests/gates

## 9. Risk analysis
- quality risk
- safety risk
- schema break risk
- observability risk
- cost-vs-savings risk

## 10. Cursor handoff prompts
- prompt for feature docs sync
- prompt for TOKEN-2 implementation
- prompt for TOKEN-3 implementation
```

---

## 10. Non-negotiable constraints

- Do not implement runtime code during architecture-adoption-audit mode.
- Do not create large unscoped PRs.
- Do not create `docs/project/maintainers/plans/TOKEN_OPTIMIZATION.md`.
- Do not duplicate existing Context Engineering budget/preflight mechanisms.
- Do not compress executable code.
- Do not compress tool call payloads by default.
- Do not compress strict JSON schema semantics.
- Do not report token savings without quality/safety validation.
- Do not overwrite persistent files before compression validation.
- Preserve Intergrax tier boundaries.
- Preserve domain architecture/plan 1:1 pairs.
- Preserve feature architecture/plan 1:1 pairs.
- Preserve existing commit-per-focused-task discipline.

---

## 11. Recommended first execution result

The ideal first Cursor result should be a docs-only adoption plan, not code.

It should answer:

```text
Where does Token Optimization live in docs/project/capabilities?
Which existing layers change?
Which domain plan rows are required?
Which ADR comes first?
Which implementation slice has highest ROI?
```

Expected likely conclusion:

```text
Accept Token Optimization as a multi-layer feature anchored in CONTEXT_ENGINEERING, with first implementation slice focused on OutputPolicy and ToolSchemaOptimizer before deeper semantic context compression.
```

---END PROMPT---

---

## TOKEN-10 future audit scope (TOKEN-10A canon)

When implementing or auditing TOKEN-10, validate in addition to existing TOKEN-1..9 checks:

1. Cache-aware lifecycle wired end-to-end (stable prefix → router → gate → pipeline).
2. Helper-level prefix contracts connected to runtime assembly (TOKEN-10B).
3. LLM_ADAPTERS owns provider cache signals; Token Optimization does not create private vLLM client.
4. Content-reduction and prefix-cache metrics remain separable in receipts, proof, and public claims.
5. Universal proof harness uses production router/pipeline — no test-only engine.
6. vLLM proof: cold, warm, changed-prefix negative control; fail when cache evidence missing.
7. In-cache compaction (TOKEN-10E): explicit opt-in, no default production enablement.
8. LKW-PF6 follows TOKEN-10G; LKW does not duplicate platform mechanisms.
9. README promotion only in TOKEN-10H.

Sync checklist: docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md §8.9; docs/project/capabilities/TOKEN_OPTIMIZATION_CLAIMS.md; docs/project/maintainers/audit/TOKEN_OPTIMIZATION_DOCS_SYNC.md.
