<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Architecture and Plan Docs Sync Instruction

**Status:** Docs-sync control prompt (copy-paste for Cursor / LLM agent)  
**Input document:** [`docs/audit/TOKEN_OPTIMIZATION.md`](TOKEN_OPTIMIZATION.md)  
**Target branch:** `development`  
**Mode:** documentation update only — architecture + implementation plans  
**Runtime implementation:** forbidden in this task  
**Primary goal:** integrate Token Optimization into the relevant Intergrax architecture and plan documents so later implementation PRs can be executed in small, traceable slices.

---

## How to use

1. Open Cursor on repository `jakbuczarnecki/intergrax`.
2. Checkout branch `development`.
3. Start a fresh agent chat.
4. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
5. Do not add extra goals.
6. Let Cursor produce a focused documentation PR.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

repo: jakbuczarnecki/intergrax
branch: development
source_instruction: docs/audit/TOKEN_OPTIMIZATION.md
mode: docs-sync
runtime_code_changes: forbidden
commit_policy: one focused commit after docs sync

# ═══ END USER CONFIG ═══

# TASK: Adopt Token Optimization into Intergrax architecture and implementation plans

You are working on the Intergrax Harness AI repository.

Your task is to use `docs/audit/TOKEN_OPTIMIZATION.md` as the source instruction and perform a **documentation-only architecture and plan sync**.

You must update the relevant Intergrax architecture documents and implementation plan documents so Token Optimization becomes a formally planned, traceable, implementable capability.

Do **not** implement runtime code.
Do **not** create Python modules.
Do **not** create tests yet.
Do **not** modify unrelated documentation.
Do **not** do a repo-wide audit.

This task is only about documentation alignment and implementation planning.

---

## 0. Mandatory working rules

Before editing, state:

```text
read scope:
edit scope:
tests/checks:
```

Follow Intergrax token budget rules:

- Read `docs/audit/TOKEN_OPTIMIZATION.md` first.
- Read only read-scope blocks and relevant sections of target documents.
- Use grep/path filters before opening large files.
- Do not load full architecture hubs unless necessary.
- Do not load full plan hubs unless necessary.
- Do not load `docs/audit_results/`.
- Do not use subagents.
- If a required document has no obvious insertion point, add a small, clearly marked section rather than rewriting the whole file.
- Keep changes minimal but sufficient.
- Preserve existing layer ownership and tier boundaries.
- Make one focused commit at the end.

---

## 1. Source document

Mandatory source:

```text
docs/audit/TOKEN_OPTIMIZATION.md
```

You must extract from it:

- target architecture concept,
- owner domains,
- integration points,
- safety rules,
- proposed capability model,
- implementation phases `TOKEN-1` through `TOKEN-7`,
- required docs/plan/ADR direction,
- non-negotiable constraints.

Treat this file as the operator-approved adoption instruction.

---

## 2. Required architecture updates

Update only the documents that actually need a reference to Token Optimization.

### 2.1 Mandatory architecture documents

Update these documents:

```text
docs/architecture/CONTEXT_ENGINEERING.md
docs/architecture/LLM_ADAPTERS.md
docs/architecture/TOOLS.md
docs/architecture/MEMORY.md
docs/architecture/OBSERVABILITY.md
docs/architecture/UNIFIED_EXECUTION_RUNTIME.md
```

### 2.2 Conditional architecture documents

Update these only if the source instruction and existing architecture make the relationship necessary:

```text
docs/architecture/RAG.md
docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md
docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md
docs/intergrax_runtime_architecture.md
docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md
```

Do not force changes if a document would only receive a vague mention.

---

## 3. Required plan updates

Create or update plan documents so implementation can proceed later in small PRs.

### 3.1 Create primary plan hub if it does not exist

Create:

```text
docs/plan/TOKEN_OPTIMIZATION.md
```

This file must become the canonical implementation plan for Token Optimization.

It must include:

- read-scope block,
- relationship to `docs/audit/TOKEN_OPTIMIZATION.md`,
- owner/related domains,
- phases `TOKEN-1` through `TOKEN-7`,
- implementation order,
- acceptance criteria per phase,
- required tests/gates per phase,
- explicit exclusions,
- delivery rule: one TOKEN-* slice per PR.

### 3.2 Update related plan hubs

Update these plan documents with cross-reference rows or backlog entries:

```text
docs/plan/CONTEXT_ENGINEERING.md
docs/plan/LLM_ADAPTERS.md
docs/plan/TOOLS.md
docs/plan/MEMORY.md
docs/plan/OBSERVABILITY.md
docs/plan/UNIFIED_EXECUTION_RUNTIME.md
```

Conditional plan docs:

```text
docs/plan/RAG.md
docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md
docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md
```

Only update conditional docs if a real TOKEN phase depends on them.

---

## 4. Required architecture content by layer

### 4.1 `CONTEXT_ENGINEERING`

Add Token Optimization as a first-class stage or extension of the context lifecycle.

Target lifecycle:

```text
COLLECT → NORMALIZE → SCORE → FILTER → RANK → BUDGET → TOKEN_OPTIMIZE → FORMAT → TOKEN_PREFLIGHT → VALIDATE → EMIT
```

Document that `CONTEXT_ENGINEERING` owns:

- `ContextPackOptimizer`,
- post-ranking context compression,
- source-aware compression policy application,
- post-compression token recalculation,
- provenance links to compression receipts,
- fallback to original fragments on validation failure.

Do not duplicate existing ContextCompiler, budget, degradation ladder, or adapter-token preflight mechanisms. Token Optimization must extend them.

### 4.2 `LLM_ADAPTERS`

Document that LLM adapters provide:

- tokenizer-consistent token counting,
- context window metadata,
- output budget metadata,
- model cost/latency signals where available.

Token Optimization consumes these signals. It does not replace adapter contracts unless a later TOKEN phase explicitly adds a missing method.

### 4.3 `TOOLS`

Document `ToolSchemaOptimizer` as a planned capability.

Rules:

- may compress tool `description` fields,
- may compress natural-language examples,
- must preserve tool names,
- must preserve parameter names,
- must preserve enum values,
- must preserve required fields,
- must preserve JSON schema semantics,
- must not compress tool call payloads by default,
- must not compress tool results unless the result type explicitly allows it.

### 4.4 `MEMORY`

Document `MemorySummaryCompressor` as a planned capability.

Rules:

- no live overwrite before validation,
- staging output first,
- protected-region validation,
- semantic validation for lossy summaries,
- compression receipt,
- rollback metadata.

### 4.5 `RAG`

If updated, document that RAG chunk compression is after retrieval/ranking, not before.

Rules:

- preserve citations,
- preserve source spans where needed,
- compression must not damage answer grounding,
- RAG compression is not the first implementation slice unless plan evidence justifies it.

### 4.6 `OBSERVABILITY`

Document events/counters for token optimization.

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

Candidate counters:

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

### 4.7 `UNIFIED_EXECUTION_RUNTIME`

Document runtime policy resolution:

- token optimization policy selection,
- output profile selection,
- compression level selection,
- safety bypass enforcement,
- environment/profile-level overrides.

### 4.8 `AGENT_CONTRACTS_AND_ASSEMBLY`

If updated, document that agents may declare hints, not manually assemble prompts:

- desired output profile,
- required context compactness,
- sources that must not be compressed,
- citation/evidence sensitivity.

### 4.9 `ADAPTIVE_HARNESS_INTELLIGENCE`

If updated, document TOKEN-7 as later adaptive optimization:

- learn optimal budgets from telemetry,
- recommend compact/full profile by task type,
- escalate context when quality drops.

---

## 5. Required `docs/plan/TOKEN_OPTIMIZATION.md` structure

Create the plan hub with this structure:

```markdown
# Token Optimization — Implementation Plan

**Architecture source:** docs/audit/TOKEN_OPTIMIZATION.md
**Primary owner:** CONTEXT_ENGINEERING
**Related domains:** ...
**Status:** Planned

---

## Cursor read scope (token budget)

...

---

## Phase TOKEN-1 — Architecture and contracts
...

## Phase TOKEN-2 — OutputPolicy runtime
...

## Phase TOKEN-3 — ToolSchemaOptimizer
...

## Phase TOKEN-4 — ContextPackOptimizer
...

## Phase TOKEN-5 — MemorySummaryCompressor
...

## Phase TOKEN-6 — Telemetry and regression gates
...

## Phase TOKEN-7 — Adaptive optimization
...

---

## Delivery rules
...

## Explicit exclusions
...

## Gates
...
```

Each TOKEN phase must include:

```text
Goal
Owner layer
Dependencies
Deliverables
Acceptance criteria
Required tests/checks
Status
```

Status should initially be `Planned` unless existing code already satisfies a phase.

---

## 6. Required ADR plan

Do not necessarily create ADR files in this docs-sync task unless repository convention clearly expects it now.

At minimum, add to `docs/plan/TOKEN_OPTIMIZATION.md` an ADR queue:

```text
ADR-TOKEN-001 — Token Optimization domain boundary and ownership
ADR-TOKEN-002 — Compression receipts and protected-region validation
ADR-TOKEN-003 — Tool schema optimization safety model
```

If ADR directories and naming conventions are obvious and lightweight, you may create ADR stubs. Otherwise, leave ADR creation as TOKEN-1 acceptance criteria.

---

## 7. Required safety constraints to document

The docs must explicitly state:

- Token Optimization does not compress private chain-of-thought.
- Token Optimization does not mutate executable code.
- Token Optimization does not rewrite strict JSON schema semantics.
- Token Optimization does not compress tool call payloads by default.
- Token Optimization does not remove required audit evidence.
- Token Optimization must preserve protected regions.
- Token Optimization must produce receipts for persistent or lossy compression.
- Token savings are invalid without quality/safety validation.

---

## 8. Tests/checks for this docs-only task

Run only lightweight documentation checks that already exist and are relevant.

Suggested discovery:

```bash
ls scripts | grep -E "check|doc|plan|arch|audit"
```

Run only scripts that are clearly safe and documentation-related.

Do not run full test suite.

If no relevant doc checks exist, state:

```text
Tests not run — docs-only change; no matching lightweight doc gate found.
```

---

## 9. Expected final response

Return:

```text
Outcome:
- what was updated

Files changed:
- ...

Plan created:
- docs/plan/TOKEN_OPTIMIZATION.md

Architecture layers synced:
- ...

Implementation phases added:
- TOKEN-1 ... TOKEN-7

Checks:
- commands run / not run and why

Next Cursor task:
- recommended first implementation slice
```

---

## 10. Commit requirement

After edits and checks, create one focused commit on `development`.

Suggested commit message:

```text
docs: add token optimization architecture and plan sync
```

Do not group runtime implementation with this commit.

---END PROMPT---
