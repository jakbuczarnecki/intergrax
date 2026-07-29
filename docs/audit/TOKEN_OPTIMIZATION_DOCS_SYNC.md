<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Feature and Domain Docs Sync Instruction

**Status:** Docs-sync control prompt (copy-paste for Cursor / LLM agent)  
**Input document:** [`docs/audit/TOKEN_OPTIMIZATION.md`](TOKEN_OPTIMIZATION.md)  
**Feature architecture target:** [`../features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)  
**Feature plan target:** [`../features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)  
**Target branch:** `development`  
**Mode:** documentation update only — feature docs + affected domain docs  
**Runtime implementation:** forbidden in this task

---

## Why this instruction exists

Token Optimization is a **multi-layer feature**, not a standalone domain-layer plan by default.

Preserve both documentation pair rules:

```text
docs/architecture/<DOMAIN>.md           ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

Do **not** create `docs/plan/TOKEN_OPTIMIZATION.md` unless the operator explicitly promotes Token Optimization into a full architecture domain with a matching `docs/architecture/TOKEN_OPTIMIZATION.md`.

---

## How to use

1. Open Cursor on repository `jakbuczarnecki/intergrax`.
2. Checkout branch `development`.
3. Start a fresh agent chat.
4. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
5. Do not add runtime implementation goals.
6. Let Cursor produce a focused documentation PR.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

repo: jakbuczarnecki/intergrax
branch: development
source_instruction: docs/audit/TOKEN_OPTIMIZATION.md
feature_architecture: docs/features/architecture/TOKEN_OPTIMIZATION.md
feature_plan: docs/features/plan/TOKEN_OPTIMIZATION.md
mode: feature-docs-sync
runtime_code_changes: forbidden
commit_policy: one focused commit after docs sync

# ═══ END USER CONFIG ═══

# TASK: Adopt Token Optimization into Intergrax feature docs and affected domain docs

You are working on the Intergrax Harness AI repository.

Use `docs/audit/TOKEN_OPTIMIZATION.md` as the source instruction and perform a **documentation-only sync**.

Token Optimization is a **multi-layer feature**. It must be documented in:

```text
docs/features/architecture/TOKEN_OPTIMIZATION.md
docs/features/plan/TOKEN_OPTIMIZATION.md
```

Then update affected domain architecture and plan files with cross-references and concrete implementation rows.

Do **not** implement runtime code.
Do **not** create Python modules.
Do **not** create tests yet.
Do **not** create `docs/plan/TOKEN_OPTIMIZATION.md`.
Do **not** do a repo-wide audit.

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
- Read `docs/features/README.md` to understand multi-layer feature structure.
- Read existing `docs/features/architecture/TOKEN_OPTIMIZATION.md` and `docs/features/plan/TOKEN_OPTIMIZATION.md` if present.
- Read only read-scope blocks and relevant sections of target domain documents.
- Use grep/path filters before opening large files.
- Do not load full architecture hubs unless necessary.
- Do not load full plan hubs unless necessary.
- Do not load `docs/audit_results/`.
- Do not use subagents.
- Keep changes minimal but sufficient.
- Preserve existing layer ownership and tier boundaries.
- Make one focused commit at the end.

---

## 1. Required feature docs updates

Update or create:

```text
docs/features/architecture/TOKEN_OPTIMIZATION.md
docs/features/plan/TOKEN_OPTIMIZATION.md
```

The feature architecture must define:

- purpose,
- scope,
- out-of-scope items,
- domain ownership matrix,
- protected-region policy,
- context lifecycle integration,
- tool schema optimization rules,
- memory compression safety rules,
- observability requirements.

The feature plan must define:

- TOKEN-1 through TOKEN-7 phases,
- owner domain per phase,
- dependencies,
- acceptance criteria,
- required checks,
- ADR queue,
- delivery rules,
- explicit exclusions,
- mapping from feature phases to domain plan files.

---

## 2. Required platform docs updates

Update these docs only if they do not already describe the multi-layer feature structure:

```text
docs/intergrax_runtime_architecture.md
docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md
docs/audit/README.md
```

They must explain that Intergrax now has two paired doc structures:

```text
docs/architecture/<DOMAIN>.md           ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

---

## 3. Required affected domain docs updates

Update affected domain docs only with precise cross-references and implementation rows. Do not rewrite whole files.

Mandatory domains:

```text
CONTEXT_ENGINEERING
LLM_ADAPTERS
TOOLS
MEMORY
OBSERVABILITY
UNIFIED_EXECUTION_RUNTIME
```

Conditional domains, only if required by the feature plan:

```text
RAG
AGENT_CONTRACTS_AND_ASSEMBLY
ADAPTIVE_HARNESS_INTELLIGENCE
```

For each affected domain:

1. Add a short architecture reference to `docs/features/architecture/TOKEN_OPTIMIZATION.md`.
2. Add concrete plan rows to the domain plan file for the relevant TOKEN phase.
3. Preserve existing plan ordering and status conventions.
4. Do not duplicate feature-level details unnecessarily.

---

## 4. Domain-specific sync requirements

### CONTEXT_ENGINEERING

Add Token Optimization as a feature-level extension of the context lifecycle:

```text
COLLECT → NORMALIZE → SCORE → FILTER → RANK → BUDGET → TOKEN_OPTIMIZE → FORMAT → TOKEN_PREFLIGHT → VALIDATE → EMIT
```

Plan rows should cover `TOKEN-4 ContextPackOptimizer`.

### LLM_ADAPTERS

Document that Token Optimization consumes tokenizer-consistent counting, context window metadata, output budget metadata, and cost/latency signals.

Plan rows should cover any missing adapter signal needed by TOKEN phases.

### TOOLS

Plan rows should cover `TOKEN-3 ToolSchemaOptimizer`.

Rules:

- preserve tool names,
- preserve parameter names,
- preserve enum values,
- preserve required fields,
- preserve JSON schema semantics,
- do not compress tool call payloads by default.

### MEMORY

Plan rows should cover `TOKEN-5 MemorySummaryCompressor`.

Rules:

- no live overwrite before validation,
- staging output first,
- protected-region validation,
- receipt,
- rollback metadata.

### OBSERVABILITY

Plan rows should cover `TOKEN-6 Telemetry and regression gates`.

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

### UNIFIED_EXECUTION_RUNTIME

Plan rows should cover `TOKEN-2 OutputPolicy runtime` and runtime policy enforcement.

---

## 5. ADR plan

Do not create ADR files unless the repo convention and scope clearly support it in this docs-sync task.

At minimum, ensure `docs/features/plan/TOKEN_OPTIMIZATION.md` contains:

```text
ADR-TOKEN-001 — Multi-layer feature boundary and ownership
ADR-TOKEN-002 — Compression receipts and protected-region validation
ADR-TOKEN-003 — Tool schema optimization safety model
```

---

## 6. Checks

Run lightweight doc checks only.

Required if available:

```bash
uv run python scripts/audit/check_docs_domain_pairs.py
```

Do not run the full test suite.

If checks cannot be run, state why.

---

## 7. Expected final response

Return:

```text
Outcome:
- what was updated

Files changed:
- ...

Feature docs:
- docs/features/architecture/TOKEN_OPTIMIZATION.md
- docs/features/plan/TOKEN_OPTIMIZATION.md

Domain docs synced:
- ...

Checks:
- commands run / not run and why

Next Cursor task:
- recommended first implementation slice
```

---

## 8. Commit requirement

After edits and checks, create one focused commit on `development`.

Suggested commit message:

```text
docs: add token optimization multi-layer feature docs
```

Do not group runtime implementation with this commit.

---END PROMPT---

---

## TOKEN-10 documentation sync expectations (TOKEN-10A)

After each TOKEN-10 subtask closeout, verify:

| Document pair / file | Sync trigger |
|---------------------|--------------|
| features/architecture + plan TOKEN_OPTIMIZATION | lifecycle, roadmap status, proof gates |
| TOKEN_OPTIMIZATION_CACHE_PREFIX_STABILIZATION.md | runtime wiring status, in-cache compaction phase |
| LLM_ADAPTERS architecture + plan | TOKEN-LLM-2/3 row status |
| TOKEN_OPTIMIZATION_CLAIMS.md | allowed vs proof-gated wording |
| LKW ARCHITECTURE, IMPLEMENTATION_PLAN, PLATFORM_PROOF_LOOP | LKW-PF6 ordering only when product proof scope changes |
| ROADMAP.md, features/README.md | concise status pointer |
| Satellites (architecture + plan cross-refs) | TOKEN-10 phase → owning plan mapping |

Stale wording to reject: runtime/provider integration deferred indefinitely; TOKEN-9 as final phase; LKW as first proof owner; in-cache compaction as undefined future only.

README main promotion: TOKEN-10H only.
