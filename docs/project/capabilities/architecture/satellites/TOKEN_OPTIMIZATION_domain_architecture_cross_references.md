<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# TOKEN_OPTIMIZATION — domain architecture cross-references

**Parent hub:** [`TOKEN_OPTIMIZATION.md`](../TOKEN_OPTIMIZATION.md)  
**Feature plan (1:1):** [`../../plan/TOKEN_OPTIMIZATION.md`](../../plan/TOKEN_OPTIMIZATION.md)  
**Plan satellite:** [`../../plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](../../plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md)  
**Source audit:** [`../../../../audit_results/TOKEN_OPTIMIZATION.md`](../../../../audit_results/TOKEN_OPTIMIZATION.md)
**Purpose:** Ensure every architecture domain participating in the `TOKEN_OPTIMIZATION` multi-layer feature has a visible cross-reference and a precise ownership statement.

---

## Why this file exists

`TOKEN_OPTIMIZATION` is a multi-layer feature. Its feature architecture and feature plan coordinate the cross-domain capability, but each participating domain architecture must still state how it participates.

This file is the canonical checklist for the remaining domain architecture cross-reference sync.

After direct domain architecture documents are updated, this file remains useful as an audit checklist and review map.

---

## Required cross-reference wording pattern

Each participating domain architecture should contain a concise line or section near the header/read-scope block:

```markdown
**Cross-feature — Token Optimization:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../features/architecture/TOKEN_OPTIMIZATION.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../../features/plan/TOKEN_OPTIMIZATION.md). <DOMAIN-SPECIFIC OWNERSHIP SENTENCE>
```

Use relative links appropriate to `docs/project/architecture/<DOMAIN>.md`:

```markdown
[`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)
[`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)
```

---

## Mandatory domain architecture cross-references

| Domain architecture | Required ownership sentence |
|---------------------|-----------------------------|
| `docs/project/architecture/CONTEXT_ENGINEERING.md` | `CONTEXT_ENGINEERING owns ContextPackOptimizer, source-aware context compression, post-compression token recalculation, receipt references in provenance/metadata, and fallback to original fragments on validation failure.` |
| `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | `UNIFIED_EXECUTION_RUNTIME owns runtime token policy resolution, output profile resolution, compression-level selection, and safety bypass enforcement.` |
| `docs/project/architecture/TOOLS.md` | `TOOLS owns ToolSchemaOptimizer and compact LLM-facing tool catalog presentation while preserving canonical ToolContract semantics and tool call payloads.` |
| `docs/project/architecture/MEMORY.md` | `MEMORY owns MemorySummaryCompressor for persistent natural-language summaries where staging, validation, compression receipts, and rollback metadata exist.` |
| `docs/project/architecture/OBSERVABILITY.md` | `OBSERVABILITY owns token optimization telemetry, savings attribution, optimization receipt visibility, typed diagnostic payloads, metrics, and regression-gate reporting through the Harness Observability Spine.` |
| `docs/project/architecture/LLM_ADAPTERS.md` | `LLM_ADAPTERS provides tokenizer-consistent token counting, context window metadata, usage accounting, and model/cost signals consumed by Token Optimization; Token Optimization must not create a parallel tokenizer.` |

---

## Conditional domain architecture cross-references

| Domain architecture | When to update | Required ownership sentence |
|---------------------|----------------|-----------------------------|
| `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | When adding `TOKEN-ACP-1` agent-level hints | `AGENT_CONTRACTS_AND_ASSEMBLY may expose declarative agent-level output/context compactness hints, but agents must not manually assemble prompts or import token optimization internals.` |
| `docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` | Before implementing `TOKEN-AHI-1` | `ADAPTIVE_HARNESS_INTELLIGENCE may consume token optimization telemetry to recommend budgets and profiles, but production auto-apply remains forbidden until governance and quality gates explicitly allow it.` |
| `docs/project/architecture/RAG.md` | Before enabling RAG chunk compression | `RAG chunk compression, if enabled, must happen after retrieval/ranking and must preserve citations, source spans, and answer grounding.` |

---

## Sync checklist

Before declaring `TOKEN_OPTIMIZATION` architecture adoption complete, verify:

- [ ] `docs/project/architecture/CONTEXT_ENGINEERING.md` contains the cross-feature reference.
- [ ] `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` contains the cross-feature reference.
- [ ] `docs/project/architecture/TOOLS.md` contains the cross-feature reference.
- [ ] `docs/project/architecture/MEMORY.md` contains the cross-feature reference.
- [ ] `docs/project/architecture/OBSERVABILITY.md` contains the cross-feature reference.
- [ ] `docs/project/architecture/LLM_ADAPTERS.md` contains the cross-feature reference.
- [ ] Conditional references are added before their TOKEN slices begin.
- [ ] [`../../plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](../../plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) domain plan row checklist remains aligned with domain plan rows.
- [ ] `uv run python scripts/docs/check_docs_domain_pairs.py` remains green.

---

## Cursor/local patch instruction

Use this when editing with a local checkout or Cursor line-level patching:

```text
We are working on repository `jakbuczarnecki/intergrax`, branch `development`.

Session goal:
Add the missing cross-reference entries to domain architecture documents for the multi-layer feature `TOKEN_OPTIMIZATION`.

Do not implement code.
Do not change the implementation plan.
Do not create `docs/project/maintainers/plans/TOKEN_OPTIMIZATION.md`.
Do not rewrite entire documents.

Source documents:
- docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md
- docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md
- docs/project/capabilities/architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md

Minimal edit scope:
- docs/project/architecture/CONTEXT_ENGINEERING.md
- docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md
- docs/project/architecture/TOOLS.md
- docs/project/architecture/MEMORY.md
- docs/project/architecture/OBSERVABILITY.md
- docs/project/architecture/LLM_ADAPTERS.md

Optional edit scope, only before starting the corresponding TOKEN slice:
- docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md
- docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md
- docs/project/architecture/RAG.md

In each file, add one concise `Cross-feature — Token Optimization` entry in the header/read-scope area, with links to:
- ../features/architecture/TOKEN_OPTIMIZATION.md
- ../features/plan/TOKEN_OPTIMIZATION.md

Use the domain-specific ownership sentence from this checklist.

Run:
uv run python scripts/docs/check_docs_domain_pairs.py

Commit:
docs: add token optimization architecture cross-references
```

---

## TOKEN-10 audit expectations (added TOKEN-10A)

When auditing TOKEN-10 implementation, verify:

- stable-prefix runtime wiring in real request assembly (not helper-only)
- append-only enforcement at runtime
- stable tool envelope fingerprinting and invalidation reasons
- provider cache evidence from LLM_ADAPTERS (TOKEN-LLM-2, TOKEN-LLM-3)
- vLLM proof reproducibility (cold/warm/changed-prefix controls)
- metric-family separation (content reduction vs prefix-cache reuse)
- universal proof ownership under intergrax/runtime/token_optimization/proof/ (not LKW)
- no LKW duplication of router, pipeline, cache gate, or proof harness
- README promotion gate — only TOKEN-10H after TOKEN-10G passes

Cross-domain rows: TOKEN-LLM-2, TOKEN-LLM-3 in docs/project/maintainers/plans/LLM_ADAPTERS.md; LKW-PF6-A..C in LKW docs after universal proof.
