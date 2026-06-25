<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Domain Architecture Cross-reference Map

**Status:** Required domain architecture sync map  
**Feature architecture:** [`../architecture/TOKEN_OPTIMIZATION.md`](../architecture/TOKEN_OPTIMIZATION.md)  
**Feature plan:** [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md)  
**Source audit:** [`../../audit/TOKEN_OPTIMIZATION.md`](../../audit/TOKEN_OPTIMIZATION.md)  
**Purpose:** ensure every domain architecture participating in the `TOKEN_OPTIMIZATION` multi-layer feature has a visible cross-reference and a precise ownership statement.

---

## Why this file exists

`TOKEN_OPTIMIZATION` is a multi-layer feature. Its feature architecture and feature plan coordinate the cross-domain capability, but each participating domain architecture must still know how it participates.

This file is the canonical checklist for the remaining domain architecture cross-reference sync.

After direct domain docs are updated, this file remains useful as an audit checklist and review map.

---

## Required cross-reference wording pattern

Each participating domain architecture should contain a concise line or section near the header/read-scope block:

```markdown
**Cross-feature — Token Optimization:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). <DOMAIN-SPECIFIC OWNERSHIP SENTENCE>
```

Use relative links appropriate to `docs/architecture/<DOMAIN>.md`:

```markdown
[`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)
[`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)
```

---

## Mandatory domain architecture cross-references

| Domain architecture | Required ownership sentence |
|---------------------|-----------------------------|
| `docs/architecture/CONTEXT_ENGINEERING.md` | `CONTEXT_ENGINEERING owns ContextPackOptimizer, source-aware context compression, post-compression token recalculation, receipt references in provenance/metadata, and fallback to original fragments on validation failure.` |
| `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md` | `UNIFIED_EXECUTION_RUNTIME owns runtime token policy resolution, output profile resolution, compression-level selection, and safety bypass enforcement.` |
| `docs/architecture/TOOLS.md` | `TOOLS owns ToolSchemaOptimizer and compact LLM-facing tool catalog presentation while preserving canonical ToolContract semantics and tool call payloads.` |
| `docs/architecture/MEMORY.md` | `MEMORY owns MemorySummaryCompressor for persistent natural-language summaries where staging, validation, compression receipts, and rollback metadata exist.` |
| `docs/architecture/OBSERVABILITY.md` | `OBSERVABILITY owns token optimization telemetry, savings attribution, optimization receipt visibility, typed diagnostic payloads, metrics, and regression-gate reporting through the Harness Observability Spine.` |
| `docs/architecture/LLM_ADAPTERS.md` | `LLM_ADAPTERS provides tokenizer-consistent token counting, context window metadata, usage accounting, and model/cost signals consumed by Token Optimization; Token Optimization must not create a parallel tokenizer.` |

---

## Conditional domain architecture cross-references

| Domain architecture | When to update | Required ownership sentence |
|---------------------|----------------|-----------------------------|
| `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | When adding `TOKEN-ACP-1` agent-level hints | `AGENT_CONTRACTS_AND_ASSEMBLY may expose declarative agent-level output/context compactness hints, but agents must not manually assemble prompts or import token optimization internals.` |
| `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` | Before implementing `TOKEN-AHI-1` | `ADAPTIVE_HARNESS_INTELLIGENCE may consume token optimization telemetry to recommend budgets and profiles, but production auto-apply remains forbidden until governance and quality gates explicitly allow it.` |
| `docs/architecture/RAG.md` | Before enabling RAG chunk compression | `RAG chunk compression, if enabled, must happen after retrieval/ranking and must preserve citations, source spans, and answer grounding.` |

---

## Sync checklist

Before declaring `TOKEN_OPTIMIZATION` architecture adoption complete, verify:

- [ ] `docs/architecture/CONTEXT_ENGINEERING.md` contains the cross-feature reference.
- [ ] `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md` contains the cross-feature reference.
- [ ] `docs/architecture/TOOLS.md` contains the cross-feature reference.
- [ ] `docs/architecture/MEMORY.md` contains the cross-feature reference.
- [ ] `docs/architecture/OBSERVABILITY.md` contains the cross-feature reference.
- [ ] `docs/architecture/LLM_ADAPTERS.md` contains the cross-feature reference.
- [ ] Conditional refs are added before their TOKEN slices begin.
- [ ] `docs/features/plan/TOKEN_OPTIMIZATION.md` domain row checklist remains aligned with domain plan rows.
- [ ] `uv run python scripts/check_docs_domain_pairs.py` remains green.

---

## Cursor/local patch instruction

Use this when editing with local checkout or Cursor line-level patching:

```text
Pracujemy na repozytorium `jakbuczarnecki/intergrax`, branch `development`.

Cel sesji:
Dopisać brakujące cross-reference wpisy do domenowych dokumentów architektury dla multi-layer feature `TOKEN_OPTIMIZATION`.

Nie implementuj kodu.
Nie zmieniaj planu implementacji.
Nie twórz `docs/plan/TOKEN_OPTIMIZATION.md`.
Nie przepisuj całych dokumentów.

Źródło:
- docs/features/architecture/TOKEN_OPTIMIZATION.md
- docs/features/plan/TOKEN_OPTIMIZATION.md
- docs/features/satellites/TOKEN_OPTIMIZATION_DOMAIN_ARCHITECTURE_CROSS_REFERENCES.md

Edytuj minimalnie:
- docs/architecture/CONTEXT_ENGINEERING.md
- docs/architecture/UNIFIED_EXECUTION_RUNTIME.md
- docs/architecture/TOOLS.md
- docs/architecture/MEMORY.md
- docs/architecture/OBSERVABILITY.md
- docs/architecture/LLM_ADAPTERS.md

Opcjonalnie, tylko jeśli zaczynasz odpowiednie TOKEN slice:
- docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md
- docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md
- docs/architecture/RAG.md

W każdym pliku dodaj jeden krótki wpis `Cross-feature — Token Optimization` w header/read-scope area, z linkami do:
- ../features/architecture/TOKEN_OPTIMIZATION.md
- ../features/plan/TOKEN_OPTIMIZATION.md

i domenowym ownership sentence z checklisty.

Run:
uv run python scripts/check_docs_domain_pairs.py

Commit:
docs: add token optimization architecture cross-references
```
