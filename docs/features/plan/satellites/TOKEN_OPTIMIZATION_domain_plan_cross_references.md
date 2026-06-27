<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# TOKEN_OPTIMIZATION — domain plan cross-references

**Parent hub:** [`TOKEN_OPTIMIZATION.md`](../TOKEN_OPTIMIZATION.md)  
**Feature architecture (1:1):** [`../../architecture/TOKEN_OPTIMIZATION.md`](../../architecture/TOKEN_OPTIMIZATION.md)  
**Architecture satellite:** [`../../architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](../../architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md)  
**Source audit:** [`../../../audit/TOKEN_OPTIMIZATION.md`](../../../audit/TOKEN_OPTIMIZATION.md)  
**Purpose:** Ensure every domain plan participating in `TOKEN_OPTIMIZATION` has visible cross-references, required TOKEN phase rows, and alignment with the feature coordination map.

---

## Why this file exists

Multi-layer feature plans coordinate cross-domain delivery. Concrete implementation rows still belong in owning `docs/plan/<DOMAIN>.md` files.

This satellite is the canonical checklist for domain plan cross-reference sync and required TOKEN rows before runtime implementation begins.

---

## Required cross-reference wording pattern

Each participating domain plan should contain a concise line near the header/read-scope block:

```markdown
**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). <DOMAIN-SPECIFIC OWNERSHIP SENTENCE>
```

Use relative links appropriate to `docs/plan/<DOMAIN>.md`:

```markdown
[`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)
[`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)
```

Ownership sentences match the architecture satellite: [`../../architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](../../architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md).

---

## Domain plan row checklist

Before runtime implementation starts, the following domain plan rows must exist:

| Domain plan | Required rows |
|-------------|---------------|
| `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` | `TOKEN-UER-1`, `TOKEN-UER-2` |
| `docs/plan/TOOLS.md` | `TOKEN-TOOLS-1` |
| `docs/plan/CONTEXT_ENGINEERING.md` | `TOKEN-CE-1`, `TOKEN-CE-2` |
| `docs/plan/MEMORY.md` | `TOKEN-MEM-1` |
| `docs/plan/OBSERVABILITY.md` | `TOKEN-OBS-1`, `TOKEN-OBS-2` |
| `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` | `TOKEN-AHI-1` |
| `docs/plan/LLM_ADAPTERS.md` | `TOKEN-LLM-1` reference row only; no duplicate tokenizer |

---

## TOKEN phase → owning plan file

| TOKEN phase | Owning plan file |
|-------------|------------------|
| `TOKEN-1` shared contracts, receipts, protected regions | feature plan + `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` |
| `TOKEN-2` OutputPolicy runtime | `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` and optional `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` |
| `TOKEN-3` ToolSchemaOptimizer | `docs/plan/TOOLS.md` |
| `TOKEN-4` ContextPackOptimizer | `docs/plan/CONTEXT_ENGINEERING.md` |
| `TOKEN-5` MemorySummaryCompressor | `docs/plan/MEMORY.md` |
| `TOKEN-6` telemetry and regression gates | `docs/plan/OBSERVABILITY.md` plus affected domain plans |
| `TOKEN-7` adaptive optimization | `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` |

---

## Sync checklist

Before declaring `TOKEN_OPTIMIZATION` plan adoption complete, verify:

- [ ] Each domain plan in the row checklist contains the cross-feature reference.
- [ ] Each required TOKEN row exists in the owning domain plan with Status and acceptance criteria.
- [ ] Domain plan TOKEN rows link back to [`../TOKEN_OPTIMIZATION.md`](../TOKEN_OPTIMIZATION.md) and [`../../architecture/TOKEN_OPTIMIZATION.md`](../../architecture/TOKEN_OPTIMIZATION.md).
- [ ] Architecture cross-references in domain architecture files match [`../../architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](../../architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md).
- [ ] `docs/plan/TOKEN_OPTIMIZATION.md` does **not** exist (feature stays multi-layer).
- [ ] `uv run python scripts/audit/check_docs_domain_pairs.py` remains green.

---

## Cursor/local patch instruction

Use this when adding domain plan rows or cross-references for `TOKEN_OPTIMIZATION`:

```text
We are working on repository `jakbuczarnecki/intergrax`, branch `development`.

Session goal:
Add missing cross-feature references and TOKEN phase rows to domain plan files for the multi-layer feature `TOKEN_OPTIMIZATION`.

Do not implement code.
Do not create `docs/plan/TOKEN_OPTIMIZATION.md`.
Do not rewrite entire documents.

Source documents:
- docs/features/plan/TOKEN_OPTIMIZATION.md
- docs/features/architecture/TOKEN_OPTIMIZATION.md
- docs/features/plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md
- docs/features/architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md

Edit only the domain plan files listed in the domain plan row checklist above.

In each file, add one concise `Cross-feature — Token Optimization` entry in the header/read-scope area.
Add or verify TOKEN-* phase rows cited in this satellite.

Run:
uv run python scripts/audit/check_docs_domain_pairs.py

Commit:
docs: add token optimization domain plan cross-references
```
