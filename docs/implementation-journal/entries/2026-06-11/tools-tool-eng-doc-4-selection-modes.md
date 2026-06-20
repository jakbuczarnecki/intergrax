---
id: IJ-2026-06-11-001
date: 2026-06-11
tiers:
  - tier-0
  - tier-1
scope: TOOLS
plan_ref:
  - TOOL-ENG-DOC.4
  - TOOL-ENG-13
  - TOOL-ENG-14
  - TOOL-ENG-15
status: completed
commit: pending
adr: none — documentation-only canon; ADR deferred to TOOL-ENG-13/14 implementation
---

# Tool selection modes canon — standard, semantic, hierarchical (TOOL-ENG-DOC.4)

## Operator request

Audit whether the Tools layer architecture documents production tool-selection strategies (standard full-catalog LLM choice, semantic vector pre-filter, hierarchical category traversal) and align architecture with the implementation plan.

## Summary

Added canonical **Tool selection modes (production strategies)** section to `architecture/TOOLS.md`: standard, semantic, and hierarchical modes with flow diagram, `ToolSelectionMode` mapping, and distinction from L0–L7 policy layers. Corrected stale L6 maturity claims (`retrieval_top_k` is keyword overlap, not semantic). Registered **TOOL-ENG-13** (semantic tool index), **TOOL-ENG-14** (hierarchical traversal), **TOOL-ENG-15** (naming clarity). Synced `plan/TOOLS.md`, `NEXUS_EXECUTION_FLOW.md` §15, `REASONING_AND_COGNITION.md` §13, audit prompt, and Appendix J.

## Project impact

Harness operators and implementers have a single canon for scaling tool selection beyond ~30 tools. Implementation queue clearly separates shipped keyword pre-filter from planned embedding-based and hierarchical strategies.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TOOLS.md` §Tool selection modes |
| Plan | `docs/plan/TOOLS.md` TOOL-ENG-DOC.4, TOOL-ENG-13/14/15 |
| Flow | `docs/architecture/NEXUS_EXECUTION_FLOW.md` §15 |
| ADR | none — DOC only |

## Changed artifacts

- `docs/architecture/TOOLS.md` — selection modes canon, maturity matrix, gap register
- `docs/plan/TOOLS.md` — TOOL-ENG-DOC.4 Done; TOOL-ENG-13/14/15 Planned; queue §6.1e
- `docs/architecture/NEXUS_EXECUTION_FLOW.md` — §15 diagram + L6/L6b stages
- `docs/architecture/REASONING_AND_COGNITION.md` — §13 tool_selection cross-ref
- `docs/audit/TOOLS.md` — open gaps + audit dimension 19
- `docs/guides/AGENT_CREATION_GUIDE.md` — Appendix J selection mode surface

## Verification

```bash
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
```

Result: pass (docs-only iteration).

## Risks and follow-ups

- **TOOL-ENG-13** requires ADR before semantic index implementation; reuse RAG `embedding_manager` without conflating document and tool collections.
- **TOOL-ENG-14** needs taxonomy decision (bundle vs `category` vs host-defined tree).
- **TOOL-ENG-15** optional enum alias — no behavior change until host migration.
