---
id: IJ-2026-06-12-003
date: 2026-06-12
tiers:
  - tier-0
scope: MEMORY
plan_ref:
  - MEM-VEC-0.1
status: completed
commit: pending
adr: ADR-MEM-002 planned before MEM-VEC-2.1 implementation
---

# Memory vector recall — three-domain catalog and MEM-VEC plan

## Operator request

Clarify memory architecture so vector-backed semantic recall works for LTM and session conversation history, with plugin/config contracts for custom agents and applications, and update the implementation plan for this layer.

## Summary

Extended `architecture/MEMORY.md` with a normative three-domain vector index catalog (`knowledge`, `ltm`, `episodic`), Tier-3 wiring contracts for RAG stack injection into `UserProfileManager`, session turn indexing write path (target), semantic recall read paths for Context Engineering, `MemoryProfile` vector flags, and plugin surface. Opened **Phase MEM-VEC** (12 tasks) in `plan/MEMORY.md`. Cross-linked `CONTEXT_ENGINEERING.md` §14.2 and `SESSION_HISTORY_SEMANTIC` fragment source.

## Project impact

Harness operators and Tier-3 authors have a single canonical spec for end-to-end vector memory (facts + session turns) and a phased delivery register closing the audit gap between existing `UserProfileManager` vector code and default host wiring.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/MEMORY.md` §5.3, §6.4–6.5, §7.1.1, §11.5 |
| Plan | `docs/project/maintainers/plans/MEMORY.md` Phase MEM-VEC |
| ADR | `docs/project/technical/adr/entries/YYYY-MM-DD/ADR-MEM-002.md` (planned MEM-VEC-0.2) |
| Cross-domain | `docs/project/architecture/CONTEXT_ENGINEERING.md` §7.2, §14.2 |

## Changed artifacts

- `docs/project/architecture/MEMORY.md` — vector catalog, wiring contract, episodic recall target
- `docs/project/maintainers/plans/MEMORY.md` — Phase MEM-VEC register (MEM-VEC-0.1–3.2)
- `docs/project/architecture/CONTEXT_ENGINEERING.md` — semantic session recall use case + fragment source
- `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md` — `SessionTurnIndexStore` plugin row

## Verification

- Documentation-only iteration; implementation deferred to MEM-VEC-1.* onward.
- `python scripts/docs/check_docs_domain_pairs.py` — run before merge.

## Risks and follow-ups

- MEM-VEC-1.1–1.2 are P0 code changes — existing hosts with `enable_long_term_memory` currently get silent no-op semantic search.
- ADR-MEM-002 required before episodic index code (MEM-VEC-2.1).
