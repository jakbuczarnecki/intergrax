---
id: IJ-2026-06-17-017
date: 2026-06-17
tiers:
  - tier-0
scope: PLATFORM_FOUNDATION
plan_ref:
  - P2-ARCH-01
status: completed
commit: aeba83f1
adr: none — index-only guide; domain canon unchanged
---

# P2-ARCH-01 — System invariants cross-domain index

## Operator request

Build a full `SYSTEM_INVARIANTS.md` guide consolidating 10–20 non-negotiable architectural rules scattered across domain pairs, link from hub and AGENTS.md, and commit.

## Summary

Added `docs/guides/SYSTEM_INVARIANTS.md` with 31 `SYS-INV-*` rules across tier boundaries, execution path, L1–L4 responsibility split, platform gateways, cognition planes, and governance. Each row links to canonical domain sections and CI scripts where enforced. Linked from hub, AGENTS.md, llms.txt, and PLATFORM_FOUNDATION architecture/plan (P2-ARCH-01 Done).

## Project impact

Architects, reviewers, and LLM agents now have one onboarding page for “never violate” rules before opening a domain pair — closes external audit gap P2-ARCH-01 without creating a 23rd domain pair or duplicating ACP/APP/ORCH tables.

## Traceability

| Link | Target |
|------|--------|
| Guide | `docs/guides/SYSTEM_INVARIANTS.md` |
| Plan | `docs/plan/PLATFORM_FOUNDATION.md` P2-ARCH-01 |
| Architecture pointer | `docs/architecture/PLATFORM_FOUNDATION.md` Dependency Direction |
| Hub | `docs/intergrax_runtime_architecture.md` header |

## Changed artifacts

- `docs/guides/SYSTEM_INVARIANTS.md` — new index
- `docs/intergrax_runtime_architecture.md` — hub link
- `AGENTS.md` — before-you-write + task routing
- `llms.txt` — crawler pointer
- `docs/architecture/PLATFORM_FOUNDATION.md` — cross-ref
- `docs/plan/PLATFORM_FOUNDATION.md` — P2-ARCH-01 Done

## Verification

```bash
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
```

Result: pass (expected).

## Risks and follow-ups

- Keep §5 rows in sync when ACP-INV / APP-INV semantics change — index only, not second canon.
- Optional future gate: verify anchor links in SYSTEM_INVARIANTS.md still resolve.
