---
id: IJ-2026-06-12-002
date: 2026-06-12
tiers:
  - tier-1
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-DOC
status: completed
commit: pending
adr: ADR-CTX-001
---

# Context Engineering — 22nd domain pair and plugin engine canon

## Operator request

Split Context Engineering from Memory into a dedicated Harness layer with full plugin-engine architecture, observability integration, and a complete implementation plan covering all gaps to production-grade extensibility (Cursor-class context assembly).

## Summary

Created `architecture/CONTEXT_ENGINEERING.md` (plugin pipeline, contracts, harness integration, gap register) and `plan/CONTEXT_ENGINEERING.md` (47 CE-EXT tasks in 6 waves). Added `ADR-CTX-001`, audit prompt domain, hub/AGENTS/audit-map/EXTENSION_AUTHOR updates, and MEMORY canon cross-links delegating Layer C to CE.

## Project impact

Platform now has a first-class domain for the context compiler engine — clear boundaries vs Memory/RAG/Tools, a roadmap for `ContextSourceProvider` catalog, step-aware assembly, codebase preset, and OBS spine completion.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/CONTEXT_ENGINEERING.md` |
| Plan | `docs/project/maintainers/plans/CONTEXT_ENGINEERING.md` — CE-DOC Done, CE-EXT Planned |
| ADR | `docs/project/technical/adr/entries/2026-06-12/ADR-CTX-001.md` |
| Audit | `docs/project/maintainers/audit/CONTEXT_ENGINEERING.md` · audit map §16 |

## Changed artifacts

- `docs/project/architecture/CONTEXT_ENGINEERING.md` — full engine canon
- `docs/project/maintainers/plans/CONTEXT_ENGINEERING.md` — implementation register
- `docs/project/technical/adr/entries/2026-06-12/ADR-CTX-001.md` — domain split decision
- `docs/project/architecture/MEMORY.md` — Layer C delegation
- `docs/project/architecture/intergrax_runtime_architecture.md` — 22nd pair + capability index
- `scripts/audit/generate_domain_audit_prompts.py` — MEMORY/CE split
- `AGENTS.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, `EXTENSION_AUTHOR_GUIDE.md`, `AGENT_CREATION_GUIDE.md` Appendix L

## Verification

```bash
uv run python scripts/audit/generate_domain_audit_prompts.py
uv run python scripts/audit/check_docs_domain_pairs.py
```

Result: pass (22 domain pairs).

## Risks and follow-ups

- CE-EXT implementation (CE-1..CE-12) not started — dual assembly paths remain until CE-3.
- `intergrax/context/` package does not exist yet — contracts are spec-only in canon.
