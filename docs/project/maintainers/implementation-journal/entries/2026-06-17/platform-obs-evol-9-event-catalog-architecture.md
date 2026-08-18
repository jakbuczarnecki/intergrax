---
id: IJ-2026-06-17-006
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9-DOC
  - P1-ARCH-02
status: completed
commit: 11b8e07d
adr: ADR-OBS-003
---

# OBS-EVOL-9 — Layered runtime event catalog architecture (P1-ARCH-02)

## Operator request

Accept layered event identity (spine + `event_kind` + `EventCatalog`) before external publication; update all documentation and implementation steps.

## Summary

Documented scalable HOS event model: frozen spine `RuntimeEventType`, unbounded `event_kind` for Tier-2/3, derived `EventCategory`, `DOMAIN_SIGNAL` carrier, pre-release spine consolidation plan (74→~50), and OBS-EVOL-9 implementation register M0–M3.

## Project impact

External developers will extend observability via `emit_domain_signal` and payload registry — not platform enum PRs. Ops/hooks subscribe by category/kind prefix.

## Traceability

| Link | Target |
|------|--------|
| ADR | `docs/project/technical/adr/entries/2026-06-17/ADR-OBS-003.md` |
| Architecture | `docs/project/architecture/OBSERVABILITY.md` §4.4 |
| Plan | `docs/project/maintainers/plans/OBSERVABILITY.md` OBS-EVOL-9 |
| UAEP | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` §42.1.6–42.1.7 |
| Author guides | `AGENT_CREATION_GUIDE.md` Appendix Q §Q.5 · `EXTENSION_AUTHOR_GUIDE.md` §11 · `APPLICATION_CREATION_GUIDE.md` §8 |
| Debt | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` Appendix B P1-ARCH-02 |

## Changed artifacts

- `docs/project/technical/adr/entries/2026-06-17/ADR-OBS-003.md`
- `docs/project/architecture/OBSERVABILITY.md`
- `docs/project/maintainers/plans/OBSERVABILITY.md`
- `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`
- `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` (cross-ref)
- `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` (P1-ARCH-02 row)
- `docs/project/technical/adr/README.md`
- `docs/project/technical/guides/AGENT_CREATION_GUIDE.md`
- `scripts/audit/generate_domain_audit_prompts.py`
- `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md` §11
- `docs/project/technical/guides/APPLICATION_CREATION_GUIDE.md` §8

## Verification

```bash
python scripts/docs/check_docs_domain_pairs.py
python scripts/maintenance/check_harness_adr.py
python scripts/audit/generate_domain_audit_prompts.py
python scripts/maintenance/check_implementation_journal.py
```

## Risks and follow-ups

- OBS-EVOL-9.1–9.7 code not started — publication must wait for spine consolidation or document interim 74-type spine.
- `phase_coverage.py` drift until `EventCatalog` replaces it in M1.
- Metrics cardinality policy for `event_kind` labels must ship with OBS-EVOL-9.5.
