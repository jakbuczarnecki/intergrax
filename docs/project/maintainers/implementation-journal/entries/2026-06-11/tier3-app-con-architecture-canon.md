---
id: IJ-2026-06-11-018
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - H-APP-CON-DOC.1
  - H-APP-CON-DOC.2
status: completed
commit: pending
adr: none — documentation-only canon; ADR-APP-001 deferred to APP-CON-1 code
---

# TIER3 APP-CON architecture canon §24–§45

## Operator request

Expand Tier-3 application environment architecture to full APP-CON canon symmetric with ACP — contracts, facades, hooks, observability, org envelope, checklists — as the single application-layer architecture document.

## Summary

Added APP-CON sections §24–§45 to `architecture/TIER3_APPLICATION_ENVIRONMENT.md`: `ApplicationManifest` contract, `run_task()` / `HarnessApplication` / `ApplicationHost` interface, `ApplicationRunSummary`, roster assembly, APP invariants, three control modes, dual observability, per-agent binding, use-case catalog, execution stack L4, organizational policy canonical home, APP-PROD gates, and implementation checklist. Updated hub with "Application in the harness environment", plan phase H-APP-CON, and ACP §39.8 cross-ref to TIER3 §39.

## Project impact

Application authors now have a decision-complete architecture pair aligned with ACP — same mental model (contracts, facade, hooks, prod gates) without duplicating Nexus or creating a 22nd domain. Documents open code gap APP-CON-1 (`ApplicationHost` pipeline mount).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §24–§45 |
| Plan | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` Phase H-APP-CON |
| Hub | `docs/project/architecture/intergrax_runtime_architecture.md` Application section |
| Agent cross-ref | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §39.8 |

## Changed artifacts

- `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — APP-CON canon
- `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — H-APP-CON phase
- `docs/project/architecture/intergrax_runtime_architecture.md` — hub Application section
- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — §39.8 canonical home pointer

## Verification

```bash
python scripts/docs/check_docs_domain_pairs.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (expected).

## Risks and follow-ups

- APP-CON-1: mount `ApplicationHost` in `build_harness_host_runtime` — planned H-APP-CON.
- APP-CON-DX.1: author guide Appendix APP — planned.
