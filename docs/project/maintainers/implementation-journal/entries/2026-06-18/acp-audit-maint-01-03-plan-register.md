---
id: IJ-2026-06-18-003
date: 2026-06-18
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-MAINT-01
  - ACP-MAINT-02
  - ACP-MAINT-03
status: completed
commit: 533589e3
adr: none — audit maintenance register only; no contract change
---

# ACP-MAINT-01..03 — Interactive layer 6 audit plan registration

## Operator request

Interactive layer-by-layer harness audit (Mode A2): confirm ACP verdict, register all maintenance proposals in domain plan, commit, advance to LLM_ADAPTERS.

## Summary

Layer 6 revalidation confirmed L3+ maturity (GAP-ACP 37/37 Closed, fleet 17/17). Identified AS-3 violation on `boundary_demo` author-time `allowed_tools` while ACP close CI passes. Registered three P2/P3 maintenance rows in `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1av and updated audit result artifact.

## Project impact

Fleet hygiene backlog is now traceable: boundary_demo ReflexAgent migration, AS-3 bundled into ACP close CI, audit prompt AUDIT-IDEAL sync.

## Traceability

| Link | Target |
|------|--------|
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1av |
| Audit result | `docs/audit_results/2026-06-18/AGENT_CONTRACTS_AND_ASSEMBLY.md` |
| Gate evidence | `check_agent_skill_resolution.py` FAIL on `boundary_demo` |

## Verification

Doc-only iteration; gates re-run during audit (ACP close OK, AS-3 FAIL expected until ACP-MAINT-01).
