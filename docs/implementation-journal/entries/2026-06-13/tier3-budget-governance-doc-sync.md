---
id: IJ-2026-06-13-007
date: 2026-06-13
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-PROD-7
  - ACP-TOK-2
  - ACP-TOK-3
status: completed
commit: 0de17aff
adr: none — documentation sync only; ACP-TOK-2/3 runtime delivered in ACP plan (2026-06-11)
---

# Tier-3 budget governance — documentation sync with ACP-TOK completion

## Operator request

Close the Tier-3 layer completion audit gap where architecture still marked kernel enforcement and host notify/hook as planned for ACP-TOK-2/3 despite runtime delivery in the ACP plan.

## Summary

Synchronized `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §34.3, §43, §44, §46, §47 with **Done** status for ACP-TOK-1..3 and ACP-TOK-CI. Updated fidelity matrix row §46 and execution-order note in `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md`. Hub line in `docs/intergrax_runtime_architecture.md` now states budget reactions **Done**.

## Project impact

Tier-3 budget governance canon matches harness runtime: mutating STRICT hosts declare `budget_reaction` + `budget_slice` (APP-PROD-7); kernel pre-LLM enforcement and host reaction paths are no longer documented as open cross-plan gaps.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §34.3 · §43 · §44 · §46 · §47 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` §46 fidelity · §6.2y step 3 |
| ADR | none — documentation sync only |
| Cross-plan | ACP-TOK-2 · ACP-TOK-3 delivered in `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` |

## Changed artifacts

- `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — budget governance status aligned to **Done**
- `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — §46 fidelity row + execution order step 3
- `docs/intergrax_runtime_architecture.md` — Tier-3 hub implementation blurb
- `docs/implementation-journal/entries/2026-06-13/tier3-budget-governance-doc-sync.md` — this entry

## Verification

```bash
uv run pytest tests/unit/agents/test_acp_token_budget_enforcement.py \
  tests/unit/agents/test_acp_token_budget_reactions.py \
  tests/unit/agents/test_acp_token_usage_metering.py -q
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
```

Result: 13 passed (ACP-TOK tests); doc pair check OK; journal check OK after section fix.

## Risks and follow-ups

- Nexus `RunBudget` environment cap remains **Partial** (COST-1) — P2 backlog.
- Historical journal entries (2026-06-11) still describe ACP-TOK-2/3 as open; no runtime impact.
