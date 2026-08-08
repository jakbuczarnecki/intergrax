---
id: IJ-2026-06-13-008
date: 2026-06-13
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-FINISH-DOC-1
  - ACP-TOK-2
  - ACP-TOK-3
status: completed
commit: 32b9e854
adr: none — documentation sync; runtime delivered 2026-06-11 via ACP-TOK-*
---

# ACP-FINISH-DOC-1 — close GAP-ACP-36/37 and sync architecture canon

## Operator request

Close the cross-plan documentation gap: ACP architecture still listed GAP-ACP-36/37 as Open and ACP-TOK-2/3 as Planned after runtime delivery and Tier-3 doc sync.

## Summary

Updated `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §25.4–§25.5, §28.3, §36.4, §40.13 to **Closed/Done** for token budget depth. Marked **ACP-FINISH-DOC-1 Done** in plan register §6.1bc. Regenerated domain audit prompt via `generate_domain_audit_prompts.py`.

## Project impact

Agent architecture canon (§13–§40) is now honestly **implementation-complete** for token metering, limits, and reactions. GAP register reads **37 Closed · 0 Open**. Cross-plan Tier-3 budget governance and ACP docs are aligned.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §25.4 · §25.5 · §28.3 · §36.4 · §40.13 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-FINISH · §6.1bc |
| Cross-plan | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §43 |
| Audit prompt | `docs/project/maintainers/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` |

## Changed artifacts

- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — GAP-ACP-36/37 Closed; §40.13 maturity sync
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-FINISH phase Done; executive summary
- `scripts/audit/generate_domain_audit_prompts.py` — AGENT known_gaps + active_phases
- `docs/project/maintainers/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` — regenerated

## Verification

```bash
uv run pytest tests/unit/agents/test_acp_token_budget_enforcement.py \
  tests/unit/agents/test_acp_token_budget_reactions.py \
  tests/unit/agents/test_acp_token_usage_metering.py -q
python scripts/audit/check_docs_domain_pairs.py
python scripts/maintenance/check_implementation_journal.py
uv run python scripts/audit/generate_domain_audit_prompts.py
```

Result: 13 passed (ACP-TOK tests); doc pair OK; journal OK after entry added.

## Risks and follow-ups

- Nexus `RunBudget` graph env cap remains **Partial** (COST-1) — P2, does not reopen GAP-ACP-36/37.
- AUDIT-IDEAL-19.1 · 20.1 · 31.1 remain parallel Planned items in §12–§20 track.
