---
id: IJ-2026-06-11-014
date: 2026-06-11
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-PAT-3
status: completed
commit: pending
adr: none — documentation consolidation; no runtime contract change
---

# ACP-CLOSE PAT-3 — single canonical §29 terminology entry

## Operator request

Continue ACP-CLOSE sprint queue: consolidate scattered author terminology into architecture §29 as the single canonical entry (GAP-ACP-07).

## Summary

Added architecture **§29.0** (single terminology index) and guide **§1 Author terminology canon** table linking to §29. Slimmed Appendix AC.1 to defer to §29 instead of duplicating definitions. Updated architecture §22–§23, §27 flows, and §36.4 alignment for ACP vocabulary (`on_next_step`, `StepOutcome`, `acp_run`). Extended `check_agent_creation_guide_acp_canon.py` with PAT-3 markers. Closed **GAP-ACP-07**.

## Project impact

Authors and LLM agents have one normative vocabulary entry (§29); appendices are examples only. Reduces UAEP/run_step mental-model drift across docs.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §29.0 · §22–§23 · §27 |
| Plan | `ACP-CLOSE-PAT-3` |
| ADR | none |

## Changed artifacts

- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — §29.0, planes, flows, GAP-ACP-07 Closed
- `docs/guides/AGENT_CREATION_GUIDE.md` — §1 terminology canon; AC.1/AC.2/AC.3 dedupe
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — PAT-3 **Done**
- `agents/README.md` — §29 terminology link
- `scripts/maintenance/check_agent_creation_guide_acp_canon.py` — PAT-3 gate markers

## Verification

```bash
uv run python scripts/maintenance/check_agent_creation_guide_acp_canon.py
uv run pytest tests/unit/scripts/test_check_agent_creation_guide_acp_canon.py -m gate -q
python scripts/audit/check_docs_domain_pairs.py
uv run python scripts/maintenance/check_implementation_journal.py
```

## Risks and follow-ups

- PAT-1 + TOOL-ENG-6 (ReAct unified tool loop) remains open — closes GAP-ACP-04 / DEBT-ACP-18.
- PAT-2 (CVL reflection hooks) and CI-2 still planned.
