---
id: IJ-2026-06-11-013
date: 2026-06-11
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-LEG-4
status: completed
commit: pending
adr: none — documentation alignment with ACP-CLOSE-LEG-1/2/3; no new runtime contract
---

# ACP-CLOSE LEG-4 — author guide ACP canon (no UAEP-first path)

## Operator request

Continue ACP-CLOSE sprint queue: close LEG-4 — §45 checklist and `AGENT_CREATION_GUIDE.md` must treat UAEP as harness-internal only.

## Summary

Rewrote author-facing sections of `AGENT_CREATION_GUIDE.md` so the canonical path is `on_next_step` + typed `AcpSessionState` (scaffold `--pattern`). Removed UAEP-first / bridge wording, updated pre-merge checklist, HITL/shadow/memory/orchestration appendices, and anti-patterns. Architecture §45 items 11–13 now ask ACP control-loop questions instead of UAEP author API questions. Added `check_agent_creation_guide_acp_canon.py` grep gate wired into `check_agent_acp_close_ci.py`.

## Project impact

New agent authors see a single ACP entry path in canon docs; UAEP remains documented only as an internal framework bridge. CI prevents regression to UAEP-first author messaging.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §45 |
| Plan | `ACP-CLOSE-LEG-4` |
| ADR | none |

## Changed artifacts

- `docs/guides/AGENT_CREATION_GUIDE.md` — ACP-canonical author path
- `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — §45 checklist items 11–13
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — LEG-4 **Done**
- `scripts/maintenance/check_agent_creation_guide_acp_canon.py` — grep gate (new)
- `scripts/gates/check_agent_acp_close_ci.py` — aggregate includes LEG-4 gate
- `tests/unit/scripts/test_check_agent_creation_guide_acp_canon.py` — gate self-test (new)

## Verification

```bash
uv run python scripts/maintenance/check_agent_creation_guide_acp_canon.py
uv run pytest tests/unit/scripts/test_check_agent_creation_guide_acp_canon.py -m gate -q
python scripts/audit/check_docs_domain_pairs.py
```

## Risks and follow-ups

- PAT-1 + TOOL-ENG-6 (ReAct unified tool loop) and CI-2 remain open.
- PAT-3 terminology consolidation still pending.
