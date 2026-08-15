---
id: IJ-2026-06-10-027
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - DEBT-ACP-05
  - ACP-8
status: completed
commit: pending
adr: none — scaffold default flip; UAEP path retained behind --uaep
---

# DEBT-ACP-05 default typed agent scaffold

## Operator request

Continue sequential agent-architecture sprints; implement legacy scaffold cleanup (B1) after Wave 5 acceptance closure.

## Summary

`intergrax.scaffold new-agent` now defaults to typed **reflex** cognitive pattern (`create_acp_pattern_agent`) with no `get_steps` / `run_step` boilerplate. Legacy UAEP tree (`steps/pipeline.py`) moved to `create_uaep_agent` behind `--uaep` (or `--reference`). CLI help and doc templates updated for typed layout. `check_scaffold_acp_pattern.py` validates default + explicit `--pattern react`.

## Project impact

New agents start on the ACP author surface (`perceive` / `reason` / `act` / `evaluate`) without opt-in `--pattern`; UAEP remains available for explicit legacy only.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §32.0 · §32.0.5 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` DEBT-ACP-05 |

## Changed artifacts

- `intergrax/scaffold/new_agent.py` — `create_uaep_agent`, default reflex in `create_agent`
- `intergrax/scaffold/cli.py` — `--uaep` flag, updated help/messages
- `intergrax/scaffold/doc_templates.py` — pattern-aware ARCHITECTURE/PLAN
- `scripts/maintenance/check_scaffold_acp_pattern.py` — default scaffold smoke
- `tests/acceptance/agent_os/test_scaffold.py` — typed default + `--uaep` legacy
- `tests/unit/scaffold/test_acp_pattern_scaffold.py` — default reflex test
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — DEBT-ACP-05 closed

## Verification

```bash
uv run pytest tests/unit/scaffold/test_acp_pattern_scaffold.py tests/acceptance/agent_os/test_scaffold.py tests/unit/scaffold/test_scaffold_doc_templates.py -q
uv run python scripts/maintenance/check_scaffold_acp_pattern.py
```

Result: pass.

## Risks and follow-ups

- `IntergraxAgent.decide_after_step` still uses deprecated `complete()` — separate cleanup.
- PROD hardening sprints (compensation, full CI matrix) remain open.
