---
id: IJ-2026-06-11-006
date: 2026-06-11
tiers:
  - tier-0
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-PROD-7
  - ACP-CLOSE-PROD-8
status: completed
commit: pending
adr: none — scoreboard threshold update reflects delivered acceptance evidence
---

# ACP-CLOSE PROD-7/8 — §40.12 reference checklist + mutating scoreboard 100%

## Operator request

Execute next ACP-CLOSE sprint: §40.12 production readiness checklist for reference mutating profile and scoreboard 100% on checkpointing/idempotency dimensions.

## Summary

Added `section_40_12_checklist.py` and `check_acp_section_40_12_checklist.py` emitting `build/acp_section_40_12_reference.json` for harness capability `harness.acp.declarative_mutating`. Updated scoreboard mutating dimensions to **100%** with acceptance test evidence (05c/05d/05e) and no blockers. Extended `check_agent_production_readiness.py` with `--require-mutating-checkpoint-idempotency-100`.

## Project impact

Reference mutating acceptance profile now has a typed §40.12 artifact. Roster mutating agents score 100% on checkpointing and idempotency — closes PROD-8 gate for promotion decisions on those dimensions.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.12 · §40.15 |
| Plan | `ACP-CLOSE-PROD-7` · `ACP-CLOSE-PROD-8` |
| ADR | none |

## Changed artifacts

- `intergrax/agents/readiness/section_40_12_checklist.py` — reference checklist builder (new)
- `intergrax/agents/readiness/scoreboard.py` — mutating dimensions 100%
- `scripts/maintenance/check_acp_section_40_12_checklist.py` — CI gate (new)
- `scripts/gates/check_agent_production_readiness.py` — PROD-8 flag
- `tests/unit/agents/readiness/test_section_40_12_checklist.py` — unit test (new)
- `tests/unit/agents/readiness/test_production_readiness_scoreboard.py` — mutating dimension tests
- `build/acp_section_40_12_reference.json` — generated artifact
- `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — PROD-7/8 Done

## Verification

```bash
uv run pytest tests/unit/agents/readiness/ -q
uv run python scripts/maintenance/check_acp_section_40_12_checklist.py --write
uv run python scripts/gates/check_agent_production_readiness.py --regenerate --require-mutating-checkpoint-idempotency-100
```

Result: 6 passed; both scripts OK.

## Risks and follow-ups

- Policy/security scoreboard blockers (STRICT per-agent) remain for full production roster promotion.
- ACP-CLOSE-PROD-5/6 (compensation queue, idempotency store cross-run depth) still open.
- ACP-CLOSE-CI-3 (`--fail-on-blockers` in gate workflow) not wired this sprint.
