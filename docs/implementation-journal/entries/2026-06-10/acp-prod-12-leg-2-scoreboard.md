---
id: IJ-2026-06-10-018
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS production readiness and fleet closure
plan_ref:
  - ACP-PROD-12
  - ACP-LEG-2
status: completed
commit: pending
adr: none — implements architecture §40.15 scoreboard canon already accepted
---

# ACP-PROD-12 scoreboard and ACP-LEG-2 fleet migration closure

## Operator request

Execute the next sprint: production readiness scoreboard and formal Wave 8 fleet migration closure.

## Summary

Added typed `AgentProductionReadinessReport` with ten scored dimensions, roster aggregation, and markdown export. Implemented `report_agent_production_readiness.py` and `check_agent_production_readiness.py` with `--require-fleet-migration-closure` for ACP-LEG-2 (Runtime 100% roster-wide, legacy_count 0, typed-state allowlist empty). Generated `build/agent_production_readiness.json` for all 16 roster agents.

## Project impact

Operators have a single artifact for roster promotion decisions. Fleet migration program (DEBT-ACP-16) is machine-verifiable: all agents score Runtime 100% with `fleet_migration_complete=true`.

## Traceability

- Plan: §6.1az · steps 7.9 · 8.8
- Architecture: §40.15

## Changed artifacts

- `intergrax/contracts/agent_readiness.py`
- `intergrax/agents/readiness/scoreboard.py`
- `scripts/gates/report_agent_production_readiness.py`
- `scripts/gates/check_agent_production_readiness.py`
- `tests/unit/agents/readiness/test_production_readiness_scoreboard.py`
- `agents/README.md`, plan register

## Verification

- `uv run python scripts/gates/report_agent_production_readiness.py --roster`
- `uv run python scripts/gates/check_agent_production_readiness.py --require-fleet-migration-closure --regenerate`
- `uv run pytest tests/unit/agents/readiness/ -q`

## Risks and follow-ups

- Policy, checkpointing, idempotency dimensions score partial until Wave 6–7 ACP-PROD/ACP-ORG items land.
- `production_eligible_recommendation` remains false for most agents until overall ≥90% and per-dimension floors met.
