---
id: IJ-2026-06-10-015
date: 2026-06-10
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS fleet migration pilot
plan_ref:
  - ACP-MIG-1
  - ACP-MIG-2
  - ACP-MIG-3
  - ACP-MIG-6
status: completed
commit: pending
adr: none — composition of Wave 4 UAEP bridge + Wave 5 Reflex pattern; no new runtime semantics
---

# ACP Wave 8 pilot fleet migration (echo, signoff_probe, research)

## Operator request

Execute the next ACP sprint: Wave 8 fleet migration program — inventory, migration tiers, pilot batch (three agents), and CI regression gate.

## Summary

Added fleet inventory auditor (`audit_agent_fleet_legacy.py`) and migration regression gate (`check_agent_fleet_migration.py`). Introduced `acp_uaep_shim` and UAEP `get_steps`/`run_step` on `CognitiveAgent` so migrated agents run `on_next_step` without author-side `RuntimeEngine`. Migrated `echo`, `signoff_probe`, and `research` to `ReflexAgent` with `cognitive_pattern=REFLEX`. Documented migration tiers in `agents/README.md` and updated the fleet tracker.

## Project impact

Three roster agents now use typed cognitive hooks end-to-end while remaining UAEP-compatible for Nexus and `AgentEngine`. Fleet program has machine-readable inventory and CI guard against legacy surface regression on migrated agents.

## Traceability

- Plan: `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1aw Wave 8, ACP-MIG-1/2/3/6
- Architecture: `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.15 fleet migration

## Changed artifacts

- `intergrax/agents/authoring/acp_uaep_shim.py`, `patterns/base.py`, `acp_run.py`
- `intergrax/contracts/acp_metadata_keys.py` (`AcpRunContextKey`)
- `agents/echo`, `agents/signoff_probe`, `agents/research`
- `scripts/audit/audit_agent_fleet_legacy.py`, `scripts/maintenance/check_agent_fleet_migration.py`
- `agents/README.md`, plan tracker, `tests/unit/agents/fleet/test_pilot_migration.py`

## Verification

- `uv run python scripts/audit/audit_agent_fleet_legacy.py`
- `uv run python scripts/maintenance/check_agent_fleet_migration.py`
- `uv run pytest tests/unit/agents/fleet/test_pilot_migration.py agents/signoff_probe/tests/ tests/integration/agents/test_agent_engine_uaep_echo.py -q`

## Risks and follow-ups

- ACP-LEG-2 and MIG-4/5/7 remain open for the rest of the roster.
- ACP-PROD-12 scoreboard not yet generated for Runtime % column in tracker.
- Host binding verification (MIG-7) deferred to post-batch pass.
