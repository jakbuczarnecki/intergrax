---
id: IJ-2026-06-10-017
date: 2026-06-10
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS fleet migration closure
plan_ref:
  - ACP-MIG-5
  - ACP-MIG-7
status: completed
commit: pending
adr: none — Reflex migration of T4 agents; org_worker retains custom decide_after_step for HITL
---

# ACP Wave 8 MIG-5/7 — remaining roster and host binding verification

## Operator request

Execute the next sprint: complete fleet migration for remaining T4 agents and verify Tier-3 host bindings.

## Summary

Migrated `organization_worker` (HITL-aware Reflex + custom `decide_after_step`), `intergrax_assistant`, `problem_radar`, and `vendor_discovery` to typed ACP. Updated contracts with `cognitive_pattern=REFLEX`. Fleet inventory now marks all 16 roster agent modules as migrated. Added MIG-7 host binding smoke tests for lab, legal, research, LKW, and DSW registries.

## Project impact

Full Tier-2 roster (excluding lab mock fixtures) runs typed cognitive hooks with UAEP shim compatibility. Host manifests materialize migrated agents with valid pattern metadata.

## Traceability

- Plan: `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1aw steps 8.5, 8.7
- Architecture: `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.15

## Changed artifacts

- `agents/organization_worker`, `intergrax_assistant`, `problem_radar`, `vendor_discovery`
- Contract files + `scripts/audit/audit_agent_fleet_legacy.py` migrated set
- `tests/unit/agents/fleet/test_remaining_migration.py`
- `tests/unit/agents/fleet/test_host_binding_mig7.py`

## Verification

- `uv run python scripts/audit/audit_agent_fleet_legacy.py`
- `uv run python scripts/maintenance/check_agent_fleet_migration.py`
- `uv run pytest tests/unit/agents/fleet/ -q`

## Risks and follow-ups

- ACP-LEG-2 fleet closure and ACP-PROD-12 scoreboard remain open.
- Lab `mock_agents` intentionally outside roster inventory.
