---
id: IJ-2026-06-10-016
date: 2026-06-10
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS fleet migration product batch
plan_ref:
  - ACP-MIG-4
status: completed
commit: pending
adr: none — Reflex stub hooks preserve legacy answer shapes; no new runtime semantics
---

# ACP Wave 8 MIG-4 product fleet migration

## Operator request

Continue Wave 8 with the product batch: legal, summary, LKW trio, and DSW quartet migrated to typed ACP Reflex agents.

## Summary

Added `acp_stub_reflex` helpers for prefixed stub answers and pipeline runtime context wiring. Migrated nine agents (`legal`, `summary`, `local_search`, `local_indexer`, `local_synthesizer`, four `dispute_*`) from UAEP `run_pipeline_step` to `ReflexAgent` with `cognitive_pattern=REFLEX`. Updated contracts, fleet inventory migrated set, plan tracker, and product-batch tests.

## Project impact

Twelve of fifteen roster agents now run typed `on_next_step` while remaining UAEP-compatible via the Wave 8 shim. Product hosts (legal, research, LKW, DSW) keep stable answer prefixes and step ids.

## Traceability

- Plan: `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1aw step 8.4 (ACP-MIG-4)
- Architecture: `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §40.15

## Changed artifacts

- `intergrax/agents/authoring/acp_stub_reflex.py`
- `agents/legal`, `agents/research/summary_agent.py`, LKW trio, DSW quartet
- Contract files with `cognitive_pattern` + `pattern_version`
- `scripts/audit/audit_agent_fleet_legacy.py` migrated set
- `tests/unit/agents/fleet/test_product_migration.py`

## Verification

- `uv run python scripts/audit/audit_agent_fleet_legacy.py`
- `uv run python scripts/maintenance/check_agent_fleet_migration.py`
- `uv run pytest tests/unit/agents/fleet/ tests/integration/agents/test_agent_engine_uaep_legal.py tests/integration/agents/test_agent_engine_uaep_research.py -q`

## Risks and follow-ups

- MIG-5 (org_worker, assistant, K-path agents) and MIG-7 host binding verification remain.
- ACP-PROD-12 scoreboard not yet generated for Runtime % column.
