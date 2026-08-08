# ADR-FLOW-004: Graph spec seed guard via `trigger_capabilities`

| Field | Value |
|-------|-------|
| **Status** | Accepted (ORCH-CONFIG.2) |
| **Date** | 2026-06-09 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/ORCHESTRATION.md`](../../architecture/ORCHESTRATION.md) §56 · plan `ORCH-CONFIG.2` · CFG-18 |

## Context

`GraphSpecSeedingPlanner` previously seeded a `NexusPlan` from `ApplicationGraphSpec` for **every** task without a pre-built `plan_id`. On multi-route Tier-3 hosts (intake + pipeline on the same manifest), a single-agent intake task was incorrectly replaced by the full application graph (CFG-18).

Product teams need declarative control over **when** topology seeding applies without forking `TaskPlanner` per application.

## Decision

1. Add **`ApplicationGraphSpec.trigger_capabilities: list[str]`** — when non-empty, graph seeding applies **only** when `task.context.capability` matches one of the listed values.
2. When **`trigger_capabilities` is empty**, retain backward-compatible convention: seed only when capability ends with **`pipeline_capability_suffix`** (default `".pipeline"`).
3. **`should_seed_plan_from_graph_spec(task, spec)`** is the single guard; `GraphSpecSeedingPlanner` delegates to it before calling `application_graph_spec_to_nexus_plan`.
3. Orchestration capabilities (``trigger_capabilities`` / ``*.pipeline`` suffix) are **routing tokens** — agents are selected by ``graph_spec`` node ``agent_id``; registry lookup by orchestration token is not required.

**Not chosen:**

| Option | Why rejected |
|--------|--------------|
| Always seed when `graph_spec` present | Breaks single-route hosts (CFG-18) |
| Per-application `TaskPlanner` subclass | Duplicates Tier-0 orchestration; violates harness-first principle |

## Consequences

- Same host can serve intake (`dispute.intake`) and pipeline (`dispute.pipeline`) without graph override.
- Lab/echo hosts must use `*.pipeline` capability or explicit `trigger_capabilities` to opt into seeding.
- Tier-3 products declare trigger capabilities alongside `graph_spec` in `ApplicationEnvironmentProfile`.

## Compliance (ORCH-CONFIG.2 acceptance)

- Unit tests: `tests/unit/applications/test_graph_spec_to_plan.py`
- Integration: `tests/integration/runtime/test_orchestration_cfg_simulation.py`
- Canon updated: `architecture/ORCHESTRATION.md` §56.11
