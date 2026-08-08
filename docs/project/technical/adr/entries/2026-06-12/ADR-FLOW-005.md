# ADR-FLOW-005: Retire Tier-1 RuntimeEngine pipeline stack

| Field | Value |
|-------|-------|
| **Status** | Accepted · **implemented** (`ACP-CLOSE-LEG-5`) |
| **Date** | 2026-06-12 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13.5 · plan `ACP-CLOSE-LEG-5` |

## Context

After ACP fleet migration (Wave 8), no Tier-2 agent used `RuntimeEngine`, `NoPlannerPipeline`, or `steps/pipeline.py`. The Tier-1 pipeline stack (`pipelines`, `runtime_steps`, `EnginePlanner`, `PlanLoopController`) remained as dead code and confused audits (engine planner vs Nexus task planner).

Agent execution is exclusively:

- **ACP** — `IntergraxAgent.run` / `on_next_step` + `HarnessKernel`
- **UAEP shim** — `get_steps` / `run_step` on legacy stubs only (gate tests), not production agents

## Decision

1. **Delete** `intergrax/runtime/nexus/pipelines`, `runtime_steps`, `engine/runtime.py` (`RuntimeEngine`), engine planner stack (`engine_planner*`, `plan_loop_controller`, `step_planner`), and `uaep_pipeline_bridge.py`.
2. **Preserve** reusable tool/context helpers under `nexus/tools` (`tool_loop.py`, `plan_context_invocation.py`) and `nexus/context/tool_context_helpers.py`.
3. **Remove** `RuntimeConfig.pipeline`, `plan_loop_policy`, and related planner config fields — policy bundle remains the source for plan-loop policy when needed.
4. **Scaffold** — ACP-only; `--uaep` and `steps/pipeline.py` generation removed.
5. **Nexus multi-agent planning** (`task_planner`, `nexus_llm_plan_builder`) is unchanged — distinct from the retired per-run pipeline.

## Consequences

### Positive

- Single agent execution model; reduced Tier-1 surface and audit noise.
- Tool loop and RAG/websearch context invocation remain shared Tier-1 utilities for `ToolRuntime`.

### Negative

- External forks importing `RuntimeEngine` or `runtime_steps` break; migration path is ACP patterns + `ToolRuntime`.
- Documentation and audit prompts updated in the same release (ACP-CLOSE-LEG-5 doc scrub).

## Compliance

- `uv run pytest -m gate` green after retirement.
- Fleet gate: zero Tier-2 `RuntimeEngine` references (`check_agent_fleet_migration.py`).
