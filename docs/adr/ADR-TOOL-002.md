# ADR-TOOL-002: Bounded multi-iteration tool loop (TOOL-ENG-6)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-11 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/TOOLS.md`](../architecture/TOOLS.md) · [`plan/TOOLS.md`](../plan/TOOLS.md) TOOL-ENG-6 · ACP-CLOSE-PAT-1 |

## Context

`ToolsStep` executed a single planner pass and injected tool output via a system prompt — not a native `role=tool` chain. ReAct agents (`ReActAgent`) maintained a separate cognitive loop with budget counters not mirrored in `acp.state.v1.budget` (DEBT-ACP-18 · GAP-ACP-04).

## Decision

1. Introduce `run_bounded_tool_loop` in `intergrax/runtime/nexus/runtime_steps/tool_loop_step.py` — shared plan→invoke→observe loop with `max_tool_iterations` (default **1** for backward compatibility).
2. Add `RuntimeConfig.max_tool_iterations`; `ToolsStep` delegates to the loop module.
3. Add `ToolPlanningService.plan_native_round` for one native LLM round returning typed `LLMToolCall` + `ToolCallPlan`.
4. When `max_tool_iterations > 1`, append `assistant` + `role=tool` messages to `messages_for_llm`; legacy single-pass keeps system-prompt injection.
5. Extend `AcpBudgetState` with `react_iterations_used` / `react_iterations_max`; `ReActAgent` mirrors counters via `react_budget.py` (ACP-CLOSE-PAT-1).

**Rejected:** Scheduling tool iterations from `GraphExecutor` (ACP-AP-02). Duplicating invoke logic outside `RuntimeToolInvoker`.

## Consequences

### Positive

- One loop implementation for pipeline and ReAct budget alignment.
- Native multi-turn tool context for ReAct-style runs when hosts set `max_tool_iterations > 1`.
- DEBT-ACP-18 closed; GAP-ACP-04 closed at pattern/budget layer.

### Negative

- JSON fallback planner path remains single-iteration only.
- Multi-iteration loops require native `supports_tools()` adapters.

## Compliance

- Tier boundaries preserved — loop composes `ToolPlanningService` + `RuntimeToolInvoker`.
- Tests: `tests/integration/runtime/test_tool_loop_integration.py`.

## Implementation notes

- `intergrax/runtime/nexus/runtime_steps/tool_loop_step.py`
- `intergrax/runtime/nexus/tools/tool_planning_service.py` — `plan_native_round`
- `intergrax/agents/authoring/patterns/react_budget.py`
- `intergrax/contracts/acp_state.py` — budget fields
