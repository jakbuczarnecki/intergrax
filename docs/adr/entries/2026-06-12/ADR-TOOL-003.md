# ADR-TOOL-003: ToolInvocationPattern protocol and orchestration plugin model (TOOL-ENG-16)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-12 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/TOOLS.md`](../../architecture/TOOLS.md) · [`plan/TOOLS.md`](../../plan/TOOLS.md) TOOL-ENG-16–23 · ADR-TOOL-002 |

## Context

Tier-1 tool orchestration (Plane 3, phase 2a) was hardcoded in `run_bounded_tool_loop`. Selection (`ToolSelectionStrategy`) and planning (`ToolPlannerProtocol`) already use plugin-style contracts, but multi-call execution could not be swapped per host or extended via entry points.

Invocation-pattern audit 2026-06-12 identified four production patterns (single, parallel batch, bounded ReAct, deterministic chain) plus graph boundary vs `ExecutionGraph` (ORCHESTRATION §50.4).

## Decision

1. Introduce `ToolInvocationPattern` Protocol with `pattern_id` and `execute(...) -> ToolInvocationResult` in `intergrax/runtime/nexus/tools/tool_invocation_pattern.py`.
2. Ship `SinglePassPattern` (TOOL-ENG-17) and `BoundedReactPattern` (TOOL-ENG-18) under `patterns/`; atomic invoke stays `RuntimeToolInvoker` (unchanged).
3. Add `ToolInvocationMode` enum and `pattern_for_mode()` factory (TOOL-ENG-21); `RuntimeConfig.tool_invocation_mode` optional — `None` infers bounded ReAct when `max_tool_iterations > 1` (ADR-TOOL-002 compat).
4. `run_bounded_tool_loop` delegates to resolved pattern (TOOL-ENG-22); `plan_context_invocation` passes config mode.
5. Bridge `ApplicationEnvironmentProfile.tool_invocation_mode` via `catalog_runtime_bridge.py` (TOOL-ENG-23).
6. Defer `ParallelBatchPattern`, `DeterministicChainPattern`, `ParallelSemanticBatchPattern` to TOOL-ENG-9/20/25 — factory raises `NotImplementedError` until shipped.

**Rejected:** Tool-level ReAct inside `GraphExecutor` (ADR-TOOL-002). Replacing `RuntimeToolInvoker` with pattern-specific invoke paths.

## Consequences

### Positive

- Hosts can select orchestration mode without forking Nexus.
- Clear boundary: graph orchestrates agents; patterns orchestrate tool batches within one step.
- Entry-point registry (TOOL-ENG-24) has a stable protocol target.

### Negative

- Interim duplication: pattern classes initially extract logic from former monolithic `tool_loop.py`.
- Unshipped modes fail at factory until follow-on PRs land.

## Compliance

- Tier boundaries preserved — patterns compose `ToolPlannerProtocol` + `RuntimeToolInvoker` only.
- Tests: `test_tool_invocation_pattern.py`, `test_tool_loop_integration.py`, bridge unit test.

## Implementation notes

- `intergrax/runtime/nexus/tools/tool_invocation_pattern.py`
- `intergrax/runtime/nexus/tools/patterns/`
- `intergrax/runtime/nexus/config_types.py` — `ToolInvocationMode`
- `intergrax/applications/_shared/catalog_runtime_bridge.py`
