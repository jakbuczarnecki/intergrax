# ADR-TOOL-001: Catalog tool dispatch and full-gateway routing

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/TOOLS.md`](../architecture/TOOLS.md) · [`plan/TOOLS.md`](../plan/TOOLS.md) TOOL-ENG-1, TOOL-ENG-2 |

## Context

Phase O unified `tool_ids` on plans, but `ToolRuntime.invoke` only executed pipeline shims (`rag.retrieve`, `websearch.query`) and the LLM `run_bounded_tool_loop` / `ctx.invoke_tool` flag (`use_tools`). Arbitrary catalog ids (e.g. `jira.search_tasks`) were stored on the plan but never invoked. The §42.12 gateway rejected any `ToolRequest` not in a fixed capability alias set with `unknown_capability_tool`.

Agents and engine plans need two equivalent paths to catalog tools:

1. **Plan dispatch** — `ToolInvocationPlan.tool_ids` → `RuntimeToolInvoker` per id.
2. **Gateway dispatch** — `ToolRequest(tool_name=<registered tool_id>)` → same invoker.

## Decision

1. Introduce `catalog_dispatch.py` as the single Tier-1 module for direct catalog invocation (input coercion, trace, budget ticks).
2. **`ToolRuntime.invoke`** dispatches non-shim `tool_ids` through `RuntimeToolInvoker` after RAG/websearch steps and before/alongside `run_bounded_tool_loop` / `ctx.invoke_tool` when `use_tools=True`.
3. **`RuntimeToolGateway`** routes any registered `tool_id` (not in the capability alias set) to the invoker when `runtime_state` is bound; runtime-bound and sandbox tools keep existing paths in `BoundToolGateway`.
4. **`ToolInvocationPlan`** gains optional `tool_inputs: Mapping[str, Mapping[str, Any]]` for per-id payloads (also accepted on `nexus.capability_plan` payloads).
5. Input coercion uses `contract.input_schema.model_validate(dict(raw or {}))` — same as runtime-bound tools.

**Rejected:** Re-planning catalog ids through `run_bounded_tool_loop` / `ctx.invoke_tool` only (loses deterministic plan semantics). Duplicating invoke logic inside the gateway (violates single invoker enforcement).

## Consequences

### Positive

- Engine plans and UAEP `invoke_tool` share one invoker path with scope policy and idempotency.
- Explicit `tool_ids` execute without `use_tools=True`.
- Gateway closes the 172+ tool gap without new Tier-0 mechanisms.

### Negative

- Plans without `tool_inputs` rely on schema defaults; invalid required fields fail at invoke with validation errors.
- `use_tools=True` plus explicit `tool_ids` may run both explicit dispatch and LLM planner until TOOL-ENG-4 constrains the planner.

## Compliance

- Tier boundaries preserved — dispatch composes existing `RuntimeToolInvoker`; no agent-specific Nexus branches.
- `ToolAccessPolicy` and `ToolScopePolicy` apply on both paths.
- Architecture and plan TOOLS pair updated.

## Implementation notes

- `intergrax/runtime/nexus/tools/catalog_dispatch.py`
- `tool_runtime.py`, `tool_gateway.py`, `tool_access_policy.py`
- `IdempotentToolInvoker.registry` property for protocol parity
- Tests: `test_tool_runtime_catalog_dispatch.py`, extend `test_tool_gateway.py`
- Verify: `uv run pytest tests/unit/runtime/nexus/tools/ tests/unit/runtime/tools/ -m gate -q`
