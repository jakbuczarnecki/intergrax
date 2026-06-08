# ADR-FLOW-001: Declarative delegation (`DELEGATES_TO`) expansion semantics

| Field | Value |
|-------|-------|
| **Status** | Accepted · **implemented** (`FLOW-2`, `FLOW-14`) |
| **Date** | 2026-06-07 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`NEXUS_EXECUTION_FLOW_REFERENCE.md`](../NEXUS_EXECUTION_FLOW_REFERENCE.md) §13 · canon [§42.14.3](../architecture/UNIFIED_EXECUTION_RUNTIME.md#42143-graph-delegation-subagent-equivalent) · plan `FLOW-2` / `FLOW-GAP-02` |

## Context

Canon §42.14.3 describes **graph delegation** as the Harness subagent equivalent: an `ExecutionGraph` node with `DelegationSpec`, isolated memory, and traceable parent/child metadata.

Tier-3 authors can declare topology via `ApplicationGraphSpec`:

- `DEPENDS_ON` — sequential dependency between agents (separate plan steps / graph nodes)
- `DELEGATES_TO` — attaches `DelegationSpec` with `child_agent_id` on the **source** step

**Runtime truth (post FLOW-2/14):** `graph_spec_to_plan.py` expands `DELEGATES_TO` into a child `PlanStep` with `DelegationSpec` on the **child** node; `GraphExecutor` routes execution to `child_agent_id` when delegation is present. `FLOW-GAP-02` is **closed**.

## Decision

Adopt **Option C — planning sugar with graph expansion**:

1. **`DELEGATES_TO` in `ApplicationGraphSpec`** remains the **author-facing** declarative edge (ergonomic subagent declaration).
2. **`graph_spec_to_plan` or `plan_to_execution_graph`** MUST **expand** each `DELEGATES_TO` into:
   - a **child `PlanStep` / `ExecutionNode`** for `child_agent_id`
   - `depends_on` from child → parent (child runs after parent completes)
   - `DelegationSpec` on the **child node** (isolated memory, parent trace metadata)
3. **`AgentDecision.HANDOFF`** remains the **runtime-dynamic** path (unchanged) — `HandoffCoordinator` inserts nodes during execution.
4. **`DEPENDS_ON` alone** remains the explicit multi-agent sequential path without delegation isolation semantics.

**Not chosen:**

| Option | Why rejected |
|--------|--------------|
| **A** — `DELEGATES_TO` executes child inside parent node | Conflicts with subagent isolation; duplicates UAEP vs graph boundaries |
| **B** — deprecate `DELEGATES_TO`; only `DEPENDS_ON` | Loses ergonomic declarative subagent API; breaks existing graph builder DX |

## Consequences

### Positive

- Canon §42.14.3 and operational flow reference align on **outcome** (child agent runs as graph node)
- Authors keep fluent `AgentGraph.delegates_to(source, target)` API
- `max_delegation_depth` can be enforced on expanded graph (FLOW-3)

### Negative / until FLOW-2 ships

- **Current runtime** still behaves as documented in flow reference §13 (gap remains)
- Existing tests for `DELEGATES_TO` plan mapping may need expansion tests for child node creation

## Implementation notes (FLOW-2 acceptance)

- `application_graph_spec_to_nexus_plan()` or `plan_to_execution_graph()` emits child step when `DELEGATES_TO` present
- `GraphExecutor` routes child node to `child_agent_id` with `DelegationSpec` on child node
- Gate tests: `test_graph_spec_to_plan.py`, integration graph executor delegation path
- Update canon §42.14.3 implementation paragraph when merged

## Compliance

- No nested Nexus / harness per child
- Child still passes `PolicyEngine`, `ToolRuntime`, UAEP stack
- Trace: `parent_run_id`, `parent_node_id` on child metadata (R-Delegate)
