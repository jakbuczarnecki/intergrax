# TOOL-LIFE-X R3.1 — Authorized execution wave evidence

| Field | Value |
| --- | --- |
| **Wave** | `TOOL-LIFE-X R3.1 AUTHORIZED WAVE` |
| **START_HEAD** | `df2fcad2df23b996c1e8ca59089be355226e90f1` |
| **Branch** | `development` |
| **Operator scope** | OD-CHILD-ARCH/IMPL, OD-PG-DOCKER, OD-GATEA-SCOPE/IMPL (bounded), F0, F4-01, F3 read-only |
| **Excluded** | OD-CROSSLOOP, OD-TL8-IMPL, production MCP catalog HTTP, formal CLOSED |

## F0 baseline

- Repository @ `df2fcad2df23b996c1e8ca59089be355226e90f1`, branch `development`, clean tree before wave edits.
- SAFE SCOPE prior evidence: [`TOOL_LIFE_X_SAFE_SCOPE_EXECUTION_EVIDENCE.md`](TOOL_LIFE_X_SAFE_SCOPE_EXECUTION_EVIDENCE.md) — not duplicated.
- Child semantic owner: `ChildExecutionRunner` @ `intergrax/runtime/execution/child.py` (sole `child_execution_id` mint).

## F1 — in-process catalog child (OD-CHILD-IMPL)

- **Work port:** `intergrax/runtime/execution/catalog_tool_child_work_port.py`
- **MSE hook:** `intergrax/applications/_shared/catalog_tool_child_mse_admission_hook.py` (child-bound `execution_id`)
- **Bridge:** `mcp_catalog_invocation_bridge.py` — no second root OWN-A2 path; HTTP MCP returns `execution_admission_denied`
- **L2:** `tests/unit/applications/test_mcp_catalog_child_execution_qualification.py` (3), gate B regressions updated (27 passed consolidated catalog suite)

## F2 — PostgreSQL L3

- **L3:** `tests/unit/tools/providers/test_tool_life_x_l3_physical_postgresql_qualification.py` — **124 passed** with `tests/unit/runtime/external_operations` in same consolidated run @ implementation candidate (see publication SHA).

## F3 — cross-loop feasibility (read-only)

**Disposition:** `F3 G1 — NOT PROVEN / OPTION C CONTINUES`

- Supervised root `asyncio.Task` exists (`HostRootExecutionTransportDetachmentSupervisor`) but MCP catalog no longer admits a second root task.
- `ContextVar` execution context does not transfer across MCP HTTP handler tasks; legal ingress requires a future `RootExecutionChildWorkIngressPort` (or equivalent) **inside** the supervised root coroutine — **OD-CROSSLOOP-ARCH not granted**.

## F4-01 — Tool deadline closed-world audit

- Production tool invoke path: `RuntimeToolInvoker` + `ToolRuntime` deadline hierarchy (BATCH-A evidence lineage).
- MCP catalog child path inherits parent deadline projection via `ChildExecutionRunner` (`narrow_child_deadline_at_utc`).
- **EC7:** not marked PASS — recommend **OD-EC7** when operator accepts DEADLINE_ONLY rollup.

## F5 — Gate A (bounded)

- Authorization fencing: existing `ExternalOperationExecutionGate` + store-native certified owner-loss (R6–R10 @ HEAD).
- **Physical effect-sink L3 per DB/HTTP/MCP:** **VALIDATION REQUIRED** — no new public owner ratified this wave; pre-send ≠ atomic remote fence unchanged.
- Consolidated L2: `tests/unit/runtime/external_operations` included in wave run — green.

## F6

- **BLOCKED** — OD-TL8-IMPL not granted; scenario 24 not claimed.

## Disposition

`TOOL-LIFE-X R3.1 AUTHORIZED WAVE — PARTIAL / BLOCKED` (F3 cross-loop, full Gate A physical L3, F6, parent CLOSED remain open).

| **Publication SHA** | `b8bd51556bfce53494fc22318e356549f341e32c` |
| **Implementation ancestry** | contains `df2fcad2df23b996c1e8ca59089be355226e90f1` |
