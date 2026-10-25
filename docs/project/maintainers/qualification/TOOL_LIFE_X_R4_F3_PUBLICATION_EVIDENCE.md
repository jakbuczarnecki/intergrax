# TOOL-LIFE-X R4 — F3 scoped publication evidence

| Field | Value |
| --- | --- |
| **START_HEAD** | `be09f5a89027f701bbcd8ace0e7185ec73f34b0e` |
| **Scope** | Cross-loop catalog child ingress (qualification configuration); **Option C** remains production default |
| **Physical L3 (CANCEL-C HTTP)** | **OPEN** — not claimed in this publication |
| **B8 / parent CLOSED** | **Not claimed** |

## Prepublication Enterprise Quality Gate

| Check | Result |
| --- | --- |
| Child semantic owner | `ChildExecutionRunner` via `CatalogToolChildExecutionWorkPort` (unchanged) |
| No second root supervisor | HTTP path uses ingress only; no `admit_protected_mcp_root_execution_supervision` on catalog |
| Parent context on drain | `ExecutionBoundary` identity bind + `peek_active_execution_*` on drain |
| Queue bound | max 32 (default); L2 queue-full |
| Option C default | `enable_cross_loop_catalog_child=False`; L2 `test_option_c_default_denies_cross_loop_without_qual_flag` |
| Shutdown / pending cleanup | `refuse_new_submissions`; L2 shutdown test |
| MSE / tenant | Reuse bridge + child qual suite |

## L2 (prepublication)

Command: `uv run pytest tests/unit/applications/test_mcp_catalog_cross_loop_ingress_qualification.py tests/unit/applications/test_mcp_catalog_child_execution_qualification.py tests/unit/applications/test_mcp_catalog_gate_b_cancel_c_qualification.py`

Log: `.tmp/session/TOOL-LIFE-X-R4/pytest-f3-prepublication-l2.log` — **36 passed** @ publication commit SHA (record after push).

## Operator decisions (this wave)

| OD | Status |
| --- | --- |
| F3 scoped publication | Approved subject to EQG |
| OD-EC7 DEADLINE_ONLY | Approved — see `TOOL_LIFE_X_R4_OD_EC7_RECORD.md` |
| OD-TL8-IMPL | Conditional — after Gate A L3 |
| G-A07 | Proposal only — no prod activation |
