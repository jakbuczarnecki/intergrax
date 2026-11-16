# Lane L1 report (transcribed by the orchestrator from the lane agent's final message)

Branch `mcp-p7-l1` @ `d52e62ef06e176a5767363e7dca742f09f5fab6a`, base `origin/development` d5c1383fc. Worktree `D:/Projekty/_wt/intergrax/p7-l1`. Tests only, no product-path change, not pushed. Trailer `Ref: governance/tickets/T-IGX-035-H.md`.

File: `tests/qualification/mcp_server_x/test_mcp_server_x_p7_l1_contract.py` (L1, 31 functions, 41 cases).
- MCP-ACC-02: `test_acc_02_*` — error-code enum has 7 stable codes; matrix error codes defined in enum; enum codes not in matrix pinned as the known documentation gap; contract version pinned; incomplete/foreign definitions rejected (8 cases); duplicate/empty binding id rejected; frozen definition; blank required fields (3 cases); unsupported contract version; unknown allowlist binding / catalog tool report offending id; catalog binding without catalog surface fails; valid definition passes; resolve_binding error codes; validation error is ValueError with code; contract imports no fastmcp (AST + clean-subprocess `sys.modules` check).
- MCP-ACC-04 (L1 part): `test_acc_04_*` — disabled policy exposes nothing; empty allowlist fail-closed; binding off allowlist dropped and not invocable; hidden/retired never exposed; definition order preserved; allowlist entries stripped; default policy exposes visible active bindings; all binding kinds filtered by the same policy.
- MCP-ACC-16 (L1 part): `test_acc_16_*` — contract version constant pinned; fingerprint golden sha256; ignores non-exposed bindings/server identity; changes when a binding is retired; sensitive to shape/visibility/order; disabled == empty definition; COMPAT inventory lists definition version.

Command: `cd D:/Projekty/_wt/intergrax/p7-l1 && D:/Projekty/intergrax/.venv/Scripts/python.exe -m pytest tests/qualification/mcp_server_x/test_mcp_server_x_p7_l1_contract.py -q` under the shared pytest lock: exit 0, 41 passed, ~13 s (two runs, before and after ruff format). `ruff check` and `ruff format` clean; intergrax imported from the worktree.

## Findings
1. Documentation gap: the "Error codes" table in `MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md` lists 6 codes; the enum has 7 (missing `MCP_BINDING_UNKNOWN_TOOL`). The test pins the difference (`..._exactly_the_known_documentation_gap`); it must be changed to `== set()` when the matrix is corrected. `MCP_BINDING_UNKNOWN` is documented as "not in policy allowlist" but is also raised for a non-existent and for a defined-but-unexposed binding.
2. Declared but never raised: `MCP_PRINCIPAL_REQUIRED`, `MCP_TENANT_ASSERTION_CONFLICT`, `MCP_RESULT_BOUND_EXCEEDED` appear only in the enum and the matrix (grep over `intergrax/` and `tests/`; not in `mcp_server_engine.py` or `mcp_developer_api.py`). No pinning test added. L2/L3 lanes must not assume these codes appear on the wire; record as residual in the P7 matrix.
3. No failing tests; no xfail/skip.

## Not verified
L2/L3 layers; ACC-04 and ACC-16 are L1+L2 rows (only the L1 part is covered); the golden fingerprint is for the current tree, Python 3.12 and the pydantic version in `.venv`; no broader suites, `igx_gov` or clean-checkout replay.
