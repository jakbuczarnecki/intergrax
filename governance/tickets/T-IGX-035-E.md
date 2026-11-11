# T-IGX-035-E — MCP-SERVER-X P4: lifecycle, compatibility, app facades

## Scope

Batch **MCP-P4** only (MCP-P4-01..05): exposure disable semantics, retired bindings, `contract_version` + exposure schema fingerprint, eight application `mcp/server.py` facades on engine/definition pattern (research custom tool via binding), COMPAT mechanical inventory for `mcp-server-definition.v1`.

**Parent:** [`T-IGX-035-D.md`](T-IGX-035-D.md) (P3 publication @ `7172f294f` on `origin/development`).

**Baseline branch:** `origin/development` @ P4 cut (`7172f294f`).

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P4-01..05.

## Approved

_(awaiting operator approval — lifecycle remains `plan_pending` until `Approved:` line is recorded.)_

## Test contract

```text
.venv/Scripts/python.exe -m pytest \
  tests/unit/contracts/test_mcp_server_definition.py \
  tests/unit/applications/test_mcp_server_engine.py \
  tests/unit/applications/test_mcp_server_p4_lifecycle.py \
  tests/unit/applications/test_mcp_host_mount.py \
  tests/unit/applications/test_reference_apps_mcp.py -q
```

Expected: exit **0**.

## Out of scope

P5–P8 qualification/IIAC/FRZ PASS; Resources/Prompts; stdio/SSE transport; stage MCP-SERVER-X **CLOSED**.

## Lifecycle state

plan_pending

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `mcp-server-x-p4` |
| P3_PUBLICATION_SHA | `7172f294f` |
| Tasks | MCP-P4-01..05 targeted |
| ACT-N11 | No documented product-path allowlist amend for `mcp_server_definition.py` / `mcp_server_engine.py` — expect governance-fast product Δ; operator merge per prior MCP batches |
| Gate | **P4 IMPLEMENTED** (scope) — P5 next |
