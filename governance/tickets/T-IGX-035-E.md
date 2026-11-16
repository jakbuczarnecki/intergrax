# T-IGX-035-E — MCP-SERVER-X P4: lifecycle, compatibility, app facades

## Scope

Batch **MCP-P4** only (MCP-P4-01..05): exposure disable semantics, retired bindings, `contract_version` + exposure schema fingerprint, eight application `mcp/server.py` facades on engine/definition pattern (research custom tool via binding), COMPAT mechanical inventory for `mcp-server-definition.v1`.

**Parent:** [`T-IGX-035-D.md`](T-IGX-035-D.md) (P3 publication @ `7172f294f` on `origin/development`).

**Baseline branch:** `origin/development` @ P4 cut (`7172f294f`).

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P4-01..05.

## Approved

Approved: operator-delegate @ 2026-11-11T12:04:00+01:00 | Delegation: MCP-SERVER-X P4 MERGE directive (operator chat 2026-11-11)

Disposition: **ACCEPT_SCOPED** — P4 lifecycle, fingerprint, eight app facades; P5 qualification next.

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

closed

## Checkpoint

| Field | Value |
| --- | --- |
| P4_PUBLICATION_SHA | `48181df7f99a0f9840e699dd774532672d4ffff3` |
| P4_CORRECTIVE_PRODUCT_SHA | `27e19f1feb31522c3103e6fce6f15a8c9395f380` |
| P4_MERGE_SHA (PR #454) | `7562a0692d7f33ce4423baf76988ced008e870fa` |
| IIIC evidence publication | `8fff85e3bfc5190929ff464b00fbe759dadcb5f8` |
| IIAC MCP activation phase 1 | `ce4b9f5ce118693d780659b38c2e11f701add0a3` |
| Operator activation | `IIAC-OD-MCP-ACT-N11-2026-11-11` @ `2026-11-11T13:43:00+01:00` |
| PR #455 | **MERGED** |
| ACT-N11 MCP rebind | **COMPLETE** — `verify_iiac_policy_activation --require-in-force` **0** @ publication |
| Residual | **B-P4-TEST-01** MemoryProvider admission (tracked; P5) |
| Disposition | **P4 CLOSED (scoped)** — P5 qualification next; stage MCP-SERVER-X **not** CLOSED |
