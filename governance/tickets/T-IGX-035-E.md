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

plan_approved

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `mcp-server-x-p4` (corrective follow-on PR vs `7562a069`; #454 merged) |
| Corrective SHA | `27e19f1feb31522c3103e6fce6f15a8c9395f380` (`F-MCP-ACT-N11-001`) |
| P3_PUBLICATION_SHA | `7172f294f` |
| Tasks | MCP-P4-01..05 targeted |
| ACT-N11 | No documented product-path allowlist amend for `mcp_server_definition.py` / `mcp_server_engine.py` — expect governance-fast product Δ; operator merge per prior MCP batches |
| Gate | **P4 IMPLEMENTED** (scope) — merge **BLOCKED** (`governance-fast` ACT-N11) |
| ACT-N11 batch | [`IIAC_ACT_N11_MCP_RECOVERY_BATCH.md`](../../docs/project/maintainers/qualification/mcp_server_x/IIAC_ACT_N11_MCP_RECOVERY_BATCH.md) — OD **PREPARED**; auditor + operator activation **PENDING** |
| CI @ `06be8bfd4` | `governance-fast` **FAIL** run `43010550205`; other mandatory PR gates **PASS** |
