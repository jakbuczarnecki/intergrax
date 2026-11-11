# T-IGX-035-D — MCP-SERVER-X P3: transport and hosting

## Scope

Batch **MCP-P3** only (MCP-P3-01..06): engine-owned FastMCP mount path (`mcp_host_mount`), streamable HTTP transport qualification matrix update, ASGI lifespan via existing `combine_lifespans` / harness coupling, `MCP_MOUNT_PATH` + `INCLUDE_MCP` fail-closed settings, startup failure on invalid server definition, concurrent tool-call bounds on the composition engine.

**Parent:** [`T-IGX-035-C.md`](T-IGX-035-C.md) (P2 publication @ `d46d9a0de316fea7b77f76c847028c9c932a9f3f`).

**Baseline branch:** `origin/development` @ P3 cut (`d46d9a0de316fea7b77f76c847028c9c932a9f3f` includes P2 ladder).

**Operator gate:** merge P2 + P3 batch authorized in chat @ 2026-11-11.

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P3-01..06.

## Approved

Approved: operator @ 2026-11-11T11:44:00+01:00

Disposition: **ACCEPT_SCOPED** — P3 transport/hosting; P4 facade migration remains blocked until P3 merge.

## Test contract

```text
.venv/Scripts/python.exe -m pytest \
  tests/unit/applications/test_fastapi_mcp.py \
  tests/unit/applications/test_mcp_host_mount.py \
  tests/unit/applications/test_mcp_engine_concurrency.py \
  tests/unit/applications/test_mcp_server_engine.py -q
```

Expected: exit **0**; `test_fastapi_mcp` regression green.

## Out of scope

P4 facade migration of eight apps; P5–P7 qualification execution; P8 independent audit/IIAC; FRZ promotion; new `FRZ-MCP-*`; stage MCP-SERVER-X **CLOSED**; Resources/Prompts; stdio/SSE transport.

## Lifecycle state

closed

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `mcp-server-x-p3` (merged PR #453) |
| P3_PUBLICATION_SHA | `4385c5e975492d0ee0d0049bd1c2af1ff785ea68` |
| P3_IMPLEMENTATION_SHA | `bee38a76b415cd4ca701fcc592e41b1f56e37f59` |
| P3 gate | P3 merge complete — P4 facade unblocked |
| Tasks | MCP-P3-01..06 delivered |
| IIAC note | `governance-fast` may be red on product Δ (ACT-N11); merge authorized by operator |
