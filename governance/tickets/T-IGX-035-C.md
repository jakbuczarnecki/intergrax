# T-IGX-035-C — MCP-SERVER-X P2: definition contract and composition engine

## Scope

Batch **MCP-P2** only (MCP-P2-01..06): typed server definition contract, capability bindings, exposure allowlist, `_shared` composition engine, schema/visibility validation + error codes, architecture gate (no shadow registry). Wire `build_nexus_mcp_server` as compatibility facade (OD-MCP-05-A).

**Parent:** [`T-IGX-035-B.md`](T-IGX-035-B.md) (P1 publication @ `fb7ccd7807b570e4afd75dfcbb4ca2a6188c61ad`).  
**Baseline branch:** `origin/development` @ P2 cut (`0bded8a54ff41a0a561fcad2d4ef949e5c280f3d` includes P1 ladder).

**Operator gate:** OD-MCP-01..07 **APPROVED — Option A** @ `2026-11-11T11:24:00+01:00` (chat authorization).

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P2-01..06.

## Approved

Approved: operator @ 2026-11-11T11:24:00+01:00

Disposition: **ACCEPT_SCOPED** — P2 product slice after OD Option A package; ADR-MCP-01..07 remain **Proposed**.

## Test contract

```text
.venv/Scripts/python.exe -m pytest \
  tests/unit/contracts/test_mcp_server_definition.py \
  tests/unit/applications/test_mcp_server_engine.py \
  tests/unit/applications/test_mcp_server_engine_architecture_gates.py \
  tests/unit/applications/test_mcp_canonical_execution.py -q
```

Expected: exit **0**; regression `build_nexus_mcp_server` path green.

## Out of scope

P3+ transport consolidation beyond P2 unit needs; P4 facade migration of eight apps; P5–P8 qualification/IIAC; FRZ promotion; new `FRZ-MCP-*`; stage MCP-SERVER-X **CLOSED**; Resources/Prompts.

## Lifecycle state

in_progress

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `mcp-server-x-p2` |
| P2 gate | OD-MCP-01..07 **APPROVED — Option A** |
| Tasks | MCP-P2-01..06 targeted |
