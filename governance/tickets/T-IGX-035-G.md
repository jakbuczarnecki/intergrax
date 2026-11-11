# T-IGX-035-G — MCP-SERVER-X P6: public API, reference server, migration, DX docs

## Scope

Batch **MCP-P6** only (MCP-P6-01..04): consolidated developer public API, P6 reference qualification server (L3), eight-application compatibility/migration matrix, developer and migration documentation, deterministic P6 acceptance suite.

**Parent:** [`T-IGX-035-F.md`](T-IGX-035-F.md) — P5 closed (scoped) @ `c59892829`; P6 entry **OD-MCP-P5-P6-01** @ 2026-11-11T16:24:00+01:00.

**Baseline SHA:** `e48106740a4a243204cadccb6a85db9a8710d6af` (`origin/development` @ P6 cut)

**Tasks:** MCP-P6-01, MCP-P6-02, MCP-P6-03, MCP-P6-04; P6 qualification batch (requirement-to-test mapping); G6 handoff package (separate session).

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P6-01..04.

## Approved

Approved: operator @ 2026-11-11T16:53:00+01:00

Operator directive (chat 2026-11-11): approve MCP-SERVER-X P6-01..04 implementation and qualification scope per canonical plan digest `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7`.

## Test contract

```text
.venv/Scripts/python.exe -m pytest tests/qualification/mcp_server_x/test_mcp_server_x_p6_qualification_batch.py -q
```

Expected: exit **0** (consolidated P6 batch).

## P6 acceptance criteria (pre-implementation)

| # | Criterion | Verification |
| --- | --- | --- |
| AC-P6-01 | Typed `McpServerDefinition` and validators are the supported contract surface (`intergrax.contracts.mcp_server_definition` + `mcp_developer_api`). | L1 import + contract tests |
| AC-P6-02 | Tier-3 facades compose via `compose_application_mcp_server` / public API only — no catalog bridge or engine private imports. | Import policy test + eight-app matrix |
| AC-P6-03 | Reference qualification server under `tests/qualification/mcp_server_x/reference_app/` exercises MCP-ACC-01 scenarios on **streamable HTTP** L3. | Dedicated L3 pytest |
| AC-P6-04 | Migration matrix documents every application: legacy → canonical → change → test → result. | SSOT markdown + regression |
| AC-P6-05 | Developer guide matches executable behavior (@ implementation SHA). | Doc review + example tests |
| AC-P6-06 | P5/P4 regression subset green in P6 batch. | Selection batch |
| AC-P6-07 | `LIM-RESEARCH-STRICT-FACTORY-MEMORY` preserved as cross-stage PROD-Q dependency (no fabricated vendor qual). | Research test contract unchanged |
| AC-P6-08 | Independent G6 audit **before** ticket `closed` (no self-certification; F-MCP-P5-GOV-001 sequencing). | Separate audit session |

## Implementation contract

- Invariant: single MCP composition authority (`compose_mcp_server_from_definition`); no second registry or execution path
- Reference server is test qualification asset only — not a ninth product application
- Resources/Prompts remain UNSUPPORTED_BY_DESIGN v1
- No FRZ promotion; no MCP-SERVER-X stage CLOSED; no P7/P8 certification claims

## Out of scope

P7 MCP-ACC exact-SHA package @ publication, P8 IIIC stage close, FRZ PASS, stdio/SSE transport, operator IIAC override.

## Lifecycle state

**quality_gate**

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `mcp-server-x-p6` (from `e48106740`) |
| P6 entry OD | OD-MCP-P5-P6-01 APPROVED @ 2026-11-11T16:24:00+01:00 |
| PARENT publication | P5 `c59892829` |
| P6_BASELINE_SHA | `e48106740a4a243204cadccb6a85db9a8710d6af` |
| P6_IMPLEMENTATION_SHA | `5ff8606b089828b334ebac133518ba47c9f6f7ce` |
| PR | #459 → `development` |
| LOCAL_TEST_EXECUTION | P6 batch **PASS** @ `5ff8606b0` |
| `igx_gov ticket` | **PASS** @ `5ff8606b0` |
| CI | `ci-smoke` **PASS**; `governance-fast` **FAIL** (ACT-N11 — `MCP_SERVER_X_P6_ACT_N11_TRIGGER.md`) |
| G6 | **PENDING** — `MCP_SERVER_X_P6_G6_AUDIT_HANDOFF.md` |
| ACT-N11 | **TRIGGERED** — IIIC P6 rebind required before merge |
