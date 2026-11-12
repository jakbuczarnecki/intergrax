# T-IGX-035-H — MCP-SERVER-X P7: exact-SHA qualification package (MCP-ACC-01..20)

## Scope

Batch **MCP-P7** only (MCP-P7-01..09): L1/L2/L3 qualification, announced-protocol-subset matrix, failure injection, regression of the existing MCP and TOOL-LIFE paths touched by the diff, EVOL change-impact and ENT delta records, clean-checkout replay, and the exact-SHA evidence package (`TEST_EXECUTION_SHA`, MCP-ACC-01..20 matrix with test IDs).

**Parent:** [`T-IGX-035-G.md`](T-IGX-035-G.md) — P6 closed (scoped); P6 terminal verification `P6_TERMINAL_VERIFIED` @ `6e53fb36f39b4a57331e6c079f2fcbe6094e2786`; P7 entry **OD-MCP-P6-P7-01** @ 2026-11-12T07:35:55+01:00.

**Baseline SHA:** `ab22e11e8fd0bedbb41171746df14d63c309fcb5` (`origin/development` at the P7 cut; a docs-only descendant is acceptable).

**Tasks:** MCP-P7-01, MCP-P7-02, MCP-P7-03, MCP-P7-04, MCP-P7-05, MCP-P7-06, MCP-P7-07, MCP-P7-08, MCP-P7-09 (canonical text in the plan below).

**Decision DR-1 (operator, chat 2026-11-12):** `REQ-MCP-OBS-01` (dedicated MCP invocation trace id) stays a documented residual. MCP-ACC-17 uses the P5 discovery/adversarial pack requalified at the P7 SHA, with the limitation stated in the matrix and the final report. **No product-path change in P7.**

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P7-01..09.

## Approved

Approved: operator-delegate @ 2026-11-12T09:07:52+01:00
Delegation: docs/project/maintainers/qualification/IIAC_OD_DELEGATION_MCP_P7_P8_AUTH_2026-11-12.md@bddcbb293f53812bebb0134b2278d8f038a86825

## Test contract

Bounded, one `pytest` process at a time (test process budget); no broad diagnostic suites.

```text
.venv/Scripts/python.exe -m pytest tests/qualification/mcp_server_x -q
.venv/Scripts/python.exe -m pytest <files listed in tests/qualification/mcp_server_x/selection.py and p6_selection.py> -q
.venv/Scripts/python.exe -m scripts.governance.igx_gov ticket
```

Expected: exit **0** for each; every log carries the command, `git rev-parse HEAD` and the UTC date.

## P7 acceptance criteria (pre-implementation)

| # | Criterion | Verification |
| --- | --- | --- |
| AC-P7-01 | Every MCP-ACC-01..20 has a test ID at one `TEST_EXECUTION_SHA` or a justified N/A. | Matrix consistency check |
| AC-P7-02 | L2 reports are labelled "not a physical transport test"; L3 is physical streamable HTTP on loopback. L2 is never counted as L3. | Report review |
| AC-P7-03 | Announced protocol subset matrix: checked vs not announced; UNSUPPORTED stays UNSUPPORTED. | L3 matrix |
| AC-P7-04 | Failure injection (drop mid-invoke, limit exceeded, unknown tool) ends in terminal states with no hang. | L2/L3 logs |
| AC-P7-05 | Regression of existing MCP tests and the TOOL-LIFE tests touched by the diff is green; no new red on cancel/catalog paths. | Selection run |
| AC-P7-06 | EVOL change-impact record and ENT Mode A delta at the exact SHA, without invalidating any CLOSED stage. | Records |
| AC-P7-07 | Clean-checkout replay on a second worktree at the same SHA gives the same result. | Replay log |
| AC-P7-08 | `product_paths_changed_between(6e53fb36f, HEAD)` is `[]` (no product-path change). Otherwise STOP (ACT-N11). | Script |
| AC-P7-09 | Independence: the P7 testers and test authors are not the P8 auditor or certifier. | Session ids in records |

## Out of scope

P8 audit/certification/closure, any product-path change, FRZ promotion, MCP-SERVER-X stage CLOSED, stdio/SSE transport, Resources/Prompts (UNSUPPORTED_BY_DESIGN v1), Docker, new vendor integrations, production configuration, broad diagnostic pytest.

## Lifecycle state

**quality_gate**

## Checkpoint

| Field | Value |
| --- | --- |
| Baseline | `ab22e11e8fd0bedbb41171746df14d63c309fcb5`; authorization pack merged @ `d5c1383fc15c005daafd131eaf03b3c49393b792` |
| TEST_EXECUTION_SHA | `abbe1525450da77da6801b080619a600c20cd497` |
| Product-path delta | none (AC-P7-08) |
| New P7 tests | 164 passed (8 files), clean-checkout replay identical |
| Consistency | 13 passed (+ ACC-20 package test after the record commit) |
| Regression selection | 141 passed; 11 pre-existing failures excluded (R-P7-02) |
| Records | `MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md`, `MCP_SERVER_X_P7_EVOL_ENT_DELTA_RECORD.md`, `iiac_evidence/MCP-SERVER-X/p7_qualification/` |
| Findings | D-P7-01, D-P7-02 fixed; R-P7-01..R-P7-06 open (scoped); P7-PROC-01, P7-PROC-02 recorded |
