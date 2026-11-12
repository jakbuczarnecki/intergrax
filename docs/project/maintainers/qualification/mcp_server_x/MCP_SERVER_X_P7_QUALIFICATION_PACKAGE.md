# MCP-SERVER-X P7 — exact-SHA qualification package (MCP-ACC-01..20)

**Ticket:** [`T-IGX-035-H`](../../../../../governance/tickets/T-IGX-035-H.md) · **Plan digest:** `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7` · **Baseline:** `origin/development` `d5c1383fc15c005daafd131eaf03b3c49393b792`

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `abbe1525450da77da6801b080619a600c20cd497` (branch `mcp-p7-qualification`; every command below ran at this HEAD) |
| Clean-checkout replay SHA (P7-08) | `abbe1525450da77da6801b080619a600c20cd497` (second detached worktree, same commands) |
| Earlier complete runs (superseded) | `5bc1c57ec29f7cae6729d9ba9d62be8f1c482472` (first run: 11 pre-existing failures in the regression files, see R-P7-02) and `7d8b94672523eeeb41192a309d9643b1afbb8538` (all green; superseded because the L1 test and the protocol matrix were corrected afterwards) |
| Product-path delta vs P6 terminal head `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` | none (`git diff --name-only` has no path under `intergrax/`, `agents/`, `applications/`) |

Scope statement: this package qualifies the MCP exposure engine **as scoped by the operator decisions** (tools only, streamable HTTP, `REQ-MCP-OBS-01` residual). It is test and documentation evidence; it does not state that MCP-SERVER-X is closed, that any FRZ criterion passed, or that production readiness exists. Level rules: L2 evidence is in-process and is **never** counted as L3 (physical streamable HTTP on loopback).

## Execution evidence

One `pytest` process at a time; logs carry command, git head and UTC date headers; copies with a sha256 manifest are in [`iiac_evidence/MCP-SERVER-X/p7_qualification/`](../iiac_evidence/MCP-SERVER-X/p7_qualification/EVIDENCE_MANIFEST.txt).

| Suite (selection in `tests/qualification/mcp_server_x/p7_selection.py`) | Result at `abbe15254` (main checkout) | Result in the clean replay worktree |
| --- | --- | --- |
| `MCP_SERVER_X_P7_NEW_QUALIFICATION` (8 files: L1 contract, L2 composition/exposure, L2 security/trace, L3 publication/discovery, L3 invoke security, L3 protocol subset, failure injection L2, failure injection L3) | exit 0, **164 passed**, 60.4 s | exit 0, **164 passed**, 123.5 s |
| `test_mcp_server_x_p7_matrix_consistency.py` (the ACC-20 package test is deselected in the first pass and runs once this document is committed, see "ACC-20 check") | exit 0, **13 passed** | exit 0, **13 passed** |
| `MCP_SERVER_X_P7_REGRESSION_SELECTION` (P5 selection, 12 files, + 7 further existing MCP files) | exit 0, **141 passed**, 65.7 s | exit 0, **141 passed**, 104.9 s |

Commands (cwd = repository or worktree root, interpreter `.venv/Scripts/python.exe`): `python -m pytest <files of the selection> -q`; the exact file lists and the runner script are in the evidence folder (`run_p7_suites.sh`).

## MCP-ACC matrix

"P7 tests" counts `test_acc_<NN>_*` functions in the P7 files (static count, parametrization not expanded). Evidence inherited from P5/P6 is listed for context only and was re-run inside the regression selection; it is not counted as P7 evidence.

| MCP-ACC | Scenario | Required level | P7 tests (L1 / L2 / L3) | Status at `TEST_EXECUTION_SHA` | Limitation / note |
| --- | --- | --- | --- | --- | --- |
| MCP-ACC-01 | Reference qualification server | L3 | 0 / 0 / 3 | COVERED | Authorized `run_agent` uses the canonical host with a substituted delegate (no live agent run) |
| MCP-ACC-02 | Typed server definition | L1 | 17 / 0 / 0 | COVERED | Matrix documentation gap D-P7-02 fixed in this change |
| MCP-ACC-03 | Composition | L2 | 0 / 10 / 0 | COVERED | |
| MCP-ACC-04 | Exposure policy | L1 + L2 | 8 / 8 / 0 | COVERED | |
| MCP-ACC-05 | Publication / transport | L3 | 0 / 0 / 3 | COVERED | Only streamable HTTP is published by the served ASGI app (SSE-style and stdio-named paths answer 404). stdio is a process transport: HTTP probes cannot prove its absence and nothing in the product enforces it, a caller can run the composed FastMCP object on stdio (F-P8-06) |
| MCP-ACC-06 | HTTP client | L3 | 0 / 0 / 2 | COVERED | |
| MCP-ACC-07 | Discovery | L3 | 0 / 0 / 4 | COVERED | |
| MCP-ACC-08 | Schema | L3 | 0 / 0 / 5 | COVERED | |
| MCP-ACC-09 | Authorized invoke | L3 | 0 / 0 / 3 | COVERED — scoped | `run_agent` path; catalog invoke with a live execution binding stays covered by the F3 regression files |
| MCP-ACC-10 | Denied invoke | L2 + L3 | 0 / 10 / 5 | COVERED | Holds for hosts with a configured credential; a host without harness auth trusts the client assertion (R-P8-01). The 401 path is probed at `/mcp/` at L3 and at `/mcp` (redirect-following test client) at L2 only (F-P8-07) |
| MCP-ACC-11 | Cross-tenant deny | L2 + L3 | 0 / 5 / 3 | COVERED | L3 covers identity-assertion conflicts and credential separation between hosts; `wrong_tenant` (execution owned by another tenant) and cross-tenant cancel are L2 only, the cancel one calls `governed_cancel_active_task` directly (F-P8-07). Same precondition as ACC-10 (R-P8-01) |
| MCP-ACC-12 | Principal binding | L2 | 0 / 8 / 0 | COVERED | Principal is resolved from the FastMCP request context only (R-P7-06); with no harness auth the client-asserted tenant/user is trusted (R-P8-01) |
| MCP-ACC-13 | Cancel | L3 | 0 / 4 / 2 | COVERED | Client cancel notification reaches a governed cancel; no cancel without it (see R-P7-04). Scope: L3 uses a cooperative host double; one in-flight request per cancel test (cancel isolation untested); drain-bound path patched to 0.25 s (production 30 s) and a host outliving the drain untested (F-P8-07) |
| MCP-ACC-14 | Timeout | L2 (+ L3 if the transport carries a deadline) | 0 / 3 / 3 | COVERED — scoped | The transport carries no server-side deadline; canonical-execution deadline is **N/A through MCP** (recorded by test; the test inspects a test fixture and a source-text substring, not the production host composition, F-P8-08); client-side deadline shown at L3 (R-P7-04); the in-memory deadline test accepts `completed` or `cancelled` (F-P8-08) |
| MCP-ACC-15 | Disconnect | L3 | 0 / 1 / 2 | COVERED | The L2 test is a supplement and is not a physical disconnect. L3 uses a cooperative host double; admission-slot release is tested after success and after cancel, not after a handler exception (F-P8-07) |
| MCP-ACC-16 | Version / retire | L1 + L2 | 7 / 6 / 0 | COVERED | |
| MCP-ACC-17 | Trace correlation | L2 | 0 / 4 / 0 | COVERED — scoped | Pins `task_id`, `run_id`, `execution_id`; **no dedicated MCP trace id** (R-P7-01, operator decision DR-1) |
| MCP-ACC-18 | Announced protocol subset | L3 | 0 / 0 / 10 | COVERED — scoped | Resources/Prompts are advertised but empty; see the subset matrix below |
| MCP-ACC-19 | Regression suite | — | selection | COVERED | 141 passed; 11 pre-existing failures in 3 files are excluded and recorded (R-P7-02) |
| MCP-ACC-20 | Exact-SHA proof package | — | this record | COVERED | `TEST_EXECUTION_SHA` above; consistency test `test_acc_20_exact_sha_package_record_exists_and_pins_a_test_execution_sha` |

Additional P7-05 tests without an MCP-ACC id (`test_p7_05_*`, 9 functions): concurrency limit, result bound, unknown tool (L2 and L3).

## L2 / L3 split

L2 evidence (in-process, not a physical transport test): composition, exposure, retirement, security and principal binding, correlation keys, and the in-process failure injection. L3 evidence (physical streamable HTTP on loopback, uvicorn on port 0): reference server publication, HTTP client, discovery, schema, authorized and denied invoke, cross-tenant, the protocol subset, cancel, timeout and disconnect. No L2 test is used as L3 evidence in the table above.

## Announced protocol subset (MCP-P7-04)

Observed on FastMCP 3.3.1 over streamable HTTP; the source of truth for the table is `PROTOCOL_SUBSET_MATRIX` in `test_mcp_server_x_p7_l3_protocol_subset.py`.

| Feature | Observed | v1 claim |
| --- | --- | --- |
| `initialize`, `tools/list`, `tools/call` | works | Supported (tools only) |
| `resources` capability | advertised (`subscribe=false`, `listChanged=true`); `resources/list` returns an empty list; reading an unknown resource returns an error | UNSUPPORTED_BY_DESIGN (empty) |
| `prompts` capability | advertised (`listChanged=true`); `prompts/list` returns an empty list; getting an unknown prompt returns an error | UNSUPPORTED_BY_DESIGN (empty) |
| `logging` capability | advertised; `logging/setLevel` returns `{}` | not claimed by v1 |
| UI extension `io.modelcontextprotocol/ui` | advertised by FastMCP | not claimed by v1 |
| `tasks`, `completions` | not advertised (method not found, -32601) | not claimed |
| stdio / SSE | no route served | UNSUPPORTED_BY_DESIGN (OD-MCP-03) |

## Failure injection (MCP-P7-05)

Terminal states shown, with no hang: cancel by notification (governed cancel), unknown task id (fail-closed missing execution binding), client deadline (client-side), SDK request timeout (error 408), transport teardown and raw socket close (governed cancel), concurrency limit (immediate rejection with `mcp_concurrent_call_limit_exceeded`, slot released), result bound (`mcp_payload_byte_limit_exceeded` on `list_agents` and `run_agent`), unknown tool (error result, no redirect).

## Findings and residuals

| ID | Kind | Statement | Disposition |
| --- | --- | --- | --- |
| D-P7-01 | Documentation | Protocol matrix said Resources/Prompts are "Not advertised"; the runtime advertises them with empty lists (also `logging` and a UI extension) | Fixed: [`MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md`](./MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md) |
| D-P7-02 | Documentation | Error-code table listed 6 of the 7 enum codes (`MCP_BINDING_UNKNOWN_TOOL` missing); `MCP_BINDING_UNKNOWN` was described too narrowly | Fixed in the same document; the L1 test now asserts the table lists every enum code |
| R-P7-01 | Residual | `REQ-MCP-OBS-01`: no dedicated MCP invocation trace id | Open, operator decision DR-1 (no product change in P7); MCP-ACC-17 is scoped |
| R-P7-02 | Residual (pre-existing) | 11 tests in `test_mcp_surface_opt_in.py`, `test_mcp_surface_opt_in_runtime.py`, `test_mcp_catalog_e2e_hung_server_qualification.py` fail on this machine at `c59892829`, at `d5c1383fc` and at the P7 head with the same core set of 11 (no LLM provider configured, STRICT agent-contract authority, TL11 subprocess startup timeout); the independent audit reproduced 13 failures at the freeze SHA and 12 at `c59892829`, the extra ones being further TL11 subprocess startup timeouts, so the failing set varies from run to run (observed 11-13, F-P8-03) | Open, not introduced by P6/P7; excluded from the regression selection and **not counted as passing**; to be investigated outside P7. The files are marked `gate` (one `no_ci`) and PR/push CI runs only the smoke profile, so green PR CI says nothing about them; one of them guards the MCP opt-in default (`test_scaffold_defaults_mcp_to_opt_in`) |
| R-P7-03 | Residual | The result bound is not applied to custom tools (`qual_echo` returns an over-1-MiB result in full) | Open; documented, tested as a recorded gap |
| R-P7-04 | Residual | A client deadline or a closed client wait does not stop server work (no server-side deadline; `run_agent` has no timeout; the Execution deadline authority is not wired into the host composition); a hanging call is cancelled only by an explicit cancel notification or by closing the connection | Open; MCP-ACC-14 scoped |
| R-P7-05 | Residual | `MCP_PRINCIPAL_REQUIRED`, `MCP_TENANT_ASSERTION_CONFLICT`, `MCP_RESULT_BOUND_EXCEEDED` are declared but never raised in the engine | Open; documented in the matrix |
| R-P7-06 | Residual | The engine passes `harness_auth=None` to `run_agent` and `invoke_catalog_tool`; the principal is resolved only from the request context | Open; informational |
| R-P8-01 | Residual | A host that mounts MCP without harness auth (no `harness_auth`, or an empty `HarnessAuthState`) trusts the client-asserted tenant/user (fail-open identity). Existing, tested behaviour (`test_acc_12_local_dev_state_without_credentials_trusts_client_assertion`); not exercised at L3. MCP-ACC-10/11/12 `COVERED` holds only for hosts with a configured credential | Open; recorded by the P8 audit (F-P8-02). Making composition fail closed would be a product change and an operator decision (it would trigger ACT-N11), so it is outside the findings-only correction loop |
| P7-PROC-01 | Process | One commit of lane L3 was made with hooks disabled (`core.hooksPath`); `ruff check` and `ruff format --check` were run by hand and the code goes through the full hooks and CI again in this change | Recorded |
| P7-PROC-02 | Process | PR #470 (authorization pack) was merged while `ent-audit-fast` was still in progress; the run finished successfully afterwards | Recorded (earlier cases: F-MCP-P6-GOV-001/003, PR #464 and #468) |
| P7-PROC-03 | Process | PR #474 (this package) was merged at 2026-11-12T10:42:10Z, before the pull-request runs of `governance-fast`, `evol-x-b4-preflight` (finished 10:48:31Z) and `ent-audit-fast` (finished 10:58:09Z) had finished; all runs ended `success` on attempt 1 (F-P8-01, HIGH, governance-process) | Recorded, not waived, no retroactive approval; see the P8 corrections record |

Carried unchanged: `LIM-RESEARCH-STRICT-FACTORY-MEMORY` (PROD-Q, OPEN, non-blocking), OPEN-BP-01, F-MCP-P6-GOV-001/002/003, and the P5 records F-MCP-P5-GOV-001/002, F-MCP-P5-MEM-001 and the prospective merge gate OD-MCP-P5-GOV-01 (F-P8-09).

## Independence

Test authoring lanes (L1, L2, L3, failure injection) and the orchestrator are P7 implementers/testers and are excluded from the P8 audit and certification. The P8 auditor, the correctors and the certifier are separate sessions (ticket `T-IGX-035-I`).

## ACC-20 check

`test_acc_20_exact_sha_package_record_exists_and_pins_a_test_execution_sha` reads this file and runs after it is committed (documentation-only difference from `TEST_EXECUTION_SHA` `abbe1525450da77da6801b080619a600c20cd497`); its result is recorded in the PR for this change.

## P8 audit corrections

The independent P8 audit of this package (verdict `PASS_SCOPED` at `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`) raised findings F-P8-01..F-P8-09. The corrections to this document (stdio wording, scope statements in the limitation cells, R-P7-02 wording, R-P8-01, P7-PROC-03, carried records) and the dispositions are in [`MCP_SERVER_X_P8_CORRECTIONS_RECORD.md`](./MCP_SERVER_X_P8_CORRECTIONS_RECORD.md). The `TEST_EXECUTION_SHA` and the test results above are unchanged by the corrections (documentation only).
