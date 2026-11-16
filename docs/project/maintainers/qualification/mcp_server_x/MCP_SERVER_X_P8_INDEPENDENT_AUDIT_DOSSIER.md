# MCP-SERVER-X P8 — independent technical audit dossier (MCP-P8-02)

**Ticket:** [`T-IGX-035-I`](../../../../../governance/tickets/T-IGX-035-I.md) · **Parent:** [`T-IGX-035-H`](../../../../../governance/tickets/T-IGX-035-H.md) · **Plan:** `governance/tickets/plans/PLAN_T-IGX-035-A.md` § MCP-P8-02 · **Policy:** `IIAC-2026-11-05.1` · **Role:** `independent-quality-auditor`

**Verdict: `PASS_SCOPED`.** No mandatory blocker is open. The P7 technical claims for MCP-ACC-01..20 reproduce, with the limitations and findings below. This dossier is an audit result, not a certification (a third session certifies), not a stage closure and not an FRZ promotion.

## PIN block

| Item | Value |
| --- | --- |
| `IMPLEMENTATION_SHA` = `audited_implementation_sha` = `audit_execution_sha` | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` (merge commit of PR #474 on `origin/development`; PR head `94b6f6d6b8ae9882568c778fd7180b8b958fc175`) |
| `TEST_EXECUTION_SHA` (P7 package) | `abbe1525450da77da6801b080619a600c20cd497` |
| P6 terminal head used for the scope check | `6e53fb36f39b4a57331e6c079f2fcbe6094e2786` |
| Baseline used for R-P7-02 | `c5989282937f5e592748cb606b2272d04837c601` (P5 publication) |
| Auditor worktree | detached at the freeze SHA; evidence committed on branch `mcp-p8-audit-evidence` (not pushed) |
| Auditor session id | `e96831ad-2c06-4753-b8e9-e6aca67c2f5b` (taken from this session's scratchpad directory name; the host did not hand over a separate id) |
| Implementer label (P7 lanes L1, L2, L3, FI) | `T-IGX-035-H-MCP-P7-IMPLEMENTERS` |
| Certifier | `PENDING_CERTIFIER` (third session) |
| Evidence folder | [`iiac_evidence/MCP-SERVER-X/p8_audit/`](../iiac_evidence/MCP-SERVER-X/p8_audit/EVIDENCE_MANIFEST.txt) |

Every log in the evidence folder carries command, git head and UTC date in its header. One `pytest` process ran at a time (shared lock taken and released).

## Result per mission item

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Replay | **Reproduced.** New qualification (8 files) 164 passed; consistency file 14 passed (package: 13 plus the ACC-20 test, which needs the committed package); regression selection (19 files) 141 passed. All exit 0. | `p8_audit_replay_{new,consistency,regression}.log` |
| 2 | Traceability | **Confirmed.** AST count per ACC row and level equals the package table for rows 01..18 and the 9 `test_p7_05_*`; 145 test functions, 164 collected. No L3 module imports an L2 module and no L3 module uses an in-process client; the only patching in the L3 failure-injection file is the host-drain constant. ACC-19 evidence is the 141-pass regression log; ACC-20 evidence is the package record and its consistency test. | `test_matrix_p8.txt` |
| 3 | Test quality | **Sound, with findings** F-P8-07, F-P8-08 (LOW). Bodies of about 135 of 145 functions read; table below. No `pytest.raises(Exception)`, no skip/xfail in the nine P7 files. | below |
| 4 | Documentation vs behaviour | D-P7-01 confirmed and fixed correctly; D-P7-02 fixed but imprecise (F-P8-05); two overstated or incomplete claims (F-P8-04, F-P8-06); residuals R-P7-01, 03, 04, 05, 06 accurate; one unrecorded residual (F-P8-02). | findings JSON |
| 5 | R-P7-02 | **Disclosed and reproduced, not a blocker, a limitation plus finding F-P8-03 (MEDIUM).** | below |
| 6 | Scope and independence | `product_paths_changed_between(6e53fb36f, eadb1c211)` is `[]`; `verify_iiac_policy_activation --require-in-force` exit 0 (IN_FORCE @ `IIAC-2026-11-05.1`); FRZ checklist and acceptance matrix unchanged since `6e53fb36f`, validator reports `FRZ=225 PASS=108 OPEN=109`; session ids differ (below). | `p8_audit_*.log` |
| 7 | Governance record | P5, P6 and ACT-N11 evidence folders are byte-identical to `6e53fb36f` (empty `git diff --stat`); F-MCP-P6-GOV-001/002/003 and P7-PROC-01/02 are preserved in the package and the P6 dossiers; none is waived. Findings F-P8-01 (HIGH) and F-P8-09 (LOW) concern this area. | findings JSON |

## Findings (full text in `iiac_audit_findings_p8.json`)

| ID | Severity | Class | Statement | Proposed owner · disposition |
| --- | --- | --- | --- | --- |
| F-P8-01 | HIGH | governance-process | PR #474 merged at 10:42:10Z, before `Regression gate` (done 10:48:31Z) and `ENT-AUDIT-X` (done 10:58:09Z) finished; all later `success`. Contradicts the prospective gate of OD-MCP-P5-GOV-01 (item 3) and the delegation condition. | orchestrator + operator · record as new governance finding, preserved not waived |
| F-P8-02 | MEDIUM | security-residual-unrecorded | Without harness auth the engine trusts client-asserted tenant/user; tested at L2 only, not stated in the package; ACC-10/11/12 hold for authenticated hosts. | corrector (docs) / operator (product) · add residual R-P8-01 |
| F-P8-03 | MEDIUM | baseline-red-tests | R-P7-02 reproduced as 13 and 12 failures (not 11/11): core 11 stable plus variable TL11 subprocess startup timeouts; files are `gate` but PR/push CI is smoke-only. | orchestrator · keep as stated limitation, fix wording, track the test debt |
| F-P8-04 | MEDIUM | documentation-overclaim | Matrix row `Progress / cancel via transport: Supported`; progress has no code or test. | corrector (docs) · split the row |
| F-P8-05 | MEDIUM | documentation-incomplete | Wire-error vocabulary the matrix points to is not in the package; `When` column for `MCP_BINDING_UNKNOWN` / `MCP_DEF_INVALID` imprecise. | corrector (docs) · add wire-error table |
| F-P8-06 | LOW | documentation-overclaim | stdio absence is not shown (and not enforced) by HTTP path probes. | corrector (docs) · reword |
| F-P8-07 | LOW | evidence-scope | Cooperative host double, drain bound patched, no cancel-isolation, no slot-release-after-exception, L3 cross-tenant narrower than the row suggests. | corrector (docs) / optional tests |
| F-P8-08 | LOW | test-quality | Weak or self-referential assertions in the consistency file and two ACC-14/18 tests. | optional tests |
| F-P8-09 | LOW | governance-record | P5 findings and OD-MCP-P5-GOV-01 missing from the package carry-forward list. | corrector (docs) |

There is no BLOCKER. A BLOCKER would be a MCP-ACC row whose evidence does not reproduce, an L2 test used as L3, an auditor-visible product delta, or a waived governance finding; none was found. Severity was not lowered to reach the verdict: F-P8-01 stays HIGH because it repeats the pattern that earlier stages recorded as CRITICAL or HIGH; it is not a technical blocker because no product path is in the delta and the checks ended green.

## Test-quality review (item 3)

Judged by reading each body: does the assertion bind what the name and the package claim?

| Test(s) | Verdict | Note |
| --- | --- | --- |
| FI-L3 `acc_13_client_cancel_notification_mid_invoke...` | holds | Cancel is asserted inside the session before close, so closing cannot be the cause; request id 1 is hard-coded, a wrong id would fail the test. |
| FI-L3 `acc_13_unknown_task_id_invoke...` | holds | Typed `missing_execution_binding`; invoker never started. |
| FI-L3 `acc_14_client_deadline_on_slow_tool...` | holds, records behaviour | Pins that a client deadline does not stop server work (outcome `completed`). Honest RECORDED BEHAVIOUR. |
| FI-L3 `acc_14_sdk_request_timeout...` | holds | Asserts 408, then handler terminal state and a follow-up call. |
| FI-L3 `acc_14_client_deadline_on_hanging_run_agent...` | holds | Cancel arises from session close, which the comment states. |
| FI-L3 `acc_15_client_transport_teardown...`, `acc_15_raw_socket_close...` | holds | Real socket close; terminal CANCELLED, supervision drained, server reusable. Host is a cooperative double (F-P8-07). |
| FI-L3 `p7_05_limit_exceeded...`, `unknown_tool` (x2) | holds | Control calls prove slot release; unknown tool does not run the host (`assert_not_awaited`); raw test asserts no redirect. |
| FI-L2 `acc_13_cancel_of_in_flight_run_agent...`, `acc_15_disconnect_observation...` | holds as logic proof | Patches identity and supervision resolvers and `_client_disconnected`; module states it is not a transport test. |
| FI-L2 `acc_13_cancel_by_another_tenant_principal...` | holds | Negative then positive control; calls the control-plane function directly (L2). |
| FI-L2 `acc_14_caller_deadline...` | holds | Cancelled outcome plus slot release with `max_concurrent_tool_calls=1`. |
| FI-L2 `acc_14_client_deadline_through_in_memory_client...` | **weak** | Accepts `completed` or `cancelled` (F-P8-08). |
| FI-L2 `acc_14_canonical_execution_deadline_is_not_reachable...` | **weak** | Test-fixture host and source-text substring (F-P8-08). |
| FI-L2 `p7_05_limit_exceeded_result_bound_*` (x3) | holds | Over-ceiling only; custom-tool gap pinned as RECORDED GAP (R-P7-03 accurate). |
| L3 invoke `acc_09_*` (x3) | holds, scoped | Principal binding into the executed task; host is an `AsyncMock`, so no agent run (package says scoped). |
| L3 invoke `acc_10_*` (x5), `acc_11_*` (x3) | holds | Exact 401 body, no tool-name leak, `identity_conflict:` prefix, concurrent isolation with 6 calls, cross-host credential separation, execution never awaited. |
| L2 security `acc_10_*` (x9) | holds | `permit` control test makes the deny test non-vacuous; invoker never reached on deny. |
| L2 security `acc_11_*` (x4), `acc_12_*` (x8) | holds | `wrong_tenant`, forged tenant/user, definition default must not win, tenant-required forbidden. `acc_12_local_dev...` pins fail-open identity (F-P8-02). |
| L2 security `acc_17_*` (x4) | holds, scoped | Ids validated and distinct per invocation; no MCP trace id asserted or denied (R-P7-01 accurate). |
| L3 protocol `acc_18_*` (x10) | holds | Exact capability snapshot, `-32601` for tasks/completions/subscribe, version negotiation; `matrix_rows_reference_existing_tests` is self-referential (F-P8-08). |
| L3 publication `acc_01/05/06/07/08` | holds | Exact tool sets, schema equality against `to_mcp_tools`, variants for retired/hidden/disabled/empty, 401 before session. stdio claim overstated (F-P8-06). |
| L1 `acc_02_*`, `acc_04_*`, `acc_16_*` | holds | Fail-closed allowlist, order, strip, golden fingerprint plus sensitivity tests; subprocess import-isolation check. |
| L2 composition `acc_03/04/16_*` | holds | Typed codes at composition, retired binding without redirect, recomposition restores. |

### Mutation-style thought checks (no product code was changed)

1. Engine releases the admission slot only on success: **killed** for cancel (L2/L3 follow-up call with a limit of 1), **survives** for handler exceptions.
2. `notifications/cancelled` cancels every in-flight request or ignores `requestId`: **survives** (one in-flight request per test).
3. Cancel hook stops draining at once instead of waiting for the host: **survives** at production timing because the drain constant is patched to 0.25 s and the host double is cooperative.
4. Principal check drops the `user_id` comparison: **killed** (`forged_user_assertion` at L2 and L3).
5. Exposure policy matches by tool name instead of binding id: **partly killed** (`echo_v1`/`qual_echo` retire test and `reference binding` variants), not for built-ins whose ids differ from names.
6. Result bound raised or removed: **killed**; bound lowered below real payloads: **survives** (only over-ceiling payloads are used).
7. Auth middleware skipped for the no-trailing-slash route: **survives at L3** (L3 probes `/mcp/`), killed at L2 by `TestClient`.
8. Capability snapshot gains a `tasks` capability: **killed** (`capabilities.tasks is None` and exact `model_dump`).

## R-P7-02 (item 5)

- Own reproduction of the three files: **13 failed / 16 passed at the freeze SHA, 12 failed / 17 passed at `c59892829`**. Package and P7 baseline logs: 11 / 11. Comparison of the failing test ids: the 11 packaged ids all fail in both of my runs; the extra ones are `test_tl11_e2e_slow_list_within_deadline` (both) and `test_tl11_e2e_http_ingress_rejects_missing_api_key` (freeze only), both `tl11_catalog_qualification_subprocess_startup_timeout` or `no_port_line`. Other Python processes (not mine) were running on the machine, so the TL11 set is load dependent.
- The pre-existing statement holds: the same core fails at the P5 publication SHA and at the freeze SHA, and no P6/P7 change touches these files. The causes in the logs are product hardening that the tests do not follow (`6d436f323 require explicit provider selection`, revision-bound agent contract authority) and subprocess startup timing.
- Decision: **not a blocker, a finding (F-P8-03) and a stated limitation.** Reason: no MCP-ACC row, replay set or regression file depends on these files, the exclusion is disclosed in the package, the selection and a test, and the failures are not counted as passing. Not accepted without comment: the files are `gate`-marked, one guards the MCP opt-in default, CI on PR/push runs only the smoke profile, and the claim `same set` is too strong.

## Independence (item 6)

The P7 lanes carry the label `T-IGX-035-H-MCP-P7-IMPLEMENTERS`; the P6 manifests use `T-IGX-035-G-MCP-P6-IMPLEMENTER`, auditor `b8d2e7f4-1a3c-4f9e-8d62-5c1a0b9e7f3d` and certifier `local_7c8553da-3398-4f9b-8fae-c794a63d6cb5`. This session's id differs from all of them and from the signing and verifier sessions named in the host's linked-session note; it authored no P2–P7 file, test or record. `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py --strict-product-diff` results on the manifest are in the evidence folder (certifier fields pending by design).

## What was not verified

- Mutation execution against product code (read-only role); the checks above are reasoning.
- Dynamic FastMCP progress behaviour (F-P8-04 rests on absence of code and tests).
- The GitHub `development` push run after the merge (taken from the orchestrator's statement; the PR #474 runs were read directly).
- `LIM-RESEARCH-STRICT-FACTORY-MEMORY` and `OPEN-BP-01` (carried, not re-examined); any test outside the three replay sets and the R-P7-02 files; CI of this evidence commit; Windows-only behaviour on other machines.
- Whether the P7 lanes' lane reports (`p7_qualification/lane_reports/`) match their logs beyond the numbers cross-checked here.

## Limitations of the verdict (carried to the certifier)

1. `PASS_SCOPED` covers tools-only streamable HTTP on loopback with the reference server; ACC-09 `run_agent` with a substituted delegate or host double; ACC-14 client-side deadlines only; ACC-17 without a dedicated MCP trace id (R-P7-01); ACC-18 as an observed subset.
2. R-P7-02 stays open (F-P8-03). R-P7-03..06 stay open as the package states.
3. F-P8-01 (HIGH) needs an operator disposition before closure; it is preserved here, not waived.
4. The auditor did not certify and did not edit any product, test, ticket, roadmap or existing document.

## Re-review of the P8 corrections (MCP-P8-03, round 1)

**Re-review SHA:** `c50547012c353463e86525411a5a065f43e5bab8` (orchestrator corrections on top of the audit evidence commit `69a2333cc`). **Result: no finding is still open; verdict stays `PASS_SCOPED`.** The diff against `eadb1c211` is documentation only (`product_paths_changed_between` is `[]` for both pins, no test, script, governance or workflow file changed), so the P7 `TEST_EXECUTION_SHA` results are untouched. One pytest process ran under the shared lock: `test_mcp_server_x_p7_l1_contract.py` plus `test_mcp_server_x_p7_matrix_consistency.py` read the corrected matrix and package and gave **55 passed** (`reaudit/p8_reaudit_replay_l1_consistency.log`). `verify_iiac_policy_activation --require-in-force` exit 0; FRZ counter unchanged (`FRZ=225 PASS=108 OPEN=109`).

| Finding | Re-review disposition | Reason (short; full text in `iiac_audit_findings_p8.json`, field `reaudit_reason`) |
| --- | --- | --- |
| F-P8-01 HIGH | `RECORDED_PENDING_OPERATOR_DISPOSITION` | Recorded as P7-PROC-03, not waived, no retroactive approval. A documentation fix cannot resolve a HIGH governance finding; it needs an operator decision. Nit: `evol-x-b4-preflight` finished 10:44:50Z, 10:48:31Z is the end of its run. |
| F-P8-02 MEDIUM | `ACCEPTED_SCOPED_LIMITATION` | R-P8-01 in the package (row and ACC-10/11/12 cells) and the gap register; product decision left to the operator. |
| F-P8-03 MEDIUM | `ACCEPTED_SCOPED_LIMITATION` | R-P7-02 reworded to core 11, observed 11-13, markers and CI scope; still open, not counted as passing. The test-debt item is not yet tracked outside the corrections record. |
| F-P8-04 MEDIUM | `RESOLVED` | Matrix row split: Cancel supported, Progress not claimed and not evidenced. |
| F-P8-05 MEDIUM | `RESOLVED` | `Wire errors observed at P7` table checked string by string against the P7 assertions and the source; `When` columns match. Non-blocking omissions: `unknown_tool:<id>` payload, protocol-level errors, 401 body text; `identity_conflict:` also under ACC-10. |
| F-P8-06 LOW | `RESOLVED` | ACC-05 wording is narrower and accurate (SSE-style and stdio-named paths 404; stdio not enforced). |
| F-P8-07 LOW | `ACCEPTED_SCOPED_LIMITATION` | Items (a)-(e), (g) stated in the ACC-10/11/13/15 cells; (f) not stated, non-blocking. |
| F-P8-08 LOW | `ACCEPTED_SCOPED_LIMITATION` | (a), (b) in the ACC-14 cell; (c)-(f) only in this register and dossier, although the record says all are in the ACC-14 cell. |
| F-P8-09 LOW | `RESOLVED` | P5 findings and OD-MCP-P5-GOV-01 added to the carried list. |

Nothing was weakened to keep `PASS_SCOPED`: every correction either narrows a claim, adds a limitation or records a process finding. What the certifier must carry as stated limitations: F-P8-01 (until an operator decision exists), R-P7-02 (with the untracked test debt), R-P8-01, and the F-P8-07/08 limits. Not verified in the re-review: the content of the PR or CI for the corrections commit, and the runtime of any test other than the 55 above.
