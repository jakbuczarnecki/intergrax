# MCP-SERVER-X RC — independent delta re-audit dossier (G6, addendum)

| Field | Value |
| --- | --- |
| **Campaign** | MCP-RC corrective campaign (residuals of the closed MCP-SERVER-X stage: R-P8-01, R-P7-01..06) |
| **Role** | `independent-quality-auditor` (IMPLEMENTER != AUDITOR != CERTIFIER). This session is a different session from the first auditor. Audit only: no product or test edit, no ticket, no `Approved:` line, no merge, no PR, no IIAC rebind, no certification |
| **Addendum to** | `MCP_SERVER_X_RC_INDEPENDENT_AUDIT_DOSSIER.md` (first audit, verdict PARTIAL at `93a089d3c`, findings AF-01..AF-16) |
| **Audited delta** | `93a089d3cca941592f1ad4f65435333c86da82e5..b0e19bf61b346e6941b2e190b703c6922e708b34` on branch `mcp-rc` (4 product files, 7 test/fixture/script files) |
| **Combined change** | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb..b0e19bf61b346e6941b2e190b703c6922e708b34` (23 product paths: the first audit's 23 paths, of which 4 changed again) |
| **Audit execution base** | `f1dde5a3e19924b82e902be18f11a21b64b822ba` (`mcp-rc` record head). `git diff --name-only b0e19bf61 f1dde5a3e` lists four documentation/evidence files only; nothing under `intergrax/`, `applications/` or `tests/` (verified). Auditor branch: `mcp-rc-reaudit` |
| **Date** | 2026-11-13 |
| **Verdict, delta** | **PASS_SCOPED** |
| **Verdict, combined change** | **PASS_SCOPED** (the first audit's PARTIAL conditions AF-01..AF-04, AF-10 are closed; open conditions are listed in section 8) |
| **Findings (new)** | 0 HIGH, 0 MEDIUM, 4 LOW, 3 INFO (7 total, `RA-01..RA-07`). Machine-readable: `iiac_evidence/PDC-X/MCP-RC/reaudit/iiac_audit_mcp_rc_reaudit_findings.json` |
| **Product change required** | **None.** Optional hardening: RA-02, RA-03, RA-04. One test required before certification is *recommended*: RA-01 |

PASS_SCOPED is an audit verdict on the scope below. It is **not** a certification, FRZ closure, stage closure or production-readiness claim, and does not close any P8 finding.

## 1. Scope and method

Scope: the corrective loop the implementer made after the first audit, judged against AF-01..AF-16, plus new defects in the new code. Treated every implementer claim (corrections record "Audit follow-up", ADR-MCP-RC-01, qualification package) as a claim.

Method:

1. Read the first dossier, its findings register and probe scripts; the corrections record, ADR and package; `OD-MCP-RC-01/02`.
2. Line-by-line review of `git diff 93a089d3c b0e19bf61 -- intergrax applications` (four files, 68 insertions, 35 deletions), and of every caller of `_bounded_observe_host_task` and `_await_host_or_deadline`.
3. Own adversarial probes (throw-away scripts, kept under `iiac_evidence/PDC-X/MCP-RC/reaudit/adversarial_scripts/`), partly re-using the first audit's scripts, run on the audited tree over real streamable HTTP where identity is concerned.
4. Re-ran the mandatory contract (one pytest process, `PYTHONPATH` = my checkout).
5. Re-ran the implementer's 17-mutant harness and added 16 mutants of my own for the new code plus one wide re-run, on throw-away worktrees outside the repository (`<parent>/_wt/intergrax/_scratch/…`, removed afterwards).
6. Diffed every changed test, fixture and script file between the two SHAs for removed or loosened assertions.
7. Measured the server start time that the 15 s → 60 s fixture budget change is based on.

## 2. Own numbers

| Check | Result |
| --- | --- |
| Mandatory contract `MCP_SERVER_X_RC_SELECTION` (39 files, 481 tests collected) | **480 passed, 1 xfailed (strict), exit 0, 421.65 s** (same counts as the implementer; shared, loaded workstation). The xfail is `test_mcp_surface_opt_in.py::test_scaffold_defaults_mcp_to_opt_in[product]` (TRANSFERRED). No start-budget failure occurred. Log: `reaudit/logs/reaudit_rc_contract_at_b0e19bf61.log` |
| Implementer mutation harness, re-run (M1..M17) | **17 of 17 killed**, every kill by an assertion in the targeted tests (no import error) |
| Own mutants R1..R16 | **14 killed, 2 survived**: R8 (real gap, RA-01) and R16 (equivalent control, see below) |
| R8 re-run against all 7 catalog suites + RC file (134 tests, `PYTHONPATH` = scratch worktree) | still **survived** |
| Own probes | all identity, error-text, deadline and concurrency probes hold; four observations (RA-02, RA-03, RA-04 and a probe limitation, section 4) |
| Mandatory test files outside MCP changed by the delta | none |

Note on a measurement error I made and corrected: the first R8 wide run reported *killed*, but the kill was an import failure in a subprocess fixture (without `PYTHONPATH` the fixture imports the project from the editable install, not from the scratch worktree). I discarded it and re-ran with `PYTHONPATH` pointing at the scratch checkout; the survivor is the valid result. Kills of M1..M17 and R1..R16 were checked for import errors: none.

Own mutants (all one-line, on a detached worktree; test target = `test_mcp_rc_audit_corrections.py` unless noted):

| Mutant | Invariant | Result |
| --- | --- | --- |
| R1 sanitizer passes any non-`ToolError` through (handler narrowed to `ZeroDivisionError`) | AF-01 | killed |
| R2 custom-tool gate bypassed for every tool (`if True`) | AF-03 | killed |
| R3 `outcome_unknown` reported as a COMPLETED `TaskResult` (verified by hand; killed by `test_non_cooperative_host_is_never_reported_as_success_and_stays_supervised`) | AF-02 | killed |
| R4 `public_custom_tool_names` test inverted (everything public by default) | AF-03 | killed |
| R5 `ToolError` no longer passes through the sanitizer | wire codes | killed |
| R6 host ending cancelled after the bound re-raises `CancelledError` | AF-02 | killed |
| R7 custom gate evaluated after the handler ran | AF-03 | killed |
| R8 `invoke_catalog_tool` without `sanitized_tool_errors` | AF-01 | **survived** (RA-01) |
| R9 `run_agent` unsanitized | AF-01 | killed |
| R10 `list_agents` unsanitized | AF-01 | killed |
| R11 custom handler unsanitized | AF-01 | killed |
| R12 custom gate evaluated in `LOCAL_DEV_EXPLICIT` instead of the server mode | AF-03 | killed |
| R13 failed host after the bound labelled `cancelled` | AF-02 | killed |
| R14 `describe_catalog_tool` unsanitized | AF-01/AF-10 | killed |
| R15 `list_catalog_tools` unsanitized | AF-01/AF-10 | killed |
| R16 observe-swallow widened from `Exception` to `BaseException` (control) | n/a | survived — equivalent for every reachable input: the preceding `except (TimeoutError, CancelledError)` already swallows the only `BaseException` the awaited task can produce here |

## 3. Closure judgement for the five required findings

| ID | Fix claimed | Independent judgement |
| --- | --- | --- |
| **AF-01** raw host exception text on the wire | `sanitized_tool_errors` around `run_agent`, `list_agents`, `list_catalog_tools`, `describe_catalog_tool`, `invoke_catalog_tool` and custom tools | **CLOSED**. Re-run of the first audit's `adv_runagent_leak`: `MCP_INTERNAL_ERROR: tool execution failed (RuntimeError)` / `(ValueError)`. Own probe over real HTTP with a verified principal: no secret, no path fragment. `ExceptionGroup` and a class named after a tenant: class name only. FastMCP echoes `str(exc)` for any exception, so the context manager is the right choke point. Mutants R1, R9, R10, R11, R14, R15 killed. Residuals: RA-01 (invoke wiring unpinned by any test), RA-02 (a `ToolError` authored by host code passes through) |
| **AF-02** deadline path when the host ends cancelled or failed | `_await_host_or_deadline` maps `CancelledError` → `outcome=cancelled`, other exception → `outcome=failed`; `_bounded_observe_host_task` no longer re-raises the host's exception | **CLOSED**. Re-run of `adv_deadline_cancelled`: both cases return `MCP_DEADLINE_EXCEEDED … outcome=cancelled|failed` in 0.38 s, no raw text. Review: `host_coro.result()` is synchronous, so the new `except asyncio.CancelledError` can only catch the *host task's* state, never the waiter's own cancellation (no masking of an outer cancel). The `from None` / `from exc` chaining keeps the cause server-side. Mutants R3, R6, R13, M16, M17 killed. Residual limits: RA-04 (pre-existing `TimeoutError` mislabel in the `LOCAL_DEV` path), RA-05 (label `cancelled` is used for any non-COMPLETED terminal `TaskResult`) |
| **AF-03** custom tools outside the principal gate | default gate in `_run_custom_host_tool_bounded` evaluated before the handler; per-tool opt-out `public_custom_tool_names` | **CLOSED in code.** Over real streamable HTTP: no credentials / wrong key / empty key → refused, handler never ran (call counter unchanged); valid key → runs; the research pipeline tool (double identity resolution: gate, then `execute_mcp_agent_task_with_resolved_identity`) behaves correctly: no auth → `MCP_PRINCIPAL_REQUIRED`, zero executions; key → one execution; key + forged tenant → `MCP_TENANT_ASSERTION_CONFLICT`, zero executions. The gate runs before the handler (R7 killed), uses the server's identity mode (R12 killed), is default-on (R2, R4 killed). The opt-out is by exact tool name, so a typo leaves the tool gated (fail closed). Compat impact: any custom host tool that was intentionally public now needs the opt-out. In this repository the only custom tool is `run_research_pipeline`, which was already identity-gated, so no in-repo host changes behaviour. Residual: RA-03 (the opt-out is not reachable through the Tier-3 facade `compose_application_mcp_server`) |
| **AF-04** five surviving auditor mutants | tests in `test_mcp_rc_audit_corrections.py`, mutants M9..M13 | **CLOSED**. The five original mutants (cross-loop bound call sites A3/A3b, discovery gates A12/A13, downgrade guard A11) correspond to M9, M10, M11, M12, M13, all killed by the named tests in my re-run. Test review: the cross-loop tests use a fake ingress and a patched binding match, which pins the wiring but not the real ingress (stated scope limit). One assertion in `test_unserializable_custom_tool_results_are_typed_not_raw` is tautological (`startswith(...) or "MCP_" in …`) but the following assertion is specific, so it does not weaken the test |
| **AF-10** deep or circular results | covered by the sanitizer | **CLOSED**. Custom tool results of depth 5000 and circular: typed. Own probe: a `run_agent` result with depth-5000 metadata → `MCP_INTERNAL_ERROR … (RecursionError)`, event loop live. The sanitizer's `_LOG.exception` with a `RecursionError` did not fail |

### New design points (checked for new defects)

**`sanitized_tool_errors`.** Passes `ToolError`, `asyncio.CancelledError` and `McpConcurrentToolCallLimitExceeded` unchanged; every other `Exception` becomes `MCP_INTERNAL_ERROR` with the class name only and a server-side log record. `BaseException` subclasses (`KeyboardInterrupt`, `SystemExit`, bare `CancelledError`) are not caught, which is correct. Server-side cancellation is not mislabelled: direct probe (`reaudit_probe_cancel.py`) — the over-cap call is refused with the declared code, the cancelled first call raises `CancelledError`, and the third call is admitted (slot released). The `McpConcurrentToolCallLimitExceeded` pass-through is unreachable in this code (the admission context is entered *outside* the sanitizer), so it is harmless dead weight (RA-06). What it could mask: a `ToolError` raised by host code keeps its host-authored text (RA-02). What it could mislabel: an exception class name in the message is attacker-influenced only through host code, not through a client.

**Custom-tool gate and compat.** See AF-03. The gate runs inside the admission slot, so an unauthenticated caller occupies a slot only for the duration of a synchronous refusal. Double identity resolution for tools that resolve identity themselves costs two verifications per call (provider-backed hosts: two provider round trips); correctness verified, performance only (RA-06).

**`_bounded_observe_host_task` now swallowing host exceptions.** `asyncio.wait_for(asyncio.shield(host_coro))` can raise a host exception only if the host task failed; the shield propagates it. All seven call sites were checked:

| Call site | Host outcome after the observe |
| --- | --- |
| `_terminate_gated_host_task_after_refused_admission` | retrieved by `_retrieve_host_task_outcome` (suppressing); the host never passed the start gate, so no business failure can exist |
| `_await_host_or_deadline` (drain) | retrieved explicitly by `host_coro.result()` and surfaced as `outcome=failed|cancelled` |
| `_finalize_transport_waiter_ownership` (three sites) | retrieved by `_retrieve_host_task_outcome`; these run inside `finally`, where a re-raise used to *replace* the in-flight deadline exception — the race the implementer fixed |
| `_await_host_task_bounded_after_transport_waiter_lost` loop | `done()` → `break` → finalize → retrieve |
| tail of `run_host_execution_with_mcp_transport_cancel` | `suppress(...)`; after a lost transport waiter a host failure now ends as `CancelledError` instead of the raw host exception (the client is gone; acceptable and consistent with the existing final branch) |

No path hides a failure that must surface: every failure of the host is either returned as a coded error to a live client or logged/retrieved for a client that is gone. Pre-existing and unchanged: the first `except (TimeoutError, CancelledError)` clause swallows a cancellation of the *calling* task during the observe (RA-05).

**`_await_host_or_deadline` outcomes.** `cancelled`, `failed` and `outcome_unknown` all raise `McpHostExecutionDeadlineExceeded`; only a COMPLETED `TaskResult` is returned. A host that completes at the bound is returned as success (correct). Mutants R3 (unknown as success) and A4 (first audit) are killed.

**Catalog discovery, event loop and threads.** Unchanged by the delta except for the wrapper: principal resolution stays on the loop thread (the request context is available there), the registry read stays in `asyncio.to_thread`. Probe: `list_catalog_tools` blocked 1.5 s in the registry while `list_agents` answered in 0.00 s. Over real streamable HTTP in the default mode: both discovery tools refused without a principal, allowed with a key.

**60 s fixture start budget.** Both subprocess fixtures still fail closed: on expiry they `kill()` the process and raise `TimeoutError("…startup_timeout")`. The TL11 never-ready test overrides the budget to 3 s with `monkeypatch`, so it still exercises the timeout path quickly. The L3-R2 never-ready test (`test_mcp_scenario_14_no_indefinite_wait_on_failed_startup`) does not override it and now waits the full 60 s (+45 s per run; no assertion was removed). Root cause: I measured server start (port line printed) on the same shared workstation: L3-R1 server 1.9–3.5 s, TL11 catalog server 6.5–8.2 s (import-heavy), 3 samples each, outside the contract run. The implementer's 8–21 s under load was **not reproduced** by me but is plausible, since TL11 alone is ~7–8 s idle and the machine is shared; the first audit's contract run passed with the 15 s budget. Judgement: a wait budget only, root-caused to import weight, fail-closed behaviour kept; it can mask up to a ~4x slow-start regression (RA-07, INFO). No evidence that it masks a regression.

## 4. Adversarial results (summary)

| Probe | Result |
| --- | --- |
| P1 `run_agent` host exception, real HTTP, valid key | `MCP_INTERNAL_ERROR … (RuntimeError)`, no secret |
| P3 custom tool, default mode, real HTTP: no auth / valid key / bad or empty key | refused (`MCP_PRINCIPAL_REQUIRED` or HTTP 401) with zero handler runs / runs for the key |
| P3d `compose_application_mcp_server` signature | **no `public_custom_tool_names` parameter** → RA-03 |
| P4 custom tool raising `ToolError("UPSTREAM-SECRET …")` | text passes through unchanged → RA-02. `ExceptionGroup` and a tenant-named exception class: sanitized |
| P5 `LOCAL_DEV_EXPLICIT`, `max_call_seconds=30`, host raises `TimeoutError` at once | `MCP_DEADLINE_EXCEEDED … outcome=cancelled` after 0.00 s → RA-04 (pre-existing mapping, wrong label, no success claim) |
| P6 blocking registry vs. loop | loop live |
| P7 deep `run_agent` metadata | typed error |
| P8 cap 1, server-side cancel | refusal code, `CancelledError` unchanged, slot freed (`reaudit_probe_cancel.py`). The in-memory `Client` cancel in `reaudit_probes.py` P8b does not reach the server task; recorded as NOTE, a probe limitation, not a product defect |
| First audit `adv_identity` re-run | all identity cases PASS (H1..H4); the two H1 catalog `FAIL` lines are the first audit's known script artefact (no tool registry in that script) and are covered by `reaudit_probe_catalog_http.py` |
| First audit `adv_runagent_leak`, `adv_deadline_cancelled` re-run | AF-01 / AF-02 symptoms gone |

## 5. New findings

Severity scale as in the first dossier (HIGH = authority bypass/data exposure by default; MEDIUM = claimed-closed requirement not met/evidenced; LOW = narrow-scope gap or hardening; INFO = observation).

| ID | Sev | Finding | Evidence | Recommended disposition |
| --- | --- | --- | --- | --- |
| RA-01 | LOW | The `sanitized_tool_errors` wiring of `invoke_catalog_tool` is not pinned by any test | mutant R8 survives the RC file, the identity/bounds/deadline suites and all 7 catalog suites (134 tests). Code reviewed correct (`mcp_catalog_tools.py`: sanitizer inside the admission context). Reachable raw paths exist: the bridge's cross-loop `submit_catalog_child_work` and `child_port.execute` may raise a non-`ToolError` | Add one test (an invoker or ingress raising a secret-bearing exception → `MCP_INTERNAL_ERROR`) before certification. No product change |
| RA-02 | LOW | `sanitized_tool_errors` passes every `ToolError` unchanged, including ones authored by host code or relayed from an upstream MCP client | probe P4: `ToolError("UPSTREAM-SECRET tenant=victim token=abc")` reaches the client verbatim. ADR decision 6 says raw exception text "never reaches the wire" | Either document the exception to the statement ("a `ToolError` raised by host code is the host's own wire message") or pass through only messages that start with a declared `MCP_` code and sanitize the rest |
| RA-03 | LOW | `public_custom_tool_names` is reachable only through `McpServerCompositionDeps` / `compose_mcp_server_from_definition`, not through the Tier-3 facade `compose_application_mcp_server`, the documented entry for hosts with custom tools | probe P3d (signature); `mcp_application_server.py` builds the deps without the field | Fail-closed direction, so no security issue. Document, or thread the parameter later; record in the migration note that a public custom tool needs the engine entry |
| RA-04 | LOW | A `TimeoutError` raised by host code is reported as `MCP_DEADLINE_EXCEEDED … outcome=cancelled` in the `LOCAL_DEV` path and in the catalog bridge | probe P5; `mcp_nexus_server.py` `except TimeoutError` around `asyncio.wait_for`, bridge `except TimeoutError` (both pre-existing, unchanged by the delta, same area as AF-07). No success is ever claimed | Optional hardening: distinguish the bound's own expiry (elapsed time or a dedicated sentinel). Accept as documented imprecision |
| RA-05 | INFO | Deadline outcome `cancelled` is used for any non-COMPLETED terminal `TaskResult` (including FAILED); `_bounded_observe_host_task` swallows a cancellation of its *calling* task (pre-existing clause) | code review of `_await_host_or_deadline` and `_bounded_observe_host_task` | Accept; mention in the ADR if the label is ever consumed programmatically |
| RA-06 | INFO | Cosmetics and cost: orphan attribute docstrings in `McpServerCompositionDeps` (the two docstrings of `identity_mode` / `max_call_seconds` now follow `public_custom_tool_names`); unreachable `McpConcurrentToolCallLimitExceeded` pass-through; two identity verifications per research-tool call | review; research-tool probe | Accept; tidy when the file is next touched |
| RA-07 | INFO | Start budget 15 s → 60 s: the L3-R2 never-ready test now takes 60 s; the budget can mask a slow-start regression up to ~4x | fixture diff; `startup_timing.results.txt` (L3-R1 1.9–3.5 s, TL11 6.5–8.2 s) | Accept; optionally give the L3-R2 never-ready test the same 3 s override as the TL11 test |

## 6. Judgement of the implementer's classifications for AF-05..AF-16

Legend: ACCEPT, ACCEPT_SCOPED (acceptable with the stated scope), EVIDENCE (acceptable only with the named evidence), OPEN.

| ID | Implementer response | Judgement |
| --- | --- | --- |
| AF-05 real factory composition unverified | ACCEPTED_SCOPED; owner "strict-product test-support/scaffold owner" | **ACCEPT_SCOPED.** The statement "real product-factory composition unverified for MCP opt-in" is honest. **Condition, open:** the owner is a role, not a named owner or ticket; the certifier must see a named owner/ticket |
| AF-06 wall time is bound + drain | Documented (record, ADR decision 5) | **ACCEPT.** ADR says "plus the host drain"; the 30 s default is not stated in the ADR and no test asserts the default drain (nice to have) |
| AF-07 "only a governed-cancel trigger" wording | ADR amended | **ACCEPT.** ADR decision 5 now names the two non-governed mechanisms. `OD-MCP-RC-02` is an operator record and rightly unchanged. See RA-04 |
| AF-08 loopback guard trusts `request.client.host` | Documented in ADR consequences | **ACCEPT** (documentation only, as recommended) |
| AF-09 `execute_mcp_agent_task` public, `unverified` label is evidence | Documented in ADR consequences | **ACCEPT** |
| AF-10 deep/circular | Fixed with AF-01 | **CLOSED** (section 3) |
| AF-11 reference server defaults to `LOCAL_DEV`; gate test name-based | Accepted as declared | **ACCEPT** (fixtures only) |
| AF-12 factories expose no `identity_mode` | Accepted as declared | **ACCEPT** |
| AF-13 `max_call_seconds` not threaded through facades | Accepted; composition parameter | **ACCEPT** (documented in the corrections record; RA-03 is the sibling for the custom-tool opt-out) |
| AF-14 `initialize`/`tools/list` unauthenticated on API-key hosts | Accepted as pre-existing protocol behaviour | **ACCEPT** (recorded only in the corrections record, not in the ADR; acceptable) |
| AF-15 domain errors as normal results | Accepted | **ACCEPT** |
| AF-16 strict xfail `[product]` is a non-MCP defect | Accepted TRANSFERRED; owner/ticket "must be named by the operator before certification" | **ACCEPT_SCOPED, open:** still no named owner or ticket in the repository (same condition as AF-05) |

## 7. Test-weakening check for the delta

`git diff 93a089d3c b0e19bf61` over the six changed test/fixture/selection files and the mutation script: **no assertion removed or loosened.** Changes: `rc_selection.py` +1 line (the new suite is in the contract); `reference_app/build.py` +2 lines (the `public_custom_tool_names` pass-through, default empty); two fixtures: the start budget plus line re-wrapping, no logic change, fail-closed path unchanged; `test_mcp_catalog_e2e_hung_server_qualification.py`: a `monkeypatch` of the budget to 3 s (the test still expects `TimeoutError`); `mcp_rc_mutation.py` +9 mutants; `test_mcp_rc_audit_corrections.py` is new (405 lines; no removed lines). One observation on the new file: the first assertion of the unserializable-results test is tautological (covered by the specific assertion that follows).

## 8. Open conditions, scope limits and what I did NOT cover

Open conditions for a certifier (none requires a product change):

1. Name an owner/ticket for AF-05 and AF-16 (restoration of a strict product vehicle and the `[product]` xfail).
2. Recommended: a test killing R8 (RA-01) — evidence only.
3. Any later product or test change voids this audit for the changed paths.

Not covered (same list as the first audit, narrowed): certification, FRZ/closure, CI contexts (ENT-AUDIT-X, EVOL-X) on the PR head, the PDC-P7-00 rebind and `IIAC_POLICY_ACTIVATION_RECORD.json`; G-DOC-01..12 (not audited); the shared harness modules (G-ID-05); a production `HostTaskExecutionPort` under cancellation after the bound (host doubles only, so the production reachability of AF-02 remains unverified but is now handled for both terminal shapes); the real cross-loop ingress and real catalog child execution under adversarial payloads (fake ingress only); the real product factory composition (AF-05); stdio/SSE absence; performance beyond the probes; Docker or non-Windows environments. My runs: Windows, one pytest process at a time, environment unchanged, no Docker, no vendors, no production configuration (OD-MCP-RC-02 bounded diagnostic authorization).

## 9. Verdicts

- **Delta `93a089d3c..b0e19bf61`: PASS_SCOPED.** AF-01, AF-02, AF-03, AF-04 and AF-10 are closed; the new code has no MEDIUM or HIGH defect; four LOW and three INFO findings are recorded.
- **Combined change `6a6279a49..b0e19bf61`: PASS_SCOPED.** The first audit's PARTIAL is superseded for AF-01..AF-04 and AF-10; the remaining scope limits (section 8) and the two ownership conditions are the reason it is not PASS.

In the IIAC session manifest the schema value `CONDITIONAL` is used, because the schema has no `PASS_SCOPED`; the manifest limitations spell out the scope. No certification is claimed.

## 10. Evidence index

All under `docs/project/maintainers/qualification/iiac_evidence/PDC-X/MCP-RC/reaudit/`: `audit_scope.json`, `iiac_audit_session_manifest.PDC-X.mcp_rc_reaudit.json`, `iiac_audit_mcp_rc_reaudit_findings.json`, `test_matrix.txt`, `EVIDENCE_MANIFEST.txt`, `logs/`, `mutation/`, `adversarial_scripts/`.
