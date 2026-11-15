# MCP-SERVER-X RC — independent audit dossier (G6)

| Field | Value |
| --- | --- |
| **Campaign** | MCP-RC corrective campaign (residuals of the closed MCP-SERVER-X stage: R-P8-01, R-P7-01..06) |
| **Role** | `independent-quality-auditor` (IMPLEMENTER != AUDITOR != CERTIFIER). Audit only; no certification, no merge, no signature, no IIAC rebind |
| **Audited product delta** | `6a6279a49a9aa5a33936a1c6a577c7194b4bcebb..93a089d3cca941592f1ad4f65435333c86da82e5` (`IMPLEMENTATION_SHA = TEST_EXECUTION_SHA`, 23 files under `intergrax/` and `applications/`) |
| **Audit execution base** | `f3078b293ecee402583b3d6ad9ef5aaeff0b2d1d` (branch `mcp-rc`, documentation/evidence only after `93a089d3c`; verified: `git diff --name-only 93a089d3c f3078b293` touches no path under `intergrax/`, `applications/`, `tests/`). Auditor branch: `mcp-rc-audit` |
| **Date** | 2026-11-13 |
| **Verdict** | **PARTIAL** (scope below) |
| **Findings** | 0 HIGH, 5 MEDIUM, 9 LOW, 2 INFO (16 total). Machine-readable: `iiac_evidence/PDC-X/MCP-RC/iiac_audit_mcp_rc_findings.json` |
| **Product change required** | AF-01, AF-02, AF-03 (AF-03 alternatively a contract amendment). AF-04 and AF-05 need tests/evidence, not product code |

## 1. Verdict and scope

**PARTIAL.** The core corrective claim holds: with the fail-closed default `HARNESS_BOUND`, no untrusted client assertion of tenant or user established authority on any entry point I exercised (`run_agent`, `list_agents`, `list_catalog_tools`, `describe_catalog_tool`, the research pipeline tool, eight facade builders, `couple_fastapi_with_mcp`, auth wired after coupling, real streamable HTTP). `LOCAL_DEV_EXPLICIT` is not selectable from the environment, refuses a real non-loopback peer, never downgrades configured auth and is refused for catalog invoke. The mandatory contract passes on the pinned SHA.

The verdict is PARTIAL, not PASS_SCOPED, because three corrections recorded as `CLOSED_IN_CODE` are not closed as claimed and need product changes:

1. **AF-01** `run_agent` echoes raw host exception text (contract matrix and G-RB-04 say it never does).
2. **AF-02** the call-bound path does not return `MCP_DEADLINE_EXCEEDED` when the host task ends cancelled or failed after the bound (request aborted / raw text).
3. **AF-03** custom host tools are outside the fail-closed principal gate (G-ID-06 says "every tool").

Plus two evidence gaps that the certifier must see closed or accepted: surviving mutants on claimed-closed invariants (AF-04) and no real-factory composition test (AF-05).

The PARTIAL verdict covers the product delta at `93a089d3c` only. It is **not** a certification, not a closure of any P8 finding, and gives no production-readiness claim. In the IIAC manifest the closest schema value is `CONDITIONAL` (the schema has no `PARTIAL`).

## 2. Method

1. Read ADR-MCP-RC-01, the contract matrix v1.1, the corrections record, the qualification package, the EVOL/ENT delta record, `OD-MCP-RC-01/02`, tickets `T-IGX-038-A..F` and the plan sections; every disposition was treated as a claim.
2. Line-by-line review of the 23-file product diff (identity, bounds, deadline/cancel/admission, invocation id, compatibility, authorities, layer boundaries). `check_no_upward_application_imports.py`: no MCP file listed; no `agents`/`applications` import added under `intergrax/`.
3. Re-ran the mandatory contract (one pytest process, `PYTHONPATH` = my checkout).
4. Re-ran the implementer's mutation harness and added 14 mutants of my own on throw-away worktrees (never on the audited branch).
5. Wrote throw-away adversarial scripts (kept as evidence under `adversarial_scripts/`, not in the product or test trees).
6. Diffed every changed test file against `6a6279a49` (removed lines reviewed) and re-ran the removed opt-in tests with `--runxfail` to check the implementer's root-cause claims.

## 3. Own numbers

| Check | Result |
| --- | --- |
| Pin | `HEAD` of the worktree cut from `mcp-rc` = `f3078b293`; contains `93a089d3c`; docs-only delta confirmed |
| Mandatory contract (`MCP_SERVER_X_RC_SELECTION`, 38 files) | **464 passed, 1 xfailed (strict), exit 0, 455.11 s** — same counts as the implementer (386.6 s). Log: `logs/auditor_rc_contract_at_93a089d3c.log` |
| Implementer mutation harness (re-run) | **8 of 8 killed** |
| Auditor mutants A1-A13 (14 incl. A3b) | **9 killed, 5 survived** (A3, A3b, A11, A12, A13). A3/A3b re-run with the cross-loop suites added: still survive |
| Adversarial scripts | identity (real HTTP): all identity cases hold; findings AF-03, AF-14; bounds/concurrency hold except AF-10; deadline: AF-02, AF-06; run_agent error echo: AF-01 |
| Removed opt-in tests re-run at HEAD (`--runxfail`) | 5 failed / 10 passed, causes `LLMProviderNotConfiguredError` and `StrictAgentContractAuthorityError` (not MCP) |

Auditor mutant detail: A1 loopback guard removed (killed), A2 catalog invoke registry unfiltered (killed), A3/A3b cross-loop payload bound removed at call sites (**survived**), A4 non-completed outcome returned as success after the bound (killed), A5 `list_agents` gate removed (killed), A6 request-time auth resolver disabled (killed), A7 custom tool exception text echoed (killed), A8 unknown-tool echo uncapped (killed), A9 constant invocation id (killed), A10 `unverified` label dropped (killed), A11 LOCAL_DEV downgrade guard for an explicit `harness_auth` argument (**survived**; unreachable through production callers, which always pass `None`), A12/A13 `list_catalog_tools`/`describe_catalog_tool` gate removed (**survived**).

## 4. Adversarial results (summary)

| Area | Result |
| --- | --- |
| Forged tenant/user, no auth, real streamable HTTP | `MCP_PRINCIPAL_REQUIRED` for `run_agent`, `list_agents`, `list_catalog_tools`, `describe_catalog_tool`; zero executions |
| Auth wired **after** coupling | no/wrong/empty key refused; key + forged tenant/user, padded or case-varied tenant -> `MCP_TENANT_ASSERTION_CONFLICT`; matching or absent tenant binds the principal tenant (`T1`) |
| `LOCAL_DEV_EXPLICIT` | loopback admitted; real LAN IP peer (192.168.0.51) refused for `run_agent` and `list_agents`, also with `X-Forwarded-For`/`Forwarded` spoofing; configured auth never downgraded; forged tenant with a valid key still conflicts |
| Custom host tool in default mode, no auth | **reachable** (AF-03) |
| Oversize/nested payloads | 3 MB string, 2M-int list, 300k objects -> `MCP_RESULT_BOUND_EXCEEDED` in <= 0.22 s, event loop live; depth-5000 / circular -> uncoded error (AF-10); custom tool exception text not echoed |
| Concurrency | cap 2, 8 concurrent calls: 2 ok, 6 `MCP_CONCURRENT_CALL_LIMIT_EXCEEDED`; slots free afterwards, also after client-side cancellation |
| Deadline, non-cooperative host | `MCP_DEADLINE_EXCEEDED outcome=outcome_unknown`, never success, work stays under the supervisor; wall time 31.2 s for a 1.0 s bound (AF-06) |
| Deadline, host ends cancelled / fails | **request aborted ("Connection closed") / raw text** (AF-02, AF-01) |
| `run_agent` host exception | raw text echoed (AF-01) |
| `initialize` / `tools/list` without credentials on an API-key host | allowed (AF-14) |

## 5. Findings register

Severity scale: HIGH = authority bypass or data exposure in the default configuration; MEDIUM = a claimed-closed requirement not met or not evidenced; LOW = documentation, hardening or narrow-scope gap; INFO = declared behaviour worth recording.

| ID | Sev | Finding | Evidence | Recommended disposition |
| --- | --- | --- | --- | --- |
| AF-01 | MEDIUM | `run_agent` (and other non-custom paths) echo raw host exception text | `adv_runagent_leak`: `RuntimeError("SECRET-HOST-FAILURE-hunter2 /srv/app/db.py:42")` returned verbatim in `Error calling tool 'run_agent'`; only custom tools are wrapped (`mcp_server_engine._run_custom_host_tool_bounded`) | **Product change**: map non-`ToolError` exceptions to `MCP_INTERNAL_ERROR` (class name only) for `run_agent`, `list_agents`, discovery; add test. Reject CLOSED for G-RB-04 |
| AF-02 | MEDIUM | After the call bound, a host task ending cancelled/failed gives no `MCP_DEADLINE_EXCEEDED` | `adv_deadline_cancelled`: CancelledError host -> `McpError Connection closed`; raising host -> raw text. `_await_host_or_deadline` calls `host_coro.result()` unguarded | **Product change** + tests. Production reachability plausible (hosts normally return `TaskResult(CANCELLED)`), not demonstrated |
| AF-03 | MEDIUM | Custom host tools are not covered by the fail-closed principal gate | `adv_identity` H4: custom tool callable over HTTP without credentials in `HARNESS_BOUND` | **Product change** (default gate with explicit opt-out) or ADR/record amendment; G-ID-06 PARTIAL until then |
| AF-04 | MEDIUM | 5 surviving auditor mutants: cross-loop bound wiring (A3, A3b), catalog discovery default-mode gates (A12, A13), explicit-auth downgrade guard (A11) | `mutation/auditor_mutants_*.json` | Add killing tests (no product change); needed for G-RB-02 and the discovery part of G-ID-06 |
| AF-05 | MEDIUM | No real factory composition test; opt-in vehicle replaced by lab app | removed tests re-run: failures are LLM-provider and STRICT agent-contract errors (implementer root cause confirmed); `logs/auditor_old_opt_in_vehicle_probe.log` | Accept as ACCEPTED_SCOPED, state "real product factory composition unverified", track a strict-vehicle restoration owner |
| AF-06 | LOW | Wall-time bound is `max_call_seconds` + 30 s drain (31.2 s for 1.0 s); deadline tests shrink the drain to 0.25 s | `adv_deadline.results.txt` | Document effective bound; add a default-drain assertion |
| AF-07 | LOW | "Only a governed-cancel trigger" is not true for LOCAL_DEV `run_agent` and catalog invoke (`asyncio.wait_for`); no canonical deadline is created | code: `execute_mcp_agent_task`, bridge `invoke_timeout_s` | Amend ADR/OD wording; accept |
| AF-08 | LOW | Loopback guard trusts `request.client.host` (same-host proxy defeats it; `testclient` literal accepted) | real-network test refused LAN peer | Document; do not use LOCAL_DEV behind a proxy |
| AF-09 | LOW | `execute_mcp_agent_task` stays public with caller-asserted tenant/user; `unverified` label is not enforced anywhere | exports in `mcp_developer_api` | Document/deprecate |
| AF-10 | LOW | Deep-nesting/circular results give uncoded errors | `adv_bounds.results.txt` | Map with AF-01 |
| AF-11 | LOW | Reference server defaults to LOCAL_DEV (12 test/fixture files opt in); second-authority gate is name-based | `reference_app/build.py`, gate test | Accept; behavioural gate optional |
| AF-12 | LOW | Factories expose no `identity_mode`; five hosts without auth wiring are inert for MCP | factory review | Accept as declared; add migration note later |
| AF-13 | LOW | `max_call_seconds` not reachable via facades; research tool ignores it | facade review | Document/thread |
| AF-14 | LOW | `initialize`/`tools/list` unauthenticated on API-key hosts | `adv_probe4` | Document (pre-existing) |
| AF-15 | INFO | Domain errors returned as normal results (isError=false) | matrix, bridge | Accept (declared) |
| AF-16 | INFO | Strict xfail `[product]` is a genuine non-MCP defect; G-DOC not audited | `--runxfail` reproduction | Accept TRANSFERRED with named owner/ticket |

## 6. Judgement of the implementer's classifications

Legend: **ACCEPT**, **REJECT** (not closed as claimed), **EVIDENCE** (acceptable only with the named evidence), **N/A** (not audited).

| Claim | Judgement |
| --- | --- |
| G-ID-01 / R-P8-01 CLOSED_IN_CODE | **ACCEPT**: default fail-closed; entry points hold under real HTTP; M1, M8, A1, A5, A6 killed |
| G-ID-02, G-ID-09 (auth after coupling) | **ACCEPT** (synthetic app, real HTTP); real factories **EVIDENCE** (AF-05) |
| G-ID-03 (8 facades) | **ACCEPT** for the builders; factories see AF-05/AF-12 |
| G-ID-04 (research tool) | **ACCEPT** |
| G-ID-05 TRANSFERRED/SCOPED | **ACCEPT** as owner-assigned; shared harness module not audited; MCP gate fails closed regardless |
| G-ID-06 / R-P7-06 "gated for every tool" | **REJECT** as stated (AF-03); discovery gates **EVIDENCE** (AF-04) |
| G-ID-07 (filtered registry for invoke) | **ACCEPT** (A2 killed) |
| G-ERR-01 / R-P7-05 | **PARTIAL**: three declared codes are raised, but not for run_agent host failures (AF-01) |
| G-ERR-02, G-ERR-03 | **ACCEPT** |
| G-RB-01 / R-P7-03, G-RB-03 | **ACCEPT** (adversarial sizes; M3 killed); AF-10 low |
| G-RB-02 (cross-loop) | **EVIDENCE**: code correct on review, wiring unpinned (A3/A3b survive) |
| G-RB-04 | **REJECT** (AF-01) |
| G-DL-01 / R-P7-04 | **PARTIAL**: governed-cancel path works (M4, M5, A4 killed; non-cooperative host never success); AF-02, AF-06, AF-07 |
| G-DL-02, G-DL-03, G-DL-04 ACCEPTED_SCOPED | **ACCEPT** (limits real; AF-07 wording amendment) |
| G-DL-05 | **ACCEPT** (non-awaiting slot accounting; M6 killed; cap and cancel-safety confirmed adversarially) |
| G-DL-06 | **ACCEPT** (supervisor handoff observed in the deadline run) |
| G-TR-01 / R-P7-01 (`mcp_invocation_id`) | **ACCEPT**: `uuid4` hex with `mcpi-` prefix, no tenant/user data, not read by any authority, not client-suppliable via the tool surface; A9 killed |
| G-TQ-01 / R-P7-02 | **ACCEPT** the three repaired files and the invalid-fixture root causes; strict xfail `[product]` **ACCEPT as TRANSFERRED** (cause reproduced as non-MCP) conditional on a named owner/ticket; vehicle replacement **ACCEPT_SCOPED** with AF-05 |
| G-TQ-02 / F-P8-08 | **ACCEPT**; name-based gate noted (AF-11) |
| G-TQ-03, G-TQ-04 | **ACCEPT** (facade builders) |
| G-TQ-05 (mutation) | **ACCEPT** the reproduced 8/8; set was too narrow (AF-04) |
| G-PR-01..04 | **ACCEPT** as scoped |
| G-DOC-01..12 | **N/A** — not audited |

### Test weakening review

Reviewed all removed lines in the 23 changed test files against `6a6279a49`. No assertion was weakened to make a test pass: the P7 "recorded gap" test and the exactly-seven-codes test were inverted/widened as the contract requires; the self-referential ACC-14/gate/accept-either assertions were replaced by behavioural ones; `pytest.raises(Exception)` became an HTTP 401 assertion; the oversized-catalog tests now use real sizes instead of patching a 256-byte limit. The one real loss is coverage of the real product factory (AF-05). About 12 test/fixture files opt into `LOCAL_DEV_EXPLICIT`; default-mode behaviour is pinned for `run_agent`, `list_agents`, the research tool and the facades, but **not** for catalog discovery (A12/A13) — that is the hidden default-mode behaviour the implementer asked me to look for.

## 7. What I did NOT cover

- Certification, FRZ/closure claims, ENT/EVOL CI contexts on the PR head, the PDC-P7-00 rebind, any `IIAC_POLICY_ACTIVATION_RECORD.json` change.
- A production `HostTaskExecutionPort` under cancellation after the bound (AF-02 production reachability), the real catalog child execution path under timeout, and the real cross-loop ingress under adversarial payloads (only reviewed and unit-tested).
- Catalog invoke in the default mode with a real principal over HTTP (reviewed, not exercised by mutation); `MCP_EXECUTION_*` wire codes beyond mapping review.
- The shared harness modules (`apply_harness_auth_middleware` no-op, G-ID-05), the transport-detachment supervisor internals, FastMCP/uvicorn behaviour on other platforms, stdio/SSE absence (G-PR-02, taken from the implementer's test), and the G-DOC-01..12 documentation corrections.
- Performance, load beyond the 8-call burst, Docker or non-Windows environments.
- A re-run of CI; my runs are Windows, one pytest process at a time, environment unchanged, no Docker, no new vendors.

## 8. Evidence index

All under `docs/project/maintainers/qualification/iiac_evidence/PDC-X/MCP-RC/`: `audit_scope.json`, `iiac_audit_session_manifest.PDC-X.mcp_rc.json`, `iiac_audit_mcp_rc_findings.json`, `test_matrix.txt`, `EVIDENCE_MANIFEST.txt`, `logs/`, `mutation/`, `adversarial_scripts/`.

## 9. Recommended next steps (for the operator and the implementer; not performed here)

1. Product change for AF-01, AF-02 (and AF-10) and a decision on AF-03 (gate or amend), each with a test.
2. Tests killing A3, A3b, A11, A12, A13 (AF-04).
3. ADR/matrix amendments for AF-06, AF-07, AF-08, AF-09, AF-14.
4. Any product or test change after this audit voids it for the changed paths; re-audit the delta before certification.
