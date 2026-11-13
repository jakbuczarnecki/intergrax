# MCP-SERVER-X Residual Remediation Campaign ("MCP-RC") — master plan

Planning disposition: **PLAN_READY_FOR_OPERATOR_REVIEW** (with the stated W0 reproduction caveat in A.3 — not a blocker).
Mode: PLAN only. No product, test, record, ruleset or GitHub change was made. Nothing here is signed.

## Context

MCP-SERVER-X is `CLOSED — INDEPENDENTLY AUDITED — PUBLISHED (SCOPED)` (P8, `PASS_SCOPED`, certified at `919e34b8f`, terminal `c11184164`). The closure carries open scoped residuals R-P7-01..06 and R-P8-01, test debt R-P7-02, hardening findings F-P8-07/08 and process findings. The operator asked for one audited, owned, implementation-ready campaign that closes every *architecture-scoped* MCP deficiency before ARCH-FREEZE, without reopening the P8 history, duplicating PDC-X, weakening ACT-N11/IIAC, or broadening MCP v1.

Fixed constraints that shape the whole plan (all verified in repo records):
- `OD-MCP-P8-RESIDUAL-01` (recorded 2026-11-12): **R-P8-01 is bundled into the PDC-P7-00 ACT-N11 activation** as its own ticketed batch, one independent delta audit, one two-phase rebind at `INTEGRATED_BASELINE_SHA`. Not a separate rebind. OPEN-BP-01 is **DEFERRED** (no ruleset change).
- `OD-PDC-SEQ-01` (Option A): PDC product merges wait for MCP P8 publication — satisfied (P8 published).
- `OD-PDC-IIAC-SCOPE` (Option A): per-stage activation; roles `IMPLEMENTER != AUDITOR != CERTIFIER`; evidence under `iiac_evidence/PDC-X/...`.
- ACT-N11 (`scripts/governance/iiac_common.py:22`, `iiac_activation_gate.py`): any change under `intergrax/`, `agents/`, `applications/` after `audit_execution_sha` `2d3eceb6c` turns `governance-fast` red until a rebind. Today the delta is empty.
- FRZ counter stays **225** (PASS=108, OPEN=109); no promotion; no production-readiness claim; no new MCP protocol capability.

---

## A. Current-state audit

### A.1 Baseline (verified this session)
| Item | Value |
|---|---|
| `origin/development` (fetched) | `9d2f8bfc212e75305fb4ae0d1b779ce1c81566a8` (= local HEAD, 0 behind) |
| Local branch / tree | `mcp-p8-pin`, clean except 3 untracked `agent-env/plans/pasted-content-*.md` (not ours) |
| ACT-N11 pin (`IIAC_POLICY_ACTIVATION_RECORD.json`) | `audit_execution_sha=2d3eceb6cf7e…`, `IN_FORCE`, policy `IIAC-2026-11-05.1`; phase-2 pin `19c617365`; product delta `2d3eceb6c..HEAD` = empty |
| MCP closure chain | audit `eadb1c211` → cert/evidence `919e34b8f` → closure `c11184164`; later docs PRs #483, #485 (merged before checks finished = P8-PROC-03) |
| Open PRs | #488 (`pdc-x-d5`, B-D5) and #489 (`pdc-x-d2`, B-D2) → `development`, both touch `intergrax/` (would trigger ACT-N11). #475/#476/#478 were closed unmerged 2026-11-12T16:24Z (reason not recorded; OD-MCP-P8-RESIDUAL-01 still names them) |
| MCP branches | no R-P8-01/residual branch or ticket exists. Highest ticket series: `T-IGX-037` (SFX); PDC at `T-IGX-036-O`; MCP stage closed at `T-IGX-035-I` |
| Authorizations | OD-MCP-P7-P8-AUTH-01 standing authorizations **lapsed** at PR #480 merge → this campaign needs a new authorization decision (D-1) |

### A.2 Residuals and ownership (current)
R-P8-01: owner = PDC-P7-00 bundle (OD-MCP-P8-RESIDUAL-01), **planned only** — no ticket, WBS row, letter or plan. R-P7-01, 03, 04, 05, 06: **no owner, no ticket, no PDC-X coverage** (PDC grep: only `PDC-D6-R017` MCP exposure, `BLOCKED_EXACT_REASON` in external batch B-MCP; B-D6 plan explicitly excludes MCP). R-P7-02: untracked test debt, "orchestrator/operator". F-P8-07/08: accepted scoped, optional hardening tests. OD-MCP-P7-P8-AUTH-01 DR-2 allows LOW/MEDIUM acceptance only "with an owner" → these residuals currently violate that precondition; the campaign gives them one.

### A.3 Defects found by code read (HEAD `9d2f8bfc2`; **read-only, not yet executed**)
Reproduction was **not** performed in Plan mode: running tests is non-read-only here and the C10/C15 freeze requires separate operator authorization for diagnostic pytest. Therefore R-P7-02 per-test root causes below are *hypotheses from code + recorded logs*, and every "defect" is *code-read confirmed*, to be reproduced by a failing test first in W0/W2–W4 (red-first rule). This is built into the plan, not a blocker.

Beyond the documented residuals the audit found these (see B for IDs): the R-P8-01 surface is **wider than the OD describes** (5 facades have no harness auth at all; 3 wire identity after coupling; `run_research_pipeline` bypasses identity entirely; `list_*`/`describe` rely on middleware only); catalog invoke checks the **unfiltered** registry; `deps.transport_detachment_supervision` / `auto_wire_catalog_transport_supervision` are **dead assignments** (`mcp_server_engine.py:~286`); the cross-loop ingress returns an **unbounded** result and leaks `str(exc)`; the two bounded-error shapes disagree on `isError`; admission slot is released for still-running detached executions.

---

## B. Complete gap inventory (closed world)

Severity is the *proposed* campaign severity; FRZ family per section C.

### B.1 Identity / tenant / principal (Batch MRC-1)
| ID | Finding (code evidence) | Sev |
|---|---|---|
| G-ID-01 (R-P8-01) | `resolve_mcp_run_agent_identity` returns `asserted_tenant_id or default_tenant_id` as authoritative when no harness auth / no HTTP request / local-dev `HarnessAuthState()`; `request_identity=None` skips principal binding, cancel ctx and supervision (`mcp_harness_principal.py:132-153`, `mcp_nexus_server.py:98-111`) | HIGH (MEDIUM in P8) |
| G-ID-02 | `couple_fastapi_with_mcp` copies `harness_auth` to MCP sub-app only if present at couple time (`fastapi_mcp.py:66,78-85`); `lab`/`legal`/`research` factories call `wire_application_identity` **after** coupling (lab `factory.py:236 vs 248`, research `190 vs 201`, legal `249 vs 260`) → MCP sees no auth; key holder may assert any tenant | HIGH |
| G-ID-03 | `dispute_sim`, `governed_contractor`, `intergrax_assistant`, `local_workspace`, `poc_template` never wire harness auth on `/mcp` (`apply_harness_auth_middleware` returns early when unset, `harness_auth.py:357-358`) | HIGH |
| G-ID-04 | `research_application/mcp/server.py:30-43` `run_research_pipeline` takes client `tenant_id`/`user_id`, never calls `resolve_mcp_run_agent_identity` | HIGH |
| G-ID-05 | `apply_harness_auth_middleware` silently attaches nothing when `require_api_key` set without key/provider (`harness_auth.py:363-366`) — violates "missing mandatory config fails closed at composition" | MEDIUM |
| G-ID-06 (R-P7-06) | `harness_auth=None` hard-coded at both call sites (engine:254, bridge:311) = "resolve lazily from `app.state`"; meaning undocumented; `list_agents`/`list_catalog_tools`/`describe_catalog_tool` perform no principal check | MEDIUM |
| G-ID-07 | Catalog allowlist applied to list/describe only; `invoke_catalog_tool` composes with unfiltered `deps.tool_registry` (engine:280-294) — latent (no facade mounts invoke, custom hosts can) | MEDIUM |
| G-ID-08 | Default shared tenants ("mcp"/"lab"/"research") for unauthenticated callers | folded into G-ID-01 |
| G-ID-09 | `app.state` is a one-shot copy at couple time; later identity changes never reach MCP | MEDIUM, folded into G-ID-02 |

### B.2 Error contract (MRC-2)
| ID | Finding | Sev |
|---|---|---|
| G-ERR-01 (R-P7-05) | `MCP_PRINCIPAL_REQUIRED`, `MCP_TENANT_ASSERTION_CONFLICT`, `MCP_RESULT_BOUND_EXCEEDED` declared (`contracts/mcp_server_definition.py:31-32`) but never raised; tests check enum membership only | MEDIUM |
| G-ERR-02 | Runtime uses ~25 undeclared lowercase/prefix strings (`unauthorized:`, `forbidden:`, `identity_conflict:`, `mcp_payload_byte_limit_exceeded`, `mcp_concurrent_call_limit_exceeded`, bridge codes) → no single wire contract | MEDIUM |
| G-ERR-03 | Overflow surfaces as `RuntimeError`→`isError=true` for lists but as a normal `isError=false` dict for `run_agent`/describe → client can read overflow as success | HIGH (correctness) |

### B.3 Result bounds (MRC-2)
| ID | Finding | Sev |
|---|---|---|
| G-RB-01 (R-P7-03) | Custom host tools unbounded (`_bind_custom_host_tool_for_fastmcp`, engine:104-138); pinned as "recorded gap" in p7 failure-injection L2:480-492 | MEDIUM |
| G-RB-02 | Cross-loop catalog path returns `terminal.payload` raw; bridge `_serialize_tool_result` never reaches the ingress (`root_execution_child_work_ingress.py:279-297`) | MEDIUM |
| G-RB-03 | `bound_mcp_structured_result` does full `json.dumps` before checking size (memory amplification); `serialize_bounded_mcp_json` unused | MEDIUM |
| G-RB-04 | Ingress exception path returns `{"error":"internal_error","detail":str(exc)}` (info leak); `describe_catalog_tool` echoes caller `tool_id` | LOW/MEDIUM |

### B.4 Deadline / cancel / resilience (MRC-3)
| ID | Finding | Sev |
|---|---|---|
| G-DL-01 (R-P7-04) | No server-side deadline anywhere; p7 L2 test asserts `host._deadline_authority_resolver is None` (N/A) | MEDIUM |
| G-DL-02 | Catalog invoke runs via `asyncio.to_thread` (`catalog_tool_child_work_port.py:73-77`): cancelling the awaiter does not stop the thread; no timeout; cross-loop `wait_timeout_s=120` hard-coded | MEDIUM |
| G-DL-03 | Detached run releases the admission slot while execution still runs → concurrency cap does not bound live executions; catalog tools are not admission-gated (engine:304-308) | MEDIUM |
| G-DL-04 | Worst case two sequential 30 s waits before `CancelledError` (`mcp_host_execution_transport_cancel.py:567-586`); `suppress(Exception)` hides host failures after waiter loss; `_client_disconnected` swallows all exceptions as "connected" | MEDIUM |
| G-DL-05 | `__aexit__` re-acquires the lock → `_inflight` leak if cancelled while contended | LOW/MEDIUM |
| G-DL-06 | Dead assignments: `deps.transport_detachment_supervision`, `deps.auto_wire_catalog_transport_supervision` are no-ops (engine:~286) | MEDIUM |
| G-DL-07 (F-P8-07) | Evidence-scope limits ACC-10/11/13/15 (items a–g) | LOW |

### B.5 Traceability (MRC-4)
G-TR-01 (R-P7-01 / REQ-MCP-OBS-01): no dedicated MCP invocation correlation id; FastMCP `RequestContext.request_id` unused; no `traceparent`; only `task_id`/`run_id`/`execution_id` (ACC-17). MEDIUM.

### B.6 Tests / qualification (MRC-5)
| ID | Finding |
|---|---|
| G-TQ-01 (R-P7-02) | 11 stable failures (observed 11–13) in 3 files; recorded hypotheses: no provider configured (`6d436f323`), STRICT agent-contract authority, TL11 subprocess startup. Code read adds: port bind-then-close race, undrained `stderr=PIPE`, 0.5–0.6 s client timeouts conflating connect/initialize with hang, `time.sleep` polling, `builtins.__import__` monkeypatching, `pytest.raises(Exception)`, missing `LOCAL_WORKSPACE_VECTOR_STORE` env in the runtime file, `governed_contractor` missing from `APPLICATION_FACTORIES`. Files are `gate`/`no_ci` → green PR CI says nothing |
| G-TQ-02 (F-P8-08) | Self-referential assertions: `inspect.getsource` (p7 L2:356-374), regex/`read_text` source scans (surface_opt_in:49-61), near-tautology (architecture_gates:46), matrix-references-test-names (p7 L3 protocol:257-265), "completed or cancelled" accept-either (p7 L2:351), disconnect injected by patching |
| G-TQ-03 | Fail-open **enshrined** as expected behaviour: `test_mcp_lab_inprocess_without_auth_preserves_client_tenant` (trust_a:227-243), p7 L2:741-748 `test_acc_12_local_dev_state_without_credentials_trusts_client_assertion` |
| G-TQ-04 | All identity tests wire auth on the **inner** app before coupling — no test goes through a real facade factory; `test_reference_apps_mcp.py` checks route existence only |
| G-TQ-05 | No mutation testing ever executed against MCP product code |

### B.7 Protocol / compatibility / documentation (MRC-6)
| ID | Classification |
|---|---|
| G-PR-01 | FastMCP advertises `logging`, `prompts`, `resources`, UI extension while empty/no-op → **misleading advertised capability** (class 3), not missing mandatory; stays UNSUPPORTED_BY_DESIGN (OD-MCP-03) |
| G-PR-02 | stdio/SSE absence not product-enforced (probe only) → class 4/accidental-exposure watch: add a negative test, no code |
| G-PR-03 | Exec-generated wrapper (`exec()` with `repr()` defaults, engine:123-137): annotations not carried; non-literal defaults would generate invalid code → robustness defect, no new capability |
| G-PR-04 | Duplicate paths: two coupling functions + `mount_mcp_definition_on_harness_host`; two mount-path normalizers (only `mcp_host_mount.py:29-40` validates); three composition entry points with different default tenants; boundary/supervision from two sources (deps vs `app.state`) |
| G-DOC-01..12 | Record defects: "same set / 11" wording in `P7_EVOL_ENT_DELTA_RECORD:30`, `p7_selection.py`, `QUALIFICATION_MATRIX` row 19, `T-IGX-035-H:69`; 13 vs 14 consistency count; closure report vs SHA ladder (`FINAL_REMOTE_HEAD` row malformed, certified manifest still 40 zeros / `PENDING_PUBLICATION_MERGE`); stale governance lists omitting P8-PROC-03, P7-PROC-01/02, F-MCP-P5-GOV-002, F-MCP-P5-MEM-001; OD-MCP-P8-RESIDUAL-01 "Yes, both" vs deferral (unsigned); no ruling recorded for closure conditions C5/C8; gap register mostly P1 snapshot; INDEX says OD "PENDING"; auditor register still `RECORDED_PENDING_OPERATOR_DISPOSITION` |

### B.8 Process / governance obligations
P8-PROC-01..03, P7-PROC-01..03, F-MCP-P5/P6-GOV-*: **recorded, not waived** — campaign must not repeat; OPEN-BP-01 deferred (accepted limitation); `LIM-RESEARCH-STRICT-FACTORY-MEMORY` PROD-Q/MEM, untouched; PDC-P7-00 protocol-header sub-question (PDC-BL-F01) open.

---

## C. Root-cause and ownership analysis

| Class | Items | Owner |
|---|---|---|
| **A MCP-owned** | G-ID-02..07, G-ERR-*, G-RB-*, G-DL-*, G-TR-01, G-TQ-*, G-PR-*, G-DOC-* | MCP-RC campaign (this plan) |
| **A→C bundle (ownership preserved)** | G-ID-01 (R-P8-01) | stays under OD-MCP-P8-RESIDUAL-01 / PDC-P7-00; implemented as MRC-1 *in the same integration window* — **not** relocated |
| **B existing PDC-X** | none of the MCP items; B-D6 (T-IGX-036-P) owns signing/poisoning/audit-trail defaults only and explicitly excludes MCP | none transferred |
| **C shared** | single audit scope + single rebind at `INTEGRATED_BASELINE_SHA`; shared hotspot `intergrax/applications/_shared/` (PDC WBS rule: ≤1 writer batch per wave) | PDC orchestrator + MCP-RC orchestrator |
| **D PROD-Q/MEM** | `LIM-RESEARCH-STRICT-FACTORY-MEMORY`; production readiness | untouched |
| **E accepted limitation** | OPEN-BP-01; progress/logging/tasks/completions unsupported; resources/prompts empty; non-cooperative sync handlers (bounded + documented, cannot be killed in-process) | operator-accepted |

Root causes (shared): (1) identity authority is split between engine, coupling wrapper and app factories with order-dependent wiring; (2) there is no single choke point for result bounding or error mapping; (3) the engine was composed from deps that are partly dead; (4) qualification was scoped to "configured hosts" so unconfigured composition was never an acceptance dimension.

### FRZ mapping and ARCH-FREEZE blocking (proposal for operator, not self-classified as final)
| Finding group | FRZ family | Proposed class |
|---|---|---|
| G-ID-01..07 | Security, Tenant isolation, Governance, Configuration | **Mandatory before ARCH-FREEZE** |
| G-ERR-03, G-RB-01..03 | Security (resource bounds), Compatibility | **Mandatory** |
| G-DL-01..06 | Execution, Reliability, Recovery | **Mandatory** (non-cooperative sync thread limit = accepted+documented) |
| G-TR-01 | Traceability/Observability | Recommended; may be ACCEPTED_SCOPED (D-5) |
| G-TQ-* | Qualification quality | Mandatory (G-TQ-01 exit-0 or classified exclusion) |
| G-PR-01..04 | Compatibility/Architecture boundaries | Doc + test; code only G-PR-03/04 minimal |
| G-DOC-* | Traceability | Docs-only supplemental record |

---

## D. Target architecture and corrective design (one authority per responsibility)

Nothing below introduces a new identity, execution, policy, deadline or registry authority.

1. **Identity & tenant (MRC-1).** Reuse `harness_auth.resolve_harness_authenticated_principal_for_state`, `harness_principal_to_request_identity`, `resolve_mcp_run_agent_identity`. Add one explicit composition parameter `identity_mode ∈ {HARNESS_BOUND (default), LOCAL_DEV_EXPLICIT}` carried by the MCP deps (code-level opt-in only; never an environment variable).
   - **HARNESS_BOUND (default, fail-closed):** composition/mount raises `McpDefinitionError` (code `MCP_PRINCIPAL_REQUIRED`) when no effective harness auth with a verifier is present at mount time; per call, no principal ⇒ `ToolError("MCP_PRINCIPAL_REQUIRED: …")`; asserted tenant/user may only *confirm* the principal's, mismatch ⇒ `MCP_TENANT_ASSERTION_CONFLICT`. Asserted values never create identity. Defaults ("mcp"/"lab"/"research") are removed from the authoritative path.
   - **LOCAL_DEV_EXPLICIT:** only by explicit opt-in at composition; admitted only for in-process/loopback transports; refuses under strict-production serving (existing `get_strict_serving_http_caller_principal`) and when a non-loopback bind is detected; emits a startup audit/warn event; asserted identity is labelled `unverified` in task metadata; never reaches catalog invoke (already fail-closed). Trust boundary and isolation documented in the contract matrix.
   - Remove the factory-ordering hazard: coupling takes auth from a resolver called at request time (not a one-shot copy), and a startup assertion verifies the MCP sub-app is covered by the same middleware as the host. All 8 facades get `identity_mode`/`harness_auth` pass-through (additive keyword, default HARNESS_BOUND). `run_research_pipeline` is routed through `resolve_mcp_run_agent_identity` and the same admission/cancel path as `run_agent`.
   - `invoke_catalog_tool` composes with the allowlist-filtered `registry_view`. `list_*`/`describe_*` get the same principal gate. `apply_harness_auth_middleware` raises when `require_api_key` is set with no key/provider.
   - Dead `deps.transport_detachment_supervision` / `auto_wire…` flags are either wired (preferred, MRC-3) or removed with documentation.
2. **Result bounds & error contract (MRC-2).** One choke point: all tools are registered through a single `register_bounded_tool` wrapper in the engine (built-in, custom `_bind_custom_host_tool_for_fastmcp`, catalog, `run_research_pipeline`). Reuse `intergrax/tools/mcp_result_bounds.py` (extend, do not duplicate): incremental bounded encoding (`JSONEncoder.iterencode` with early abort) instead of full `json.dumps`; byte accounting after UTF-8 encoding; nested/large exception messages truncated and never `str(exc)` on the wire. Overflow always raises `ToolError("MCP_RESULT_BOUND_EXCEEDED: …")` ⇒ `isError=true` on every path. The cross-loop ingress receives the bridge's serializer through its existing parameter (no edit of `runtime/` unless discovery in W1 shows no injection point — then STOP+escalate). A single `mcp_error_codes` mapping in `contracts/mcp_server_definition.py` maps every legacy lowercase code to a declared `MCP_*` code with a compatibility table (legacy strings kept as `detail`).
3. **Deadline, cancel, resilience (MRC-3).** W1 finding: the canonical Execution deadline authority is `ExecutionDeadlineAuthorityResolver` (`intergrax/runtime/execution/deadline_authority/resolver.py`), composed by the host via the run budget; MCP cannot and must not feed or replace it. Per OD-MCP-RC-02, `max_call_seconds` is therefore an MCP transport *observer trigger* of the existing governed-cancel path (same as client disconnect), not a deadline authority and not a timer lifecycle. Catalog invoke: the cross-loop wait becomes configurable; on expiry the outcome is `UNKNOWN_OUTCOME` (never success), the slot is released only after bounded drain or after supervisor handoff **and** accounted in a separate `detached_inflight` counter so the cap still bounds live executions; catalog tools pass through the admission gate. `__aexit__` release becomes cancellation-safe (non-awaiting release). Disconnect probe logs and fails safe (treat probe exception as unknown, not connected, after N consecutive errors). Document the hard limit: a non-cooperative synchronous thread cannot be killed in-process; MCP guarantees bounded wait + no false success + slot/ledger accounting, not termination.
4. **Traceability (MRC-4).** `mcp_invocation_id` = per-call opaque correlation id (uuid4 hex, or FastMCP `request_id` when present, prefixed), generated at the single registration choke point, additive field in the `run_agent` payload and in task/execution metadata, and in MSE/audit evidence. It is a **correlation key only** (not derived from tenant/user, no PII, never an execution authority). Reuse existing TRACE-X/OBS emit API (W1 locates it; none exists in MCP files today). `traceparent` propagation only if the existing observability API already supports it (no new tracer).
5. **Tests & mutation (MRC-5).** Behavioural tests replace source-string tests; real-facade identity tests; hardened subprocess fixtures (port 0 + readiness probe, drained stderr, separate connect vs call timeouts, explicit provider fixtures — no timeout inflation, no xfail, no exclusion without evidence). Mutation harness: `scripts/qualification/mcp_rc_mutation.py` applies the 8 mutants to a throw-away worktree under `_wt/_scratch/`, runs the targeted tests (one pytest process), records killed/survived to `.tmp/proof/` then publishes the report; never touches accepted branches.
6. **Protocol (MRC-6).** No capability change. Classification table (G-PR-01..04) in the contract matrix; negative tests for stdio/SSE; G-PR-03 made robust (build signature with `inspect.Signature`, no `exec`) only if W1 shows it is exercised by custom tools; G-PR-04 duplicates consolidated minimally (single mount-path normalizer). A suppression of advertised-empty capabilities is **not** implemented (would be a protocol behaviour change) — decision D-6.
7. **Compatibility policy.** Breaking-by-design changes (fail-closed identity default, `isError` semantics for overflow, `MCP_*` wire codes) are versioned: MCP contract "v1 → v1.1 (security hardening)" with migration note per facade, a deprecation window for legacy code strings (kept in `detail`), and `LOCAL_DEV_EXPLICIT` as the compatibility path for dev hosts. Final classification needs D-4.

---

## E. Consolidated remediation backlog

Ticket ids are **proposed** `T-IGX-038-A..F` (series `037` is taken by SFX; W0 confirms 038 is free). Role abbreviations: ORCH orchestrator, IMPL implementer lane, AUD independent auditor, CERT certifier, VER terminal verifier. Status for all: `PLANNED`.

| TASK_ID | FINDING_ID | DOMAIN_OWNER | SEV | PRODUCT_CHANGE | DEPENDENCIES | FILES (write set) | TESTS | ACCEPTANCE | CERTIFICATION | EVIDENCE |
|---|---|---|---|---|---|---|---|---|---|---|
| MRC-0.1 | all | ORCH | — | no | — | `.tmp/session/mcp-rc/baseline.*` | baseline replay of 3 R-P7-02 files + recorded contract (needs D-2) | reproduced list per test id | — | baseline log w/ SHA header |
| MRC-0.2 | ownership | ORCH | — | no | — | gap-register supplement draft | `pdc_x_wbs.py --check` | PDC overlap table signed off by PDC orch | — | ownership matrix |
| MRC-1.1 | G-ID-01,06,08 | MCP-RC (bundle R-P8-01) | HIGH | yes | W1 design, D-1 | `mcp_harness_principal.py`, `contracts/mcp_server_definition.py` | red-first L1/L2 | HARNESS_BOUND default fails closed at mount and per call; asserted ≠ authority | G6 + IIAC delta | test logs @SHA |
| MRC-1.2 | G-ID-02,03,05,09 | MCP-RC | HIGH | yes | 1.1 | `fastapi_mcp.py`, `harness_auth.py`(guard only), `applications/*/factory.py`(lab, legal, research, + 5 others pass-through) , 8× `applications/*/mcp/server.py` | real-facade L2 per app ×8 | MCP covered by same auth as host in every factory order | G6 | per-app matrix |
| MRC-1.3 | G-ID-04 | MCP-RC | HIGH | yes | 1.1 | `applications/research_application/mcp/server.py` | cross-tenant L2/L3 | research pipeline resolves principal; forged tenant rejected | G6 | tests |
| MRC-1.4 | G-ID-07 | MCP-RC | MED | yes | 1.1 | `mcp_server_engine.py` | invoke non-allowlisted tool negative | invoke uses filtered view | G6 | tests |
| MRC-1.5 | G-ID-06 | MCP-RC | MED | yes | 1.1 | `mcp_nexus_server.py`, `mcp_catalog_tools.py` | principal gate on list/describe | no unauthenticated discovery in HARNESS_BOUND | G6 | tests |
| MRC-2.1 | G-RB-01..04,G-ERR-03 | MCP-RC | HIGH | yes | 1.* serial on engine | `mcp_result_bounds.py`, `mcp_server_engine.py`, `mcp_nexus_server.py`, `mcp_catalog_tools.py`, `mcp_catalog_invocation_bridge.py` | oversize, nested, UTF-8, giant exception, concurrent | all paths ⇒ `isError=true` + `MCP_RESULT_BOUND_EXCEEDED`; memory bounded | G6 | tests |
| MRC-2.2 | G-ERR-01,02 | MCP-RC | MED | yes | 2.1, 1.1 | `contracts/mcp_server_definition.py` | runtime-path emission tests per code | every declared code raised by a real path; legacy→declared table | G6 | tests+table |
| MRC-3.1 | G-DL-01,02,04,06 | MCP-RC | MED | yes | 2.1 | `mcp_host_execution_transport_cancel.py`, `mcp_server_engine.py`, `mcp_catalog_invocation_bridge.py`, definition | deadline L2/L3, non-cooperative handler, disconnect | server work bounded by the MCP transport trigger (OD-MCP-RC-02) and the host-owned canonical run budget; outcome unknown never reported as success | G6 | tests |
| MRC-3.2 | G-DL-03,05 | MCP-RC | MED | yes | 3.1 | `mcp_engine_concurrency.py` | saturation, detached accounting, cancel-while-contended | cap bounds live executions; no leaked `_inflight` | G6 | tests |
| MRC-4.1 | G-TR-01 | MCP-RC | MED | yes | 2.1 | engine registration choke point, `mcp_nexus_server.py`, MSE hook | ACC-17 extension; privacy test | id present, unique, correlated, not tenant-derived (or D-5 defer) | G6 | tests |
| MRC-5.1 | G-TQ-01 | MCP-RC | MED | tests only (+ product defects if found) | 0.1 | 3 test files, `tests/fixtures/tool_life_x_tl11_*`, `p7_selection.py` | the 29 tests | each of 11 failures: root cause + fix/classified exclusion; contract exit 0 or accepted exclusion list | G6 | per-test disposition |
| MRC-5.2 | G-TQ-02..04 | MCP-RC | MED | tests only | 1–4 | `tests/unit/applications/`, `tests/qualification/mcp_server_x/` | rewritten behavioural tests | no source-string/self-referential assertion for critical invariants | G6 | grep audit |
| MRC-5.3 | G-TQ-05 | MCP-RC | MED | script only | 1–4 | `scripts/qualification/mcp_rc_mutation.py` | 8 mutants | each mutant killed by ≥1 test, survivors dispositioned | G6 | mutation report |
| MRC-6.1 | G-PR-01..04 | MCP-RC | LOW/MED | minimal | 1–3 | contract/protocol matrix, `fastapi_mcp.py`/`mcp_host_mount.py` normalizer | stdio/SSE negative, advertised-capability snapshot | classification recorded; no new capability | G6 | matrix |
| MRC-6.2 | G-DOC-01..12 | MCP-RC | LOW | docs | — | new `MCP_SERVER_X_RC_*` records only (P8 files untouched) | `verify_*` doc validators | corrections recorded additively | VER | records |
| MRC-7.1 | EVOL/ENT/FRZ | ORCH+CERT | — | no | 1–6 | EVOL/ENT delta records | EVOL-X, ENT-AUDIT delta | PASS; FRZ unchanged at 225 | CERT | records |
| MRC-8.x | audit/cert/rebind | AUD/CERT | — | no | 7.1 | `iiac_evidence/…`, activation record (CERT only) | gate scripts | see G | — | manifests |
| MRC-9.1 | closure | VER | — | no | 8 | final dossier, residual register | terminal verification | see I | VER | dossier |

Shared-fix batches (one mechanism, many findings): **MRC-1** = {R-P8-01, R-P7-06, R-P7-05 principal/tenant codes, G-ID-*}; **MRC-2** = {R-P7-03, R-P7-05 bound code, G-ERR-*}; **MRC-3** = {R-P7-04, F-P8-07, G-DL-*}; **MRC-4** = {R-P7-01}; **MRC-5** = {R-P7-02, F-P8-08}; **MRC-6** = docs/protocol.

Definition of done per batch: red-first tests merged with the fix; contract + adversarial + regression green on the exact head SHA; no new authority (checked by `test_mcp_server_engine_architecture_gates.py` extended); ticket `quality_gate`; CI five contexts green on the pinned head before it counts.

---

## F. Master execution plan W0–W9

Single integration branch `mcp-rc` (+ per-batch worktrees under `<parent>/_wt/intergrax/<slug>`, slugs ≤24 chars, e.g. `mrc1-identity`), PRs target the **PDC-X integration target confirmed in W0** (see D-1; product PRs must not hit `development` before the rebind or `governance-fast` goes red). One orchestrator; serial ownership of SSOT/activation records; engine file `mcp_server_engine.py` is a serial hotspot merged in order MRC-1→2→3→4.

| Wave | Content | Role | Parallelism / write-set | Gate / STOP |
|---|---|---|---|---|
| **W0** | Baseline (A.1) re-verified at execution SHA; R-P7-02 replay of 3 files + recorded contract (one pytest process, logs to `.tmp/session/mcp-rc/`); PDC overlap check vs WBS hotspot rule; identify PDC-X integration target branch and INTEGRATED_BASELINE_SHA timing; confirm ticket series free; reproduce each G-ID/G-RB/G-DL defect as a failing test spec (no product edits) | ORCH + 1 Explore | read-only | STOP if base moved by a product commit, if D-1..D-3 unsigned, or if write set collides with a PDC batch in the same wave |
| **W1** | Design freeze: name the canonical deadline authority, the TRACE/OBS emit API, the ingress serializer injection point; compatibility table; per-batch ticket + plan digest + `Approved:` (delegation per existing pattern); ADR-MCP-RC-01 only if D-4 says breaking | ARCHITECT (read-only) → ORCH | docs only | STOP if a required hook does not exist (would need a new authority) → escalate |
| **W2** | MRC-1 Identity (1.1→1.2‖1.3‖1.4→1.5) | IMPL-A | lane A owns `mcp_harness_principal`, `fastapi_mcp`, factories, facades | per-batch quality gate; STOP if factories collide with PDC B-D4/D6 writers in same wave |
| **W3** | MRC-2 then MRC-3 (serial on engine); MRC-2.2 mapping | IMPL-B | lane B owns bounds/bridge/transport_cancel/concurrency; tests-writer lane C parallel in new test files | STOP if a runtime/ edit is needed |
| **W4** | MRC-4 trace id; closes with ACC-17 extension | IMPL-B or A | engine choke point (serial) | skipped if D-5 = ACCEPTED_SCOPED |
| **W5** | MRC-5.1/5.2 (can start in W2 on test files not yet in flux), 5.3 mutation, full relevant regression L1/L2/L3 (P7 new 164 + matrix 14 + regression selection 141 + new MRC suites) | TESTER lanes | tests dirs only; ≤1 pytest process per step | STOP if a failure is a product defect → back to owning batch |
| **W6** | EVOL change-impact, ENT-AUDIT delta, FRZ impact classification, COMPAT report, cross-domain acceptance (8 facades) | ORCH (EVOL/ENT scripts) | serial | FRZ must stay 225; any promotion ⇒ STOP |
| **W7** | Independent G6 audit (new session, no implementer context) + corrective loop (≤3 loops; each loop = targeted fix + exact-SHA requalification) | AUD (≠IMPL) | none | any product change after accepted audit ⇒ requalify |
| **W8** | Evidence publication (exact SHA), IIAC manifest + `verify_iiac_role_separation --strict-product-diff`, operator/delegate activation line, two-phase rebind at INTEGRATED_BASELINE_SHA with PDC-P7-00, **merge only after all five CI contexts green on pinned head** (`ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate`) | CERT (≠AUD≠IMPL) | serial | merge-before-checks = STOP + deviation record |
| **W9** | Terminal verification (new session), `MCP_SERVER_X_RC_CLOSURE_REPORT`, residual register, ARCH-FREEZE/PDC handoff, roadmap/INDEX supplemental pointer (not overwriting P8 text) | VER (≠ all) | serial | — |

STOP conditions apply only to: policy/approval blockers, security findings, architecture-authority needs, evidence gaps, a red mandatory check — not after each subtask. Operator escalations are batched into one decision pack (section H) presented at W0/W1; later escalations only for new blockers.

---

## G. Independent qualification and certification plan

**Mechanism chosen:** *Supplemental corrective qualification* (new additive records `MCP_SERVER_X_RC_*`; P8 files untouched) + *targeted recertification* of the affected acceptance rows (ACC-02, 09–15, 17 and any matrix row whose test files change) at the new exact SHA + EVOL-X change-impact + ENT-AUDIT delta, with the ACT-N11 delta audit and the **single two-phase rebind carried by PDC-P7-00** per OD-MCP-P8-RESIDUAL-01. The historical `PASS_SCOPED` is never described as covering the new changes. A standalone MCP rebind is rejected: it would force a second rebind for PDC-X and contradicts the recorded operator instruction (alternative = D-1 option B).

- **Roles/independence:** IMPL lanes ≠ AUD (IA-MCP-RC, new session, no implementer context) ≠ CERT (new session) ≠ VER (new session). Signing of activation line by operator, or by a commissioned delegate under the existing delegation pattern (Amendment 01 R-1..R-6); never by the implementer; no fabricated timestamps/signatures.
- **Test levels:** L1 contract; L2 composition (real facade factories, all 8); L3 streamable-HTTP physical (loopback) for identity, bounds, deadline/disconnect; failure-injection L2/L3; regression selection; mutation report. L2 never counted as L3.
- **Mutants (MRC-5.3):** tenant admission bypass; principal/asserted mismatch ignored; result-size check removed on custom tools; deadline not propagated; cancellation isolation removed; admission slot release dropped; error code mapping swapped; duplicate/unauthorized execution path (research pipeline bypass restored).
- **Audit scope:** `audit_scope.json` lists the exact changed product paths of MRC-1..4; PDC-X audit lanes keep their own paths; shared-hotspot files assigned to exactly one lane.
- **Exact-SHA discipline:** PR head pinned (`headRefOid` compared with pushed SHA before any "ready to merge" claim); audit/evidence/cert SHAs recorded in a new `MCP_SERVER_X_RC_SHA_LADDER`; any product commit after audit acceptance voids it.
- **No** ruleset change (OPEN-BP-01 deferred); the five-context rule remains the only control and is enforced by the orchestrator checklist.

---

## H. Operator decision pack (prepared, unsigned)

Not requested because already resolved: ownership of R-P8-01 (OD-MCP-P8-RESIDUAL-01), OPEN-BP-01 deferral, SEQ-01, IIAC-SCOPE structure.

| ID | Decision | Recommendation |
|---|---|---|
| D-1 | Merge/rebind path: (A) carry all MRC batches inside the PDC-P7-00 integration window with one rebind (extends the OD from R-P8-01 to R-P7-01..06 + hardening); (B) separate MCP rebind first | **A**. Confirm the integration branch/target for product PRs (#488/#489 still target `development`) and that the lapsed OD-MCP-P7-P8-AUTH-01 authorizations (branches, sessions, merges) are re-granted for this campaign |
| D-2 | Authorize *bounded diagnostic pytest* (C10/C15 freeze): the 3 R-P7-02 files (29 tests) + mandatory contract, one process at a time, Windows environment unchanged | **Yes** (W0/W5 depend on it) |
| D-3 | Add an MCP-RC auditor lane and an MCP-RC line in `OD-PDC-IIAC-SCOPE` evidence layout (`iiac_evidence/PDC-X/MCP-RC/`), and who commissions signing delegates | **Yes** |
| D-4 | Breaking-change classification: fail-closed default identity, overflow `isError=true`, `MCP_*` wire codes ⇒ contract "v1.1" + ADR + migration note, `LOCAL_DEV_EXPLICIT` as dev path | Accept as versioned, non-silent |
| D-5 | R-P7-01: implement correlation id (recommended, additive, cheap in the same audit) or ACCEPTED_SCOPED with owner | Implement |
| D-6 | Advertised-but-empty capabilities (logging/prompts/resources/UI): document only (recommended) vs suppress (protocol behaviour change) | Document only |
| D-7 | ARCH-FREEZE blocking set = classes marked "Mandatory" in section C; G-TR-01 non-blocking if D-5 defers | Approve |
| D-8 | Ticket series `T-IGX-038-A..F` and WBS/DAG amendment (`pdc_x_wbs.py` generator) to register MCP-RC as an additional shared-hotspot writer in an explicit wave | Approve |
| D-9 | PDC-BL-F01 (IIAC header wording) — remains with PDC-P7-00; no MCP-RC action | Informational |
| D-10 | Why #475/#476/#478 were closed and whether OD-MCP-P8-RESIDUAL-01 text referencing them needs an addendum | Informational |

---

## I. Terminal closure matrix

Per finding: `ROOT CAUSE → CORRECTIVE IMPLEMENTATION → CONTRACT TEST → ADVERSARIAL TEST → REGRESSION → INDEPENDENT AUDIT → CERTIFICATION → FINAL DISPOSITION ∈ {CLOSED, TRANSFERRED, ACCEPTED_SCOPED}`. No ID may disappear; renames require an alias row.

| # | Terminal condition | Validator / evidence |
|---|---|---|
| 1 | All mandatory security gaps resolved (G-ID-*, G-RB-*, G-ERR-03) | adversarial suite list: missing creds, no harness auth, empty `HarnessAuthState`, forged tenant, forged user, cross-tenant tool call, custom tools, catalog tools, local-dev mode, streamable HTTP L3, conflicting identities |
| 2 | No unexplained MCP regression failures; R-P7-02 contract exit 0 or classified/accepted exclusions | per-test disposition table + replay logs |
| 3 | Canonical identity/tenant authority preserved | architecture-gate test: no second authenticator |
| 4 | Governance/Execution authorities preserved | same test + EVOL change-impact |
| 5 | Bounds on every result-producing path | bounds matrix tests |
| 6 | Deadline/cancel contract proven (incl. non-cooperative limit documented) | L2/L3 + failure injection |
| 7 | Declared error codes ⇔ runtime emission | per-code runtime test (not enum check) |
| 8 | Trace correlation accepted or ACCEPTED_SCOPED (D-5) | ACC-17 extension |
| 9 | Resilience scenarios (3.8 list) demonstrated | scenario→test table |
| 10 | Mutation-resistant tests for critical invariants | mutation report, survivors dispositioned |
| 11 | No new execution/policy/registry/identity/trace-authority owner | architecture gate + audit |
| 12 | Compatibility reviewed, versioned if changed | contract matrix v1.1 + migration note |
| 13 | FRZ impact classified; counter 225 unchanged | FRZ validator |
| 14 | EVOL/ENT delta PASS | scripts output at exact SHA |
| 15 | ACT-N11 valid on published SHA | `verify_iiac_policy_activation.py --require-in-force` = 0 |
| 16 | Independent audit accepted | auditor dossier + manifest |
| 17 | Required CI green on pinned head before merge | five-context check recorded with timestamps |
| 18 | Every finding CLOSED/TRANSFERRED/ACCEPTED_SCOPED with evidence | residual register |
| 19 | No unresolved mandatory blocker | VER report |
| 20 | One final published terminal dossier | `MCP_SERVER_X_RC_CLOSURE_REPORT` |

---

## J. Estimated remaining scope and risk

- **Implementation batches:** 6 (MRC-1..6) + 4 governance waves; ~6 tickets; product files ≈ 14 under `intergrax/applications/_shared`, 3 contracts/tools, 8 facade `mcp/server.py`, ≈ 4–8 factories; test files: 3 repaired + ≈ 8–10 new/rewritten + 1 mutation script.
- **Tests:** existing contract = 164 + 14 + 141 passed baseline (+ 29 in R-P7-02 files, 11 red); expect +80–120 new tests; one pytest process per step.
- **Cross-domain dependencies:** PDC-P7-00 rebind timing; shared hotspot `_shared/` with B-D4/D6/D5/D8; B-REG-P (ollama, T-IGX-036-M) is independent and waits for MCP P8 only.
- **Certification blockers (risk-ranked):** (1) D-1 not resolved ⇒ merge path undefined (high); (2) IIAC audit lane/delegation not commissioned (high); (3) hidden dependency on a canonical deadline hook that does not exist ⇒ architectural STOP (medium); (4) ingress serialization needs a `runtime/` edit (medium); (5) fail-closed default breaks reference apps/fixtures that rely on trusted assertions — mitigated by `LOCAL_DEV_EXPLICIT` and a migration sweep (medium); (6) R-P7-02 subprocess failures turn out to be environment-bound and not fixable ⇒ classified exclusion needing independent acceptance (medium); (7) repeat of premature-merge process deviation (mitigated by pinned-head checklist).

---

## Consolidated execution instruction (for Agent mode after operator approval)

> Execute the MCP-RC master plan at `agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md` as ONE coordinated campaign under a single orchestrator. Prerequisites: operator has recorded D-1 (path A, integration target, re-granted authorizations), D-2 (bounded diagnostic pytest), D-3, D-4, D-8. Start in an isolated worktree per `CLAUDE.md` rules (`<parent>/_wt/intergrax/<slug>`), ticket-first (`T-IGX-038-A..F`, plan digest, `Approved:` per existing delegation pattern), stay on the operator-named branch, commit trailer `Ref: governance/tickets/<id>.md`. Run W0 and W1 first, publish the decision pack for any item still unsigned, then W2–W6 with red-first tests, serial ownership of `mcp_server_engine.py`, one pytest process at a time, no xfail/exclusion/timeout inflation without a recorded root cause. Do not alter P8 records, ACT-N11/IIAC logic, GitHub rulesets, FRZ counters or MCP v1 capabilities; do not create a second identity/execution/deadline/trace/registry authority. Implementers never audit, certify, sign or verify; W7 audit, W8 certification and W9 verification run in separate fresh sessions. Merge only when the five required CI contexts are green on the exact pinned PR head, after an accepted independent audit and the PDC-P7-00 two-phase rebind. Stop only on policy/approval/security/architecture/evidence blockers; otherwise continue autonomously through W9 and publish the final terminal dossier.

**PLAN_READY_FOR_OPERATOR_REVIEW**
