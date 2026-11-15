# PDC-P7 independent audit of domain D4 (HOST-RUNTIME) - IA-D4

| Field | Value |
| --- | --- |
| **Document** | `PDC_P7_D4_AUDIT_REPORT.md` |
| **Stage** | PDC-X / P7 / D4 (`iiac_evidence/PDC-X/D4/`) |
| **Role** | Independent auditor `IA-D4` (policy `IIAC-2026-11-05.1`, Option A of `OD-PDC-IIAC-SCOPE`). Never implementer, signer or certifier of D4; no product, test, script or governance file was edited. |
| **Audited SHA** | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (last product/test-relevant commit) |
| **Audit execution / test execution SHA** | `3b371ef9ddee209eb707f184337708303a956023` (descends from the audited SHA; `git diff --name-only c08efe46a HEAD` over `intergrax agents applications scripts tests` is empty; the only non-docs delta is six `governance/tickets/T-IGX-038-*.md` closeout files) |
| **Auditor session** | `PDC-P7-D4-IA-D4-AUDITOR-pdc-x-ia-d4-966471` (branch `claude/pdc-x-ia-d4-966471`) |
| **Implementation under audit** | `T-IGX-036-R` (B-D4, commit `145b49bad`), B-P6-R item 17 launcher stub (`97a5187b2`, `T-IGX-036-T`), docs `0e92fd812` (`T-IGX-036-H`), `T-IGX-036-I` (GR-12), `T-IGX-040-A` (impact map) |
| **Overall verdict** | **PARTIAL** (manifest value `CONDITIONAL`; the manifest schema has no `PARTIAL`) |
| **Findings** | 0 HIGH, 9 MEDIUM, 7 LOW, 4 INFO (20) |
| **Rows** | 20 requirement rows: 9 PASS, 3 PASS_SCOPED, 8 PARTIAL, 0 FAIL |

This report certifies nothing. It states what held, what did not, and how to reproduce each statement. Repairs are for a later ticket of the owning domain by a different session, followed by a fresh re-audit (at most two loops).

## 1. Independence and scope

- Read as inputs: `OD-PDC-IIAC-SCOPE.md`, the D4 rows (`PDC-D4-R001..R020`) of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, `PDC_OWNERSHIP_MATRIX.md`, `PDC_ARCHITECTURE_DECISIONS.md` and `ADR_PDC_04_HOSTING_LIFECYCLE_STABLE_API.md`, the B-D4 block of `PDC_IMPLEMENTATION_WBS.md`, ticket `T-IGX-036-R`, the P7-00 dossier and `ADDENDUM_1..4` finding records (routed items only), and the `OD-PDC-SCOPE-02` record.
- Not read: any other domain's P7 audit report or findings (the directory listing shows their names; no content was opened). No implementer checkpoint was used as evidence; the P7-00 dossier items are re-verified in section 6.
- Layering checked as part of the audit: `intergrax/hosting` imports only `intergrax.hosting.*`, `intergrax.contracts`, `intergrax.utils` and `intergrax.runtime.observability` (grep over all hosting modules); no `agents/` or `applications/` import.

## 2. Method and environment

1. Per row: read the owner code, read the cited tests, run them, then attack the claim (probes and mutants below).
2. Runs (Windows 11, CPython 3.12.11, pytest 8.4.2, uv 0.13.4, fresh worktree `.venv`; the `mcp` extra was added with `uv run --extra mcp` because `host_01` imports `fastmcp`; one pytest process at a time; logs under `iiac_evidence/PDC-X/D4/logs/`):

| Run | Command (paths narrow) | Result |
| --- | --- | --- |
| D4 suite | `uv run pytest tests/qualification/pdc_x/d4 -q` | 36 passed in 4.70 s; the process exited normally (debt R-05 hang not observed for this directory) |
| host_01 + hosting unit | `uv run --extra mcp pytest tests/qualification/host_01 tests/unit/hosting -q` | 422 passed, 3 failed (all in `tests/unit/hosting/instance`: symlink creation needs a privilege this Windows account lacks, WinError 1314 - environment, see F-D4-19) |
| GR-12 | `uv run --extra mcp pytest tests/qualification/governance/gr12 -q` | 139 passed |
| Governance directory | `uv run --extra mcp pytest tests/qualification/governance -q` | 403 passed (equals the B-REG-T claim) |
| TENANT-X final fast | `uv run --extra mcp pytest tests/qualification/tenant_x/test_tenant_x_final_fast.py -q` | 6 passed (campaign meta checks, not environment wiring) |
| Environment profile / package wiring / architecture | `uv run --extra mcp pytest tests/unit/applications/test_environment_profile.py tests/unit/applications/test_package_wiring.py tests/unit/applications/architecture -q` | 102 passed, 8 failed (none in a D4 row; see F-D4-19) |
| Mutants | `adversarial_scripts/run_mutants.sh` (12 in-memory mutants, one pytest each) | 12 of 12 killed (section 5) |

3. Probes (python, in-process, no file of the repository edited): `adversarial_scripts/probe_a.py` (profile typo/bind, API fingerprint blind spot, PID reuse, metric export, bounded-phase overrun), `probe_b.py` (post-runtime rollback, cancellation, hanging publisher), `probe_c.py` (classifier parity, supervisor conflict, ALWAYS restart), `probe_d.py` (hard-kill of the lease holder), `probe_f.py` (hung `runtime.ready`). `probe_e.py` was inconclusive (empty manifest iterator outside pytest path setup) and is not relied on. The scripts were run from `.tmp/session/ia-d4/probes/` (repo root = `parents[4]`); edit `ROOT` to replay from the evidence directory.

## 3. Per-row verdicts

Vocabulary: PASS (claim holds, no open MEDIUM+ finding on the row), PASS_SCOPED (Operator-authorized exclusion verified), PARTIAL (claim holds in part, or a MEDIUM+ finding undermines it), FAIL (claim false).

| Row | Requirement | Verdict | Evidence at the audited SHA | Findings |
| --- | --- | --- | --- | --- |
| PDC-D4-R001 | Canonical `ApplicationEnvironmentProfile` and `HostedApplicationProfile` | **PASS** | `hosting/contracts/profile.py` frozen, `extra=forbid`, canonical digest; `tests/unit/hosting/test_hosted_application_profile*.py` and `tests/unit/applications/test_environment_profile.py` pass. The matrix cites `host_01`, which does not test this row. | F-D4-15 (LOW) |
| PDC-D4-R002 | Single effective composition root | **PARTIAL** | `wire_application_environment` is the entry used by the nine product `host/wiring.py`; but the lab host factory builds a second tool wiring for MCP, `conformance_check=False` skips all conformance, and no gate enforces one root. | F-D4-05 (MEDIUM), F-D4-15 |
| PDC-D4-R003 | Startup dependency ordering and rollback | **PARTIAL** | Reverse-order rollback proven (`test_r003_*`, mutants M1/M10/M11 killed); post-runtime failure also rolls back correctly (probe P8: `post_ok`, runtime, `core`) but is not in the suite; cancellation skips rollback entirely. | F-D4-01 (MEDIUM), F-D4-14 |
| PDC-D4-R004 | Readiness and degraded states | **PARTIAL** | Degraded optional component, recovery, required-component blocking proven (M2 killed); a hung `runtime.ready()` leaves readiness stale `True`. | F-D4-04 (MEDIUM) |
| PDC-D4-R005 | Graceful shutdown with in-flight drain | **PARTIAL** | Engine drain/cancel/flush proven with an injected controller (`test_xd05_*`, M3/M9 killed); no product adapter or runner seam; cancellation unhandled; unbounded publishes; cooperative bounds. | F-D4-01, F-D4-02 (MEDIUM), F-D4-12, F-D4-16 (LOW) |
| PDC-D4-R006 | Instance ownership guard | **PASS** | In-process and cross-OS-process rejection (`test_r006_*`, M4 killed); hard-kill of the real holder -> new engine READY with `stale_recovered` (probe D). | F-D4-11 (LOW) |
| PDC-D4-R007 | Restart supervision with an OS-process crash proof | **PARTIAL** | A real worker `Popen` is hard-killed and the real supervisor restarts the engine (M7 killed), but the crash detector that asks for the restart is test code; the supervisor is in-process. | F-D4-03 (MEDIUM) |
| PDC-D4-R008 | Failure classification | **PASS** | `tests/unit/hosting/supervisor` passes; startup-phase sets in `classification.py` and `diagnostics.py` are equal (probe Q1); supervisor conflict -> `instance_conflict` label in classifier and metric (Q2); M12 killed. | none |
| PDC-D4-R009 | Lifecycle metrics through the canonical observability port | **PASS** | Six metrics emitted as `hosting.lifecycle.metric` events; all six survive the export bridge when enabled (probe P5); hosting owns no store (test); M6/M12 killed. | F-D4-12, F-D4-13 (LOW) |
| PDC-D4-R010 | Restart preserves configuration/profile digest | **PASS** | Digest identical on every instance and the lease; a drifted engine is rejected before start (`test_r010_*`, M5 killed). Semantics are limited (F-D4-20). | F-D4-20 (INFO) |
| PDC-D4-R011 | InteractionProfile facade | **PASS_SCOPED** | `OD-PDC-SCOPE-02` Option A (authorized exclusion, FRZ-DEBT-06); no `InteractionProfile` symbol in `intergrax/`, `applications/`, `agents/`. | none |
| PDC-D4-R012 | OS service-manager adapters | **PASS_SCOPED** | Same decision; no systemd/launchd/Windows-service code in `intergrax/hosting`. | none |
| PDC-D4-R013 | Host plugins in `HostedApplicationProfile` | **PASS_SCOPED** | Same decision; the profile has no plugin field; only the `hosting.plugin.loaded/failed` event names exist (`contracts/events.py:62-63`). | none |
| PDC-D4-R014 | Stable public hosting API declaration | **PARTIAL** | Manifest, version history and gate exist and fail on added/removed/changed symbols (`test_r014_*`, M8 killed); the fingerprint is blind to `model_config`, validators and bounds, and the gate runs in no pipeline. | F-D4-07 (MEDIUM), F-D4-14 |
| PDC-D4-R015 | Generic `HostDeploymentProfile` carries LKW-specific fields | **PARTIAL** | The three `lkw_*` keys left the generic model and fail closed in `LkwHybridDaemonDeployment`; `business_agents_deploy_enabled` remains, LKW vocabulary now sits in a platform-tier module, the model accepts unknown keys, and the docs still say "planned". | F-D4-08, F-D4-09 (MEDIUM), F-D4-10 (LOW) |
| PDC-D4-R016 | TL-FIX-D implemented in code, plan row stale | **PASS** | `DefaultRunService.bind_execution_adapter` exists (`fastapi_core/runs/default_service.py:39`) and the legal and dispute-sim factories call it (`factory.py:157`, `:136`); plan row `plans/TIER3_APPLICATION_ENVIRONMENT.md:130` says Done with evidence; the architecture says IMPLEMENTED (lines 27, 47, 274). | none (see F-D4-09 for TL-FIX-C) |
| PDC-D4-R017 | No host-owned execution authority | **PASS** | `intergrax/hosting` has no import of execution runtime, agents or applications; `tests/unit/hosting/test_hosting_import_boundaries.py` and all `host_01` tests pass. | none |
| PDC-D4-R018 | Tenant-specific host configuration | **PARTIAL** | `tenant_x_final_fast` passes (campaign meta checks only); composition opens the hosted sandbox under the fixed tenant `harness` (code read, not run against a backend). | F-D4-06 (MEDIUM) |
| PDC-D4-R019 | Sandbox/shadow isolation is not a universal guarantee (docs) | **PASS** | The bound is stated in the architecture (`TIER3_APPLICATION_ENVIRONMENT.md:271,561`) and in the main plan row (`plans/...:71-74`). Three satellite rows still say Done without the bound. | F-D4-09 (part, LOW aspect) |
| PDC-D4-R020 | GR-12 composition under STRICT contract authority | **PASS** | GR-12 139 passed and the whole governance directory 403 passed; STRICT+PRODUCT without revision-bound authority fails closed (`package_wiring.py:108-118`); the lab-compat authority is used only when `production_mode` is false (`harness_host_runtime.py:396-404`). A `no_ci` unit gate on the same composition is red for an unrelated, fail-closed reason (F-D4-19). | F-D4-19 (INFO) |

Tally: PASS 9 (R001, R006, R008, R009, R010, R016, R017, R019, R020), PASS_SCOPED 3 (R011-R013), PARTIAL 8 (R002, R003, R004, R005, R007, R014, R015, R018), FAIL 0.

## 4. ADR-PDC-04 conformance

| ADR item | Status |
| --- | --- |
| Generic lifecycle qualification suite over the real engine, guard and supervisor | Present and passing; blind to cancellation, product drain wiring and a hung readiness probe (F-D4-01, -02, -04) |
| Metrics through the canonical port, no hosting-owned backend | Holds; export is opt-in (F-D4-13), publishes unbounded (F-D4-12) |
| Digest-preserving restart for every hosted application | Holds within the semantics of F-D4-20 |
| Stable public API manifest under COMPAT-X policy | Present; structural only and not wired (F-D4-07, F-D4-14) |
| TL-FIX-C corrected instead of product vocabulary in the generic profile | Partly (F-D4-08) |
| APP-HOST-6/7/9E excluded (`OD-PDC-SCOPE-02` A) | Verified absent (R011-R013) |

## 5. Mutation results (strength of the D4 tests)

Twelve in-memory mutants (plugin `adversarial_scripts/mut_plugin.py`, selected by `IAD4_MUT`; no file edited), each run against the one D4 test file that should detect it. All were killed: M1 forward stop order and M10 no startup rollback and M11 lease not released (R003), M2 no runtime degradation (R004), M3 no cancel and M9 no drain (R005), M4 no exclusive lock (R006), M5 no digest verification (R010), M6 no metrics and M12 constant exit label (R009), M7 never restart (R007), M8 added public method (R014). Observation: with M4 the in-process `test_r006_second_instance_is_rejected...` still passes (the stored PID of the same process triggers the conflict); only the cross-process test proves the OS lock.

## 6. Re-verification of routed items

| Item | Disposition at the audited SHA |
| --- | --- |
| P7-00 F-13 (retry budget redelivers crashed tasks with `idempotency_store=None`) | Confirmed by code read; owner is `intergrax/queueing`, not a D4 row -> F-D4-17 (INFO), route to the queueing/background owner |
| P7-00 F-16 (lifecycle metric publish unbounded in shutdown) | Confirmed by probe and widened to all downstream publishes and the supervisor -> F-D4-12 (LOW) |
| P7-00 F-22 (`HostDeploymentProfile` `extra=allow`) | Confirmed by probe; wide bind is not a regression -> F-D4-10 (LOW) |
| P7-00 F-04 (host composition wiring has no production caller for several hardening capabilities) | Not a D4 row (routed to D6/orchestrator); no D4 claim depends on it |
| Addendum_2 F-41 / T-IGX-040-A root `applications/` impact rule | Rule inspected (`impact_map.py:203-224`): matches only the root `applications/` prefix, repeats existing ids; no D4 row owns such a path; not a D4 defect -> F-D4-18; related gap F-D4-14 |
| Addendum_3/4 F-49 / `openai` catalog dispatch | Integration catalog, outside D4 -> F-D4-18 |
| Addendum_4 F-59 (positive record) | Nothing to re-verify for D4 |

## 7. Known classified reds (not re-reported)

EBH-7 identity gate, the five `roadmap_replay_x` tests, `dist_scale_x`/Docker scope (C10/C15 freeze) and the strict `xfail` `test_scaffold_defaults_mcp_to_opt_in[product]` were not run and are not reported. The `pdc_x` pytest exit hang (debt R-05) did not occur for `tests/qualification/pdc_x/d4` (exit 0 immediately after the summary).

## 8. Findings

### F-D4-01 - MEDIUM - Task cancellation during start() or run_until_stopped() skips all rollback: lease, started components and runtime stay held

- **Rows:** PDC-D4-R003, PDC-D4-R005
- **Verification:** PROBE_CONFIRMED
- **Evidence:** engine.py start() wraps _startup_sequence in `except Exception`; asyncio.CancelledError is a BaseException, so _startup_failure_cleanup never runs. run_until_stopped() has no cleanup path either, and the supervisor loop catches Exception only. Probe P9: a required component whose start() sleeps, task.cancel() after 0.5 s -> journal ['start:core','start:slow'] with no stop, lifecycle state 'starting', instance_lease_acquired=True, instance_lease_released=False, context_closed=False; a second engine in the same run directory is rejected (HostedApplicationStartupError). Probe P10: cancelling run_until_stopped() on a READY engine leaves state 'ready', runtime and components started, lease not released.
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_b.py  (keys P9_cancel_during_startup, P10_cancel_run_until_stopped; output logs/probe_b.out.txt)`
- **Impact:** Any embedding that cancels the host task (wait_for, TaskGroup, a signal path that cancels instead of requesting shutdown) leaks components (for example an owned OS worker), the instance lease and the health task. The D4 rollback and drain proofs cover only Exception-class failures and the cooperative control path. FRZ-REL-04 (cancellation propagates correctly) is listed for R005.
- **Suggested repair scope:** Handle CancelledError in HostedApplicationEngine.start/run_until_stopped (and the supervisor's finally): run the bounded terminal cleanup under asyncio.shield, then re-raise; add negative tests for cancel during startup, during READY and during stop.
- **Owner path:** `intergrax/hosting/engine/engine.py, intergrax/hosting/supervisor/supervisor.py, tests/qualification/pdc_x/d4/`

### F-D4-02 - MEDIUM - The in-flight drain and flush machinery is unreachable from the product host: no product HostedApplicationActiveWorkController and no seam in run_hosted_application

- **Rows:** PDC-D4-R005
- **Verification:** CODE_READ_BY_AUDITOR+PROBE
- **Evidence:** grep over intergrax/ applications/ agents/ scripts/ finds no implementation of stop_intake/wait_for_idle and no non-test `active_work_controller=` argument. runner._build_engine_factory builds HostedApplicationEngine without active_work_controller or flush_services and run_hosted_application() accepts only a profile and an event publisher factory, so in the product runner active_work_before is always 0 and the DRAIN/CANCEL/FLUSH phases are SKIPPED or empty. The only proofs (tests/qualification/pdc_x/d4/test_xd05_drain_background_work.py and the XD-05 integration) use the test-side AsyncioActiveWork adapter; the XD-05 docstring itself states 'the product ships the port but no implementation'. The LKW hosted runtime stop() only sets uvicorn should_exit and states 'platform engine owns shutdown budget'.
- **Reproduction:** `grep -rn 'def wait_for_idle\|active_work_controller=' --include=*.py intergrax applications agents scripts ; read intergrax/hosting/runner.py:216-250 and applications/local_workspace_application/hosting/runtime.py:213-226`
- **Impact:** R005 ('graceful shutdown with in-flight drain') is proven for the engine mechanism with an injected port, not for the product composition; background work of a product host is not drained by the host engine.
- **Suggested repair scope:** Provide a product adapter over the background worker runtime (D7 contract) and a seam in the runner/profile to install an active-work controller and flush services; add a runner-level test that submits work, requests shutdown and asserts drain, cancel and exit class.
- **Owner path:** `intergrax/hosting/runner.py, intergrax/hosting/shutdown.py, intergrax/background_tasks (adapter), tests/qualification/pdc_x/d4/`

### F-D4-03 - MEDIUM - The OS-process crash proof relies on a test-owned crash detector; the product supervisor is in-process and ships no process watcher

- **Rows:** PDC-D4-R007
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** test_d4_crash_restart_subprocess.py defines _SupervisedWorkerRuntime in the test: its _watch() polls the child Popen and calls control.request_restart('worker_process_exited'). HostedApplicationSupervisor ('In-process hosted application supervisor (APP-HOST-5C)') only restarts on the exit of its own engine; nothing in intergrax/hosting observes an OS child or the host process. The proof therefore shows 'a restart request after a worker death is honoured', not that the product detects a crash. A crash of the host process itself has no product supervisor (R012 OS service adapters are an authorized exclusion).
- **Reproduction:** `read tests/qualification/pdc_x/d4/test_d4_crash_restart_subprocess.py:50-97 and intergrax/hosting/supervisor/supervisor.py:1-5; mutation M7 (never restart) is killed, so the restart half is real`
- **Impact:** The R007 wording 'restart supervision with an OS-process crash proof' overstates what the product provides; the claim holds only for applications that bring their own process monitor.
- **Suggested repair scope:** Either narrow the claim to 'supervised restart on a control-plane restart request, proven with an OS worker kill', or add a product component/port that watches an owned child process and requests the restart, plus a crash proof of the host process through an external launcher script.
- **Owner path:** `intergrax/hosting/supervisor/, intergrax/hosting/contracts/components.py, tests/qualification/pdc_x/d4/test_d4_crash_restart_subprocess.py, PDC_DOMAIN_REQUIREMENTS_MATRIX R007 wording`

### F-D4-04 - MEDIUM - A hung runtime.ready() leaves the readiness snapshot reporting ready indefinitely (stale, no health_evaluation_failed)

- **Rows:** PDC-D4-R004
- **Verification:** PROBE_CONFIRMED
- **Evidence:** engine._refresh_health awaits self._runtime.ready(context) without a timeout (components have health_timeout_seconds, the runtime does not). Probe F: after READY, ready() made to hang and the runtime marked not started; after 1.5 s and 1.0 s health_snapshot().ready stays True, health_evaluation_failed False, last_evaluated_at unchanged. Shutdown still returns because stop_polling is bounded.
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_f.py  (output logs/probe_f.out.txt)`
- **Impact:** A wedged runtime is advertised as ready and keeps accepting new work (accepting_new_work stays True).
- **Suggested repair scope:** Bound runtime.ready (and the refresh as a whole) with a configurable timeout; on timeout mark_health_evaluation_failed so ready turns False; add a hung-probe test.
- **Owner path:** `intergrax/hosting/engine/engine.py (_refresh_health), intergrax/hosting/engine/health.py, tests/qualification/pdc_x/d4/`

### F-D4-05 - MEDIUM - 'Single effective composition root' does not hold: the lab host factory builds a second tool wiring beside wire_application_environment, and no gate asserts the single root

- **Rows:** PDC-D4-R002
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** applications/lab_application/host/factory.py:225 calls wire_lab_tools() (host/tool_wiring.py -> build_application_tool_wiring with a hard-coded sandbox tenant 'lab') when include_mcp is set and passes that registry to build_lab_mcp_server, independent of the registry produced by wire_application_environment. wire_application_environment also accepts conformance_check=False (skips registry/graph/package/tool conformance) and ProfileInvariantValidator(fail_on_violation=False). The matrix cites host_01, which does not mention wire_application_environment (see F-D4-15); no test or script gate forbids a second composition path.
- **Reproduction:** `read applications/lab_application/host/factory.py:218-236, applications/lab_application/host/tool_wiring.py:96-117, intergrax/applications/_shared/environment_wiring.py:542-568`
- **Impact:** The tools exposed over MCP by the lab host are not the tools the composition root validated; the 'single root' claim is not enforceable as stated.
- **Suggested repair scope:** Route the lab MCP path through the composition root's tool wiring (or declare and gate the exception); add a gate that product host factories do not call build_application_tool_wiring; document the conformance_check opt-out as lab-only.
- **Owner path:** `applications/lab_application/host/factory.py, intergrax/applications/_shared/environment_wiring.py, a new gate under tests/qualification/pdc_x/d4/`

### F-D4-06 - MEDIUM - Composition opens the hosted sandbox session under the fixed tenant 'harness' and task 'bootstrap', whatever tenant_id the host is composed for

- **Rows:** PDC-D4-R018
- **Verification:** CODE_READ_BY_AUDITOR (not reproduced against a live sandbox backend)
- **Evidence:** environment_wiring.py:438-448: when no sandbox session is passed and an integration profile exists, resolve_hosted_sandbox_session(..., tenant_id='harness', task_id='bootstrap') is called and the session is handed to build_application_tool_wiring for every tenant. HostedSandboxSession.open is a cloud session. The tenant_id parameter of wire_application_environment is not used for it.
- **Reproduction:** `read intergrax/applications/_shared/environment_wiring.py:438-476 and intergrax/applications/_shared/sandbox_host_wiring.py`
- **Impact:** With sandbox_host configured, tenant-specific host composition shares one sandbox identity across tenants (R018 TENANT_IMPACT is HIGH); not verified against a real backend.
- **Suggested repair scope:** Derive the sandbox tenant/task from the composition tenant_id (fail closed when absent) or open the session lazily per tool invocation; add a tenant-isolation test with a fake SandboxHostBackend.
- **Owner path:** `intergrax/applications/_shared/environment_wiring.py, intergrax/applications/_shared/sandbox_host_wiring.py`

### F-D4-07 - MEDIUM - The public hosting API fingerprint ignores model_config, validators and Field constraints, so a contract loosening passes the COMPAT gate

- **Rows:** PDC-D4-R014
- **Verification:** PROBE_CONFIRMED
- **Evidence:** describe_public_symbol() records bases, own public member signatures and model field annotation/required only. Probe P2: two pydantic models with the same field annotation but (extra=forbid, ge=1, le=5, frozen) versus (extra=allow, ge=-100, le=500) get an identical fingerprint; flipping HostedApplicationProfile.model_config['extra'] from 'forbid' to 'allow' leaves its fingerprint unchanged. This is the exact class of change made to HostDeploymentProfile in the same wave (F-D4-10). Positive: structural changes are caught (mutation M8, a new public method, is killed).
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_a.py  (keys P2_*; output logs/probe_a.out.txt)`
- **Impact:** HOSTING_PUBLIC_API_VERSION=1.0.0 does not guarantee behavioural stability of public models (strictness, bounds, cross-field validators).
- **Suggested repair scope:** Include model_config, field constraints (metadata) and validator names in describe_public_symbol; regenerate the manifest with a new version entry; add a test that loosening extra/bounds changes the fingerprint.
- **Owner path:** `intergrax/hosting/public_api.py, tests/qualification/pdc_x/d4/test_d4_public_api_manifest.py`

### F-D4-08 - MEDIUM - TL-FIX-C is closed only for the three lkw_* keys: business_agents_deploy_enabled stays typed on the generic profile, the LKW vocabulary moved into a platform-tier module, and the guard vocabulary cannot see the remaining key

- **Rows:** PDC-D4-R015
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** The architecture table TIER3_APPLICATION_ENVIRONMENT.md:179-182 lists four keys as product-owned (lkw_hybrid_daemon_enabled, lkw_daemon_bind_host, lkw_daemon_port, business_agents_deploy_enabled). B-D4 moved three; sub_profiles.py:517 still declares business_agents_deploy_enabled, consumed by intergrax/applications/_shared/business_agent_deploy_wiring.py and enabled by the product preset (presets.py:342). The new LkwHybridDaemonDeployment model lives in intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py (platform tier), while the plan row 129 requires 'application-owned or typed extension-owned' configuration. test_d4_no_product_vocabulary.py uses PRODUCT_VOCABULARY=('lkw','daemon','local_workspace',...), which by construction cannot flag 'business_agents'.
- **Reproduction:** `read docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md:171-184, intergrax/applications/contracts/environment_profile/sub_profiles.py:506-525, tests/qualification/pdc_x/d4/test_d4_no_product_vocabulary.py:31-32`
- **Impact:** R015 is at best partially closed by the repository's own TL-FIX-C definition; a regression test would pass although a documented product key remains.
- **Suggested repair scope:** Decide ownership of business_agents_deploy_enabled (move to a product/extension-owned model or record it as platform vocabulary with an ADR note); move LkwHybridDaemonDeployment under applications/local_workspace_application or a declared typed extension; align the guard vocabulary with the table.
- **Owner path:** `intergrax/applications/contracts/environment_profile/sub_profiles.py, intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py, tests/qualification/pdc_x/d4/test_d4_no_product_vocabulary.py`

### F-D4-09 - MEDIUM - Documentation status is stale after B-D4: TL-FIX-C still 'ACCEPTED / PLANNED / not fixed' in the architecture and the plan; APP-CON-8 'Done' rows in the satellite plan carry no isolation bound

- **Rows:** PDC-D4-R015, PDC-D4-R019
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** TIER3_APPLICATION_ENVIRONMENT.md (architecture) lines 27, 47, 171-173, 273, 282, 297, 601, 619 and the plan row at plans/TIER3_APPLICATION_ENVIRONMENT.md:129 still describe TL-FIX-C as planned/not fixed and 'A4, not A5 while TL-FIX-C remains accepted'. The same wave reconciled TL-FIX-D (R016, a freeze-blocker row) for exactly this plan/code mismatch. For R019 the bounded claim ('not a universal production isolation guarantee') is present in the architecture (lines 271, 561) and in the main plan row (plans/TIER3...:71-74) but not in the satellite plans/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md rows 11, 12, 60 which mark shadow/sandbox lifecycle 'Done'.
- **Reproduction:** `grep -n 'TL-FIX-C' docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md ; grep -n 'APP-CON-8' docs/project/maintainers/plans/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md`
- **Impact:** FRZ-DOC-01 style plan/code mismatch re-introduced; readers get contradictory maturity statements (A4 vs implemented).
- **Suggested repair scope:** After F-D4-08 is decided, update status rows, the A-level text and the key table in architecture and plan; add the isolation bound (or a pointer) to the three satellite rows.
- **Owner path:** `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md, docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md, docs/project/maintainers/plans/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md`

### F-D4-10 - LOW - HostDeploymentProfile changed from extra='forbid' to extra='allow': typos in generic keys are silently accepted and a non-loopback bind host is not constrained

- **Rows:** PDC-D4-R015
- **Verification:** PROBE_CONFIRMED (dossier F-22 re-verified)
- **Evidence:** Probe P1: HostDeploymentProfile(business_agents_deploy_enabld=True) is accepted into model_extra while the typed flag stays False; LkwHybridDaemonDeployment.from_host_deployment accepts lkw_daemon_bind_host='0.0.0.0'; a misspelled lkw_ key (lkw_daemon_prot) and nested values are rejected. The previous model (git show 145b49bad^) was extra='forbid' but also had no bind-host validator, so the wide bind is not a regression. test_r015_generic_profile_accepts_only_flat_scalar_product_keys pins the new permissive behaviour.
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_a.py  (keys P1_*)`
- **Impact:** A mistyped deployment switch is ignored without diagnostics; a daemon can be bound to all interfaces by configuration (default stays 127.0.0.1).
- **Suggested repair scope:** Require product-owned keys to match a registered prefix and fail on unknown unprefixed keys (or expose a typed extension registry); validate bind_host against loopback unless an explicit opt-in flag is set.
- **Owner path:** `intergrax/applications/contracts/environment_profile/sub_profiles.py, intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py`

### F-D4-11 - LOW - Instance guard: stale lock metadata whose PID now belongs to an unrelated live process rejects a legitimate start (PID reuse)

- **Rows:** PDC-D4-R006
- **Verification:** PROBE_CONFIRMED
- **Evidence:** file_guard._classify_prior_metadata raises 'live_process_metadata' when prior.process_id is alive, although the exclusive OS lock was acquired (so the prior owner no longer holds it); process_started_at is stored but never compared. Probe P3: lock file with metadata of pid=<this live pid>, started_at=2020, no lock held -> acquire() raises HostedApplicationInstanceConflictError (reason live_process_metadata). Positive: probe D (hard-kill of the real lease holder process) -> a new engine starts READY with hosting.instance.stale_recovered.
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_a.py (P3) and probe_d.py`
- **Impact:** After a crash on a host with fast PID reuse (Windows) the restart can be refused until the metadata file is removed manually; fail-closed, no safety violation.
- **Suggested repair scope:** Treat metadata as live only if both pid and process_started_at match the live process; add a PID-reuse test with a fake probe.
- **Owner path:** `intergrax/hosting/instance/file_guard.py, tests/unit/hosting/instance/`

### F-D4-12 - LOW - Downstream event/metric publishes are awaited without a bound in the shutdown path (engine and supervisor)

- **Rows:** PDC-D4-R009, PDC-D4-R005
- **Verification:** PROBE_CONFIRMED (dossier F-16 re-verified and widened)
- **Evidence:** engine._publish_metric awaits the dispatcher directly; _graceful_stop_sequence also awaits the APPLICATION_STOPPING publish and _publish_exit_class_metric outside run_bounded_phase (only the terminal stopped event is bounded). HostingEventDispatcher.publish awaits the downstream publisher with no timeout, and the observability export policy has no timeout. The supervisor's _publish_metric_safe and _publish_restart_event_safe are equally unbounded. Probe P11: a publisher that hangs on the drain_duration metric -> engine.stop() did not return within 60 s (computed budget 150 s; every bounded phase finished in < 1 s).
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_b.py  (key P11_hanging_metric_exporter_in_stop)`
- **Impact:** A hanging exporter or custom publisher can block shutdown and restart past the declared budget.
- **Suggested repair scope:** Publish through run_bounded_phase/asyncio.wait_for using the observer timeout in the engine and supervisor; record a secondary failure on timeout; add a hanging-publisher test.
- **Owner path:** `intergrax/hosting/engine/engine.py, intergrax/hosting/eventing.py, intergrax/hosting/supervisor/supervisor.py`

### F-D4-13 - LOW - Lifecycle metrics are exported only when the host enables an observability policy; the product runner default exports nothing, and the bridge test covers 3 of 6 metrics

- **Rows:** PDC-D4-R009
- **Verification:** PROBE_CONFIRMED
- **Evidence:** run_hosted_application() defaults to ObservabilityHostedApplicationEventPublisher() whose ObservabilityExportPolicy is enabled=False. Probe P6: all six metric events published through a default publisher produce 0 exported envelopes; with enabled=True all six survive with name and label (probe P5). test_r009_metrics_reach_the_canonical_observability_export_bridge asserts only start, ready, drain_duration. No test composes the product runner with an enabled exporter.
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_a.py  (keys P5_*, P6_*)`
- **Impact:** R009 holds at the event-port level; a default product host emits the events but exports no metric until the operator wires an exporter.
- **Suggested repair scope:** Assert all six metrics through the bridge; add a runner-level test with an enabled InMemory exporter; document the opt-in in the hosting architecture.
- **Owner path:** `intergrax/hosting/runner.py, tests/qualification/pdc_x/d4/test_d4_lifecycle_metrics.py`

### F-D4-14 - LOW - The D4 qualification suites and host_01 are not wired into any CI workflow, regression profile or ENT-AUDIT impact rule

- **Rows:** PDC-D4-R014, PDC-D4-R003, PDC-D4-R005
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** .github/workflows/*.yml run tests/unit (-m 'gate and not no_ci'), ent_audit_x and diagnostics suites only; no workflow or scripts/ci/run_regression_gate_ci.py profile references tests/qualification/pdc_x/d4 or tests/qualification/host_01. The ENT-AUDIT impact map maps intergrax/hosting to ENT-A-* structural checks, none of which runs the D4 suite or the public-API gate (PDC_TIER3_ROOT_IMPACT_RULES covers the root applications/ prefix only, as intended by T-IGX-040-A).
- **Reproduction:** `grep -n 'pdc_x\|host_01\|tests/qualification' .github/workflows/*.yml scripts/ci/run_regression_gate_ci.py ; read intergrax/platform/enterprise_audit/impact_map.py:77-90`
- **Impact:** The R014 version gate and the lifecycle regressions are protected only when a person replays the campaign suites.
- **Suggested repair scope:** Add the D4 directory (or the public-API test) to the smoke/enterprise regression profile and a hosting impact rule that selects it.
- **Owner path:** `scripts/ci/run_regression_gate_ci.py, .github/workflows/unit-tests.yml, intergrax/platform/enterprise_audit/impact_map.py`

### F-D4-15 - LOW - The cited evidence 'host_01' does not exercise R001 or R002 (the closed-world inventory itself labels it 'execution boundary only')

- **Rows:** PDC-D4-R001, PDC-D4-R002
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** grep -c over tests/qualification/host_01/*.py for HostedApplicationProfile, ApplicationEnvironmentProfile and wire_application_environment returns 0 in every file; HOST-Q1..Q12 cover the task execution port, root launcher, error mapping and transport purity. R001 is independently supported by tests/unit/hosting/test_hosted_application_profile*.py (passed) and tests/unit/applications/test_environment_profile.py (passed); R002 has no direct gate (see F-D4-05).
- **Reproduction:** `grep -c 'HostedApplicationProfile\|ApplicationEnvironmentProfile\|wire_application_environment' tests/qualification/host_01/*.py`
- **Impact:** 'Closure via PDC-P7-D4 replay of host_01' cannot evidence these two rows; the matrix EVIDENCE_PATH is generic for every D4 row.
- **Suggested repair scope:** Map R001 to the hosting profile and environment profile suites and add a single-root gate for R002; correct the matrix TEST_CASE.
- **Owner path:** `docs/project/maintainers/qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.json (R001, R002, R017 rows)`

### F-D4-16 - LOW - run_bounded_phase is cooperative: work that swallows cancellation overruns its limit and is still reported COMPLETED

- **Rows:** PDC-D4-R005
- **Verification:** PROBE_CONFIRMED
- **Evidence:** Probe P7: limit 0.2 s, work that sleeps 2 s inside `except CancelledError` -> outcome 'completed' after 2.22 s. A TimeoutError raised by the work itself is also classified TIMED_OUT.
- **Reproduction:** `uv run --extra mcp python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/adversarial_scripts/probe_a.py  (key P7_*)`
- **Impact:** The 'bounded shutdown' guarantee depends on well-behaved controllers/components; a stubborn one defeats the budget silently.
- **Suggested repair scope:** Document the cooperative contract; optionally race the phase against a deadline with asyncio.wait and abandon the task, recording FORCED.
- **Owner path:** `intergrax/hosting/shutdown.py`

### F-D4-17 - INFO - Dossier F-13 confirmed at this SHA but it is not a D4 requirement row: queue recovery redelivers crash-interrupted tasks while the worker passes idempotency_store=None

- **Rows:** none (routed/observational)
- **Verification:** CODE_READ_BY_AUDITOR (dossier F-13 re-verified)
- **Evidence:** intergrax/queueing/providers/document_store/colocated_worker.py:83-86 (recover_interrupted_running with the RetryPolicy budget) and :141 (idempotency_store=None); document_store_task_queue.recover_interrupted_running settles against the budget instead of FAILED when max_retries is set. Owner is the queueing/background domain.
- **Reproduction:** `read the two files at the cited lines`
- **Impact:** Opt-in at-least-once redelivery without an idempotency ledger can double-run side effects (default without a policy keeps FAILED).
- **Suggested repair scope:** Route to the owner of intergrax/queueing (D7); not repaired under D4.
- **Owner path:** `intergrax/queueing/providers/document_store/colocated_worker.py`

### F-D4-18 - INFO - Dossier F-41 (impact-map root applications/ rule, T-IGX-040-A) and F-49 (openai catalog dispatch) are outside the D4 rows

- **Rows:** none (routed/observational)
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** PDC_TIER3_ROOT_IMPACT_RULES (impact_map.py:203-224) matches only paths under the root applications/ prefix and repeats existing ENT-A ids; no D4 row owns a root-applications path in its write set, and the D4 owner paths are under intergrax/hosting and intergrax/applications (mapped by their own prefixes). F-49 concerns intergrax/integrations catalog dispatch, not hosting or environment composition. Both were therefore not judged as D4 defects; F-D4-14 notes that the impact map selects no D4 suite.
- **Reproduction:** `read intergrax/platform/enterprise_audit/impact_map.py:60-90 and 196-224`
- **Impact:** None for D4.
- **Suggested repair scope:** None under D4.
- **Owner path:** `n/a`

### F-D4-19 - INFO - Environment and out-of-row red tests observed during the audit

- **Rows:** none (routed/observational)
- **Verification:** TEST_RUN
- **Evidence:** (a) A clean worktree .venv (default groups test+quality) lacks fastmcp: tests/qualification/host_01 fails at collection until `--extra mcp` is used; psycopg is absent (4 tests in tests/unit/applications/architecture fail with IntegrationConfigurationError; a fifth, test_npsc1_*, fails with OrchestrationGovernanceEvidenceCompositionError 'requires governance_evidence_persistence'). (b) 3 tests in tests/unit/hosting/instance (symlink rejection) fail with WinError 1314 (no symlink privilege) instead of skipping. (c) tests/unit/applications/test_package_wiring.py::test_strict_product_package_closure_passes (marked no_ci) fails with MemoryProviderAdmissionError qualification_evidence_missing (fail-closed: no evidence registry is passed); two further architecture gates are stale (test_gr10_r10_r2_missing_generic_production_policy_still_fail_closed patches a removed attribute; test_npsc3g_worker_runtime_uses_host_execution asserts text no longer in nexus_worker_execution.py). None of these was compared with the activation pin; none is in a D4 row.
- **Reproduction:** `logs/host01_unit_hosting.log, logs/env_profile_unit.log`
- **Impact:** Replays on a clean Windows checkout need the mcp and postgresql extras and symlink privilege; stale gates add noise to unit runs.
- **Suggested repair scope:** Document the extras in the gate runbook, make the symlink tests skip when privilege is missing, retarget or retire the stale gates (owners outside D4).
- **Owner path:** `tests/unit/hosting/instance/, tests/unit/applications/architecture/, tests/unit/applications/test_package_wiring.py`

### F-D4-20 - INFO - R010 semantics: restart never re-resolves configuration; the digest check compares the engine definition's stored digest strings

- **Rows:** PDC-D4-R010
- **Verification:** CODE_READ_BY_AUDITOR
- **Evidence:** HostedApplicationSupervisor holds one immutable definition and _verify_engine_contract compares engine.definition.profile_digest/definition_digest with it. The digest covers application_factory_id (a string), not the callable. The check rejects an engine composed from a different definition (mutation M5 killed) but cannot detect an engine whose definition object was built from the same digest with different code.
- **Reproduction:** `read intergrax/hosting/supervisor/supervisor.py:396-428 and intergrax/hosting/contracts/profile.py:183-190`
- **Impact:** 'Restart preserves configuration' means 'the supervisor refuses a mismatching definition digest', which is what the test proves.
- **Suggested repair scope:** None required; keep the wording of R010 aligned with this semantics.
- **Owner path:** `n/a`


## 9. Overall verdict

**PARTIAL.** The engine, guard, supervisor, classifier, metric and digest mechanisms are real, narrowly tested and sensitive to regression (12 of 12 mutants killed; the D4 suite and GR-12/governance replays are green). The claim breaks where the product composition meets the mechanism: no drain/flush seam in the product runner, no cancellation rollback, an unbounded readiness probe and unbounded publishes, a test-owned crash detector, a second tool composition path, an incomplete TL-FIX-C, a COMPAT gate that cannot see contract loosening, and stale TL-FIX-C documentation. No finding is HIGH; none is a security exposure reproduced against a live backend (F-D4-06 is a code-read finding). Three rows are PASS_SCOPED by the Operator exclusion `OD-PDC-SCOPE-02`.

Suggested repair grouping for the owning ticket(s): (a) engine lifecycle hardening - F-D4-01, F-D4-04, F-D4-12, F-D4-16 with tests; (b) product wiring - F-D4-02, F-D4-03, F-D4-05, F-D4-06; (c) contract and gate quality - F-D4-07, F-D4-08, F-D4-10, F-D4-14, F-D4-15; (d) documentation - F-D4-09. A fresh auditor re-audits after the repair (loop 1 of 2).

## 10. Limitations - what was not covered

- Windows only. The POSIX `fcntl` branch of `_native_lock.py` and Linux/macOS PID semantics were not exercised. The three symlink-rejection tests could not run (no privilege), so symlink hardening of the guard was read, not executed.
- F-D4-06 is a code-read finding; no live sandbox backend or second tenant was composed.
- `hooks`, `event subscriptions`, `bootstrap_failure`, `process_bootstrap`, `signals` and `control` modules were read only where the rows touch them; the signal bridge and the guarded bootstrap were not attacked.
- The LKW product hosting (`applications/local_workspace_application/hosting`) was read only for `stop()`/`ready()`; no end-to-end LKW run and no review of `scripts/proof/dg001d_r4_*` prior proofs.
- `tenant_x` was replayed through `test_tenant_x_final_fast.py` only; EBH-7, roadmap replay, dist_scale_x, the full `tests/unit` tree, CI contexts, network and live-provider paths were not run.
- The eight out-of-row unit failures (F-D4-19) were not compared with the activation pin.
- No performance, soak, memory or fuzz testing; concurrency was exercised only by the probes listed.
- Documentation was checked only for the TIER3 architecture/plan/satellite statements named in R015, R016 and R019; the wider documentation canon and G-DOC gates were not audited.
- No certification, FRZ promotion, closure or production-readiness claim is made; `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the manifest are placeholders for a later certifier.

## 11. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/D4/`: `audit_scope.json`, `test_matrix.txt`, `iiac_audit_d4_findings.json`, `iiac_audit_session_manifest.PDC-X.d4.json`, `EVIDENCE_MANIFEST.txt` (sha256 of every file), `logs/` (suite logs, probe outputs, `mutants/`, `NOTES.txt`), `adversarial_scripts/` (probes, mutation plugin and runner).
