# APPLICATION_HOSTING — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** APPLICATION_HOSTING
- **Tier(s):** Tier-0 `intergrax/hosting/` — `HostedApplicationEngine`, `HostedApplicationSupervisor`, instance guard, shutdown/exit classification, `run_hosted_application(profile)`
- **audited_sha:** `a323dfa7a95292725a925a8b4c4370adc947adf7`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/APPLICATION_HOSTING.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/APPLICATION_HOSTING.md`
- **Scope in:**
  - `run_hosted_application(profile)` public foreground runner and `_default_runner_factories()`
  - `ObservabilityHostedApplicationEventPublisher` / canonical Observability export path
  - component lifecycle health and `ComponentFailureAction` enforcement after READY
  - `ComponentCoordinator` / `stop_started()` shutdown aggregation and exit classification
  - `HostedApplicationShutdownExecutor` FLUSH phase vs `HostedApplicationExitClassifier` critical cleanup set
  - `FileHostedApplicationInstanceGuard` stale ownership and PID/process metadata
  - historical APP-HOST **Done** delivery (W1/W2/W3, LKW 8C–8E, 9A) as positive controls
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - second hosting runtime or private hosting event bus
  - rewriting historical Done rows or universal production qualification claims
  - systemd/Kubernetes/service-manager ownership
- **Prior audit reference(s):** [`OBSERVABILITY_EVIDENCE`](OBSERVABILITY_EVIDENCE.md); [`TIER3_APPLICATION_ENVIRONMENT`](TIER3_APPLICATION_ENVIRONMENT.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `2ca992940f10e998afe819eaa02eaa1ca71cf8a0`

## Executive summary

**Verdict: FAIL.** Four accepted HIGH and two accepted MEDIUM findings show that canonical public foreground hosting can emit lifecycle evidence into a default NoOp Observability exporter; runtime component failure policy is not re-evaluated after READY; component stop failures can be hidden behind COMPLETED shutdown phases and clean terminal classification; required durability flush failure does not block CLEAN_STOP; and stale instance recovery can misclassify PID reuse as live ownership conflict while plan current-state header drifted from shipped W1/W2/W3 facts. Positive controls: Hosting owns lifecycle not task execution; restart remains distinct from task retry; local supervision remains distinct from ECP; engine/supervisor separation; real file-lock instance guard; LKW live proof intact; InteractionProfile/plugins/OS adapters remain honestly **PLANNED**; maturity remains **A4/I4/P3/E4** unless independently re-assessed. Remediation is **PLANNED**, not implemented. Findings harden existing Hosting — no second hosting runtime required.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-APPLICATION_HOSTING-01

- **Severity:** HIGH
- **Category:** OBSERVABILITY / EVIDENCE DISCONNECT
- **Status at publication:** ACCEPTED
- **Remediation block:** HOSTING-SHUTDOWN-EVIDENCE-INTEGRITY
- **Claim falsified:** Public production-capable hosting reuses one configured canonical Observability export authority; lifecycle evidence is not silently discarded via default NoOp exporter.
- **Observation:** Public `run_hosted_application(profile)` uses `_default_runner_factories()`. Its event publisher factory is `ObservabilityHostedApplicationEventPublisher`. But `ObservabilityHostedApplicationEventPublisher.__init__()` defaults to `NoOpObservabilityExporter()` when no exporter is explicitly supplied. The public `run_hosted_application(profile)` API exposes no exporter parameter; the injectable `_RunnerFactories` path is private/test-oriented. Therefore canonical public foreground hosting can emit all hosting lifecycle events into a NoOp exporter and retain/export no lifecycle evidence.
- **Location:**
  - `intergrax/hosting/runner.py` — `run_hosted_application()`, `_default_runner_factories()`
  - `intergrax/hosting/eventing.py` — `ObservabilityHostedApplicationEventPublisher`
- **Reproduction:** Run `run_hosted_application(profile)` without injecting factories; observe publisher constructed with default NoOp exporter while lifecycle events publish successfully into void.
- **Impact:** Hosted lifecycle evidence absent from canonical Observability spine in default public path; operational/debug/restart forensics unreliable.
- **Confidence:** CONFIRMED

### AUDIT-20260818-APPLICATION_HOSTING-02

- **Severity:** HIGH
- **Category:** RUNTIME HEALTH / POLICY ENFORCEMENT
- **Status at publication:** ACCEPTED
- **Remediation block:** HOSTING-RUNTIME-HEALTH-INTEGRITY
- **Claim falsified:** `ComponentFailureAction` applies across the entire component lifecycle; every post-READY health transition maps through the effective registered action.
- **Observation:** `ComponentFailureAction` is correctly applied to startup/dependency failures. After READY, periodic component health refresh only replaces health snapshots. Runtime health changes do not re-evaluate effective `FAIL_HOST`, `MARK_NOT_READY`, `MARK_DEGRADED`, or `IGNORE_WITH_DIAGNOSTIC`. `HostedApplicationHealthCoordinator` blocks unhealthy required components and members of the startup-derived mark_not_ready set, but does not evaluate each registration's current `failure_action`. Therefore an optional component configured `FAIL_HOST` can fail after READY without failing the host, and runtime `MARK_DEGRADED` / `MARK_NOT_READY` semantics are not authoritatively enforced.
- **Location:**
  - `intergrax/hosting/engine/health.py` — `HostedApplicationHealthCoordinator`
  - `intergrax/hosting/engine/components.py` — runtime health refresh
  - `intergrax/hosting/contracts/components.py` — `ComponentFailureAction`
- **Reproduction:** Register optional component with `FAIL_HOST`; reach READY; simulate post-READY health failure; observe host remains live/READY without canonical hosted failure path.
- **Impact:** Registered failure policy is startup-only in practice; degraded/not-ready/fail-host semantics drift from author intent after READY.
- **Confidence:** CONFIRMED

### AUDIT-20260818-APPLICATION_HOSTING-03

- **Severity:** HIGH
- **Category:** SHUTDOWN INTEGRITY / FALSE SUCCESS
- **Status at publication:** ACCEPTED
- **Remediation block:** HOSTING-SHUTDOWN-EVIDENCE-INTEGRITY
- **Claim falsified:** Component shutdown is best-effort but outcome-truthful; failed component stop cannot masquerade as COMPLETED aggregate phase success.
- **Observation:** `ComponentCoordinator._stop_one()` catches `component.stop()` exceptions, records secondary diagnostics, then discards the component from `_started` and publishes `COMPONENT_STOPPED`. `_stop_level()` gathers with `return_exceptions=True` and does not aggregate per-component failures. `stop_started()` therefore normally returns even when one or more real component stops failed. Engine wraps `stop_started()` with `run_bounded_phase`; because no exception escapes, shutdown records `COMPONENT_STOP` as COMPLETED. Exit classifier only treats `COMPONENT_STOP` as critical when the phase outcome is FAILED. A failed component stop can therefore be hidden behind COMPLETED and permit a clean terminal classification.
- **Location:**
  - `intergrax/hosting/engine/components.py` — `_stop_one()`, `_stop_level()`, `stop_started()`
  - `intergrax/hosting/engine/engine.py` — bounded shutdown phase wrapping
  - `intergrax/hosting/supervisor/classification.py` — critical cleanup evaluation
- **Reproduction:** Inject failing `component.stop()` during shutdown; observe `COMPONENT_STOP` phase COMPLETED and terminal CLEAN_STOP despite stop failure diagnostics.
- **Impact:** Shutdown success and restart policy may treat partially failed cleanup as clean stop.
- **Confidence:** CONFIRMED

### AUDIT-20260818-APPLICATION_HOSTING-04

- **Severity:** HIGH
- **Category:** DURABILITY / CLEAN-SHUTDOWN TRUTHFULNESS
- **Status at publication:** ACCEPTED
- **Remediation block:** HOSTING-SHUTDOWN-EVIDENCE-INTEGRITY
- **Claim falsified:** `CLEAN_STOP` requires required shutdown durability obligations to succeed; required flush failure makes terminal outcome non-clean.
- **Observation:** `HostedApplicationShutdownExecutor` records FLUSH phase outcomes for all configured flush services. A failed flush produces `FLUSH=FAILED`. However `HostedApplicationExitClassifier`'s critical cleanup set contains only `COMPONENT_STOP`, `RUNTIME_STOP`, and `LEASE_RELEASE`. FLUSH is excluded. A failed flush does not automatically set shutdown `forced` or `timed_out`. Thus failed trace/event/checkpoint flush may coexist with terminal `STOPPED` → `CLEAN_STOP`.
- **Location:**
  - `intergrax/hosting/shutdown.py` — `HostedApplicationShutdownExecutor`
  - `intergrax/hosting/supervisor/classification.py` — `HostedApplicationExitClassifier`
- **Reproduction:** Configure required flush service that fails; complete shutdown otherwise; observe FLUSH=FAILED with terminal CLEAN_STOP classification.
- **Impact:** Durability loss invisible in exit evidence and restart policy; cross-link [`OBSERVABILITY_EVIDENCE`](../../project/architecture/OBSERVABILITY_EVIDENCE.md) where journal durability overlaps.
- **Confidence:** CONFIRMED

### AUDIT-20260818-APPLICATION_HOSTING-05

- **Severity:** MEDIUM
- **Category:** INSTANCE OWNERSHIP / PID REUSE
- **Status at publication:** ACCEPTED
- **Remediation block:** HOSTING-INSTANCE-RECOVERY-INTEGRITY
- **Claim falsified:** Stale process metadata distinguishes process incarnation, not PID alone; native lock remains authoritative without false ownership conflicts from PID reuse.
- **Observation:** File lease metadata stores `process_id`, `process_started_at`, `host_id`, `user_scope_id`, and `ownership_token`. After acquiring the native lock, stale-owner classification tests only `process_probe.is_alive(prior.process_id)`. If that PID has been reused by an unrelated live process, stale metadata is classified as a live ownership mismatch even though this process successfully acquired the exclusive application lock. The default runner also sets `process_started_at` to current hosting-runner time, not a verified OS process birth/incarnation identifier.
- **Location:**
  - `intergrax/hosting/instance/file_guard.py` — stale ownership classification
  - `intergrax/hosting/runner.py` — default `process_started_at` assignment
- **Reproduction:** Simulate stale lease metadata whose PID was reused by unrelated live process after original owner exit; observe false live-ownership mismatch despite successful exclusive lock acquisition.
- **Impact:** Stale recovery may reject valid lock holder or mis-route instance conflict handling on PID reuse platforms.
- **Confidence:** CONFIRMED

### AUDIT-20260818-APPLICATION_HOSTING-06

- **Severity:** MEDIUM
- **Category:** DOCUMENTATION / PLAN STATE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** HOSTING-PLAN-STATE-INTEGRITY
- **Claim falsified:** Plan current-state header agrees with authoritative APP-HOST row register and architecture shipped posture.
- **Observation:** APPLICATION_HOSTING plan header currently states: "Public hosting foundation and single-instance engine foundation complete. Process control, real instance ownership and supervision not started." The same plan later marks APP-HOST-4A..4E Done, APP-HOST-5A..5C Done, APP-HOST-W3 Done, APP-HOST-9A Done, and APP-HOST-8C..8E Done; architecture correctly describes control, instance ownership, and in-process supervision as shipped.
- **Location:**
  - `docs/project/maintainers/plans/APPLICATION_HOSTING.md` — current-status header vs row register
  - `docs/project/architecture/APPLICATION_HOSTING.md` — shipped W1/W2/W3 posture
- **Reproduction:** Compare plan header with APP-HOST-4/5/8/9 Done rows and architecture maturity notes.
- **Impact:** Operators and agents misread shipped hosting scope; backlog vs delivered W3 conflated.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Hosting owns lifecycle, not task execution | NOT falsified |
| Restart remains distinct from task retry | NOT falsified |
| Local supervision remains distinct from ECP autoscaling | NOT falsified |
| Engine owns one lifecycle; Supervisor owns repetition | NOT falsified |
| Supervisor creates fresh `instance_id` per attempt | NOT falsified |
| Supervisor verifies engine application/profile/definition identity | NOT falsified |
| Supervisor blocks restart when prior lease/context cleanup unverified | NOT falsified |
| File guard uses real native exclusive locking, not PID file alone | NOT falsified |
| Ownership token verified on lease release | NOT falsified |
| Lifecycle transition state machine mechanically validated | NOT falsified |
| Readiness checks runtime ready + valid lease + lifecycle + required components | NOT falsified |
| Global shutdown uses monotonic bounded budget | NOT falsified |
| Runtime-stop failure visible as critical cleanup failure | NOT falsified |
| Lease-release failure visible as critical cleanup failure | NOT falsified |
| InteractionProfile/plugins/OS service adapters remain honestly PLANNED | NOT falsified |
| Application Hosting does not claim systemd/Kubernetes ownership | NOT falsified |
| Current architecture maturity remains A4/I4/P3/E4 unless independently re-assessed | NOT falsified |
| Findings harden existing Hosting; no second hosting runtime required | NOT falsified |

## Historical APP-HOST delivery vs Protocol-v2 residual defects

Historical **Done** plan rows (W1 public foundation, W2 engine, W3 process control/supervision, APP-HOST-9A author facade, LKW 8A–8E live proof) remain valid delivery facts — real engine, supervisor, file-lock guard, graceful shutdown, restart, and LKW ProofReceipt path were delivered as claimed. The six accepted Protocol-v2 findings document **residual observability export, runtime health policy, shutdown truthfulness, durability flush classification, instance recovery, and plan-state gaps** at `audited_sha`. Remediation hardens the existing hosting stack; it does **not** reopen closed historical Done rows, claim universal production qualification, or require a second hosting runtime.

## Root-cause remediation grouping

### HOSTING-RUNTIME-HEALTH-INTEGRITY — effective component failure policy after READY

**Findings:** `AUDIT-20260818-APPLICATION_HOSTING-02`

Effective `ComponentFailureAction` remains enforceable throughout READY/runtime health evolution. Do not build a second health subsystem.

### HOSTING-SHUTDOWN-EVIDENCE-INTEGRITY — lifecycle evidence and truthful shutdown classification

**Findings:** `AUDIT-20260818-APPLICATION_HOSTING-01`, `AUDIT-20260818-APPLICATION_HOSTING-03`, `AUDIT-20260818-APPLICATION_HOSTING-04`

Public hosted lifecycle evidence reaches canonical Observability; terminal shutdown classifications truthfully reflect component-stop and required-durability outcomes. Cross-link [`OBSERVABILITY_EVIDENCE`](../../project/architecture/OBSERVABILITY_EVIDENCE.md) rather than duplicate evidence infrastructure. Do not create a hosting-specific event bus/store.

### HOSTING-INSTANCE-RECOVERY-INTEGRITY — PID reuse resistant stale recovery

**Findings:** `AUDIT-20260818-APPLICATION_HOSTING-05`

Local stale ownership recovery is robust against PID reuse while preserving native file-lock authority.

### HOSTING-PLAN-STATE-INTEGRITY — plan header vs row register

**Findings:** `AUDIT-20260818-APPLICATION_HOSTING-06`

Current-state summary is consistent with APP-HOST row register and architecture shipped W1/W2/W3 vs remaining backlog (3D, 5D/5E, 6, 7, 9B–9F).

## Cross-links to existing remediation

| Existing block | Relationship |
|----------------|--------------|
| **OBS-JOURNAL-IDENTITY-INTEGRITY** / **OBSERVABILITY_EVIDENCE** | Cross-link for canonical export authority and journal durability — HOSTING-SHUTDOWN-EVIDENCE-INTEGRITY coordinates rather than duplicates |
| **T3-RUNTIME-SCOPE-INTEGRITY** | Orthogonal Tier-3 composition bus scope — hosting export disconnect is separate finding HOST-01 |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `a323dfa7a95292725a925a8b4c4370adc947adf7`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical APP-HOST **Done** plan rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- Finding 04: required vs best-effort flush policy surface — deferred to remediation design.
- Finding 05: OS process incarnation identifier choice — deferred to remediation; native lock remains authoritative.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-APPLICATION_HOSTING-01` … `AUDIT-20260818-APPLICATION_HOSTING-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
