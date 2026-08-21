# EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
- **Tier(s):** Tier-0 `intergrax/experiments/` · Tier-0 `intergrax/cli/mvp_evolution.py` · Tier-1 `intergrax/runtime/architecture/` (KPI, satisfaction, online eval) · Tier-3 lab host exposure
- **audited_sha:** `84b2477571650ade894f2d52a6b5398aa86922cc`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 7 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`
- **Scope in:**
  - `ExperimentSession`, `RegisterExperimentRequest`, `ExperimentRecord`, SQLite experiment store
  - `evaluate_against_criteria`, experiment↔run linkage, `summarize_trace`
  - `ProductKpiDefinition` / `FileProductKpiRegistry`
  - `UserSatisfactionEvent`, `satisfaction_to_online_observation`, `OnlineEvaluationRegistry`
  - `intergrax mvp simulate` / `replay` CLI and lab `/v1/mvp/*` HTTP routes
  - G0–G2 promotion gate script (infrastructure-readiness scope)
  - Historical MVP-EVOL **Done** delivery facts (positive control)
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - production deployment or hosting activation
  - second experimentation subsystem invention
  - visual trace UI or central golden-scenario registry claims
- **Prior audit reference(s):** [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md) PBA-FIX-D — experiment persistence port; historical MVP-EVOL plan rows — delivery facts only
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Executive summary

**Verdict: FAIL.** Five accepted HIGH and two accepted MEDIUM findings show that experiment registry operations are not tenant-scoped; stored `validation_criteria` are not evaluated; satisfaction→online-eval bridge drops tenant/task identity; KPI registry collides on `kpi_id` across tenants; `ExperimentSession.run()` falls back from missing `RunId` to `TaskId`; lab HTTP routes invoke argparse-bound CLI functions without arguments; and file-backed KPI/online-eval registries use whole-file read-modify-write without concurrency semantics. Positive controls: experimentation remains a lab workflow, not production deployment authority; Observability owns execution evidence; replay reconstructs persisted events only; CLI remains canonical DX surface; lab routes mount only under harness auth; G0–G2 honestly qualifies infrastructure readiness; maturity remains A4/I3/P2/E3; MVP-EVOL historical **Done** rows remain valid delivery facts — including route **exposure** for MVP-EVOL.7, distinct from residual functional defect DX-06. Remediation is **PLANNED**, not implemented. Cross-link existing **PBA-FIX-D** for persistence-provider work — do not duplicate.

## Verdict

**FAIL** — 0 CRITICAL / 5 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-01

- **Severity:** HIGH
- **Category:** TENANT ISOLATION / EXPERIMENT OWNERSHIP
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-EXPERIMENT-IDENTITY-INTEGRITY
- **Claim falsified:** Canonical experiment identity and every registry operation are tenant-scoped; experiment↔run linkage proves matching canonical tenant/run identity.
- **Observation:** `ExperimentSession` carries `tenant_id`/`user_id` for runtime Task creation. But `RegisterExperimentRequest` has no tenant identity; `ExperimentRecord` has no tenant identity; SQLite `experiments` table has no `tenant_id`; `experiment_runs` has no `tenant_id`; `get`/`set_decision`/`link_run`/`list_experiments` operate globally by `experiment_id` or decision. Experiment ownership is therefore not tenant-bound and a run can be linked to an experiment without proving common tenant scope.
- **Location:**
  - `intergrax/experiments/models.py` — `RegisterExperimentRequest`, `ExperimentRecord`
  - `intergrax/experiments/store.py` — SQLite schema and registry operations
  - `intergrax/experiments/workflow.py` — `ExperimentSession` tenant on Task only
- **Reproduction:** Register experiments under two tenants; link a run from tenant B to tenant A's experiment via shared global store — observe no tenant validation on linkage.
- **Impact:** Cross-tenant experiment evidence corruption; laboratory comparisons can attach runs to the wrong tenant scope.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-02

- **Severity:** HIGH
- **Category:** EXPERIMENT SEMANTICS / PAPER CRITERIA
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-EVALUATION-EVIDENCE-INTEGRITY
- **Claim falsified:** Every active experiment criterion is executable and versioned/typed; stored criteria cannot silently be ignored.
- **Observation:** `RegisterExperimentRequest` and `ExperimentRecord` expose `validation_criteria`. `ExperimentSession.evaluate_against_criteria()` never reads `record.validation_criteria`. It only checks: `completed`, `result.metadata["validation_valid"]` is True, non-empty answer, optional `expected_output` substring. A named/free-text validation criterion can therefore be recorded while `ExperimentRunOutcome.passed` is computed without evaluating it.
- **Location:**
  - `intergrax/experiments/models.py` — `validation_criteria` field
  - `intergrax/experiments/workflow.py` — `evaluate_against_criteria()`
- **Reproduction:** Register experiment with non-empty `validation_criteria`; run with failing semantic criterion but passing lightweight checks — observe `passed=True` while criteria were never evaluated.
- **Impact:** Experiment decisions can qualify candidates on paper criteria that were never executed.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-03

- **Severity:** HIGH
- **Category:** EVIDENCE IDENTITY / TENANT ATTRIBUTION
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-EVALUATION-EVIDENCE-INTEGRITY
- **Claim falsified:** Evaluation evidence preserves trusted canonical execution identity across domain bridges; reuse tenant + TaskId + RunId (+ AttemptId where semantically required).
- **Observation:** `UserSatisfactionEvent` contains `tenant_id`, `task_id`, `run_id`, `agent_id`. `satisfaction_to_online_observation()` drops `tenant_id` and `task_id`. `OnlineEvaluationObservation` itself has no tenant/task identity fields. `OnlineEvaluationRegistry` stores/lists observations globally. Tenant-scoped satisfaction evidence loses canonical tenant/task provenance when bridged into online evaluation.
- **Location:**
  - `intergrax/runtime/architecture/user_satisfaction.py` — `UserSatisfactionEvent`, `satisfaction_to_online_observation()`
  - `intergrax/runtime/architecture/online_evaluation_models.py` — `OnlineEvaluationObservation`
  - `intergrax/runtime/architecture/online_evaluation_registry.py` — global store/list
- **Reproduction:** Record satisfaction with tenant A; inspect stored online observation — observe missing tenant/task fields and global registry listing.
- **Impact:** Adaptive/evaluation aggregation can consume observations whose tenant ownership was discarded.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-04

- **Severity:** HIGH
- **Category:** TENANT ISOLATION / BUSINESS EVIDENCE CORRUPTION
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-EVALUATION-EVIDENCE-INTEGRITY
- **Claim falsified:** KPI identity is at least `tenant_id + kpi_id`; observation→definition linkage is same-scope and validated; cross-tenant definition collision is impossible.
- **Observation:** `ProductKpiDefinition` carries `tenant_id` + `kpi_id`. `FileProductKpiRegistry.register_definition()` removes previous definitions by `kpi_id` only. A definition from tenant B using the same `kpi_id` therefore removes tenant A's definition. `record_observation()` also does not prove that the referenced KPI definition belongs to `observation.tenant_id`.
- **Location:**
  - `intergrax/runtime/architecture/product_kpi_registry.py` — `ProductKpiDefinition`, `FileProductKpiRegistry`
- **Reproduction:** Register KPI `kpi_id=latency` for tenant A; register same `kpi_id` for tenant B — observe tenant A definition removed; record observation for tenant A against orphaned/mismatched definition.
- **Impact:** Cross-tenant KPI evidence corruption falsifies experiment and product comparison loops.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-05

- **Severity:** HIGH
- **Category:** EXECUTION IDENTITY / EVIDENCE LINKAGE
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-EXPERIMENT-IDENTITY-INTEGRITY
- **Claim falsified:** Missing canonical RunId is an evidence-linkage failure; never synthesize/fallback RunId from TaskId.
- **Observation:** `ExperimentSession.run()` does `run_id = result.run_id or result.task_id` and persists the value as experiment `run_id`. TaskId and RunId are separate canonical execution identities. A missing RunId therefore causes TaskId to masquerade as RunId. `summarize_trace()` later interprets the linked value as a real run ID.
- **Location:**
  - `intergrax/experiments/workflow.py` — `ExperimentSession.run()`, `summarize_trace()`
- **Reproduction:** Execute experiment run where `TaskResult.run_id` is absent; inspect linked experiment run_id and call `summarize_trace()` — observe TaskId used as RunId for trace lookup.
- **Impact:** Trace summaries and experiment evidence can reference the wrong execution identity.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-06

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION DEFECT / DX SURFACE
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-SURFACE-PERSISTENCE-INTEGRITY
- **Claim falsified:** HTTP surface uses typed request DTOs and a shared service layer; CLI and HTTP are adapters around the same service; HTTP route must be executable with typed parameters.
- **Observation:** CLI functions require positional args: `run_mvp_simulate(args: argparse.Namespace)`, `run_mvp_replay(args: argparse.Namespace)`. But `create_mvp_evolution_router` invokes `run_mvp_simulate()` and `run_mvp_replay()` with no arguments. Routes are mounted on the lab host when `settings.harness=True` and protected by `require_harness_auth`. Both wrappers therefore fail before executing the CLI action.
- **Location:**
  - `intergrax/cli/mvp_evolution.py` — `run_mvp_simulate`, `run_mvp_replay`
  - `intergrax/applications/_shared/mvp_evolution_routes.py` — `create_mvp_evolution_router`
  - `applications/lab_application/host/factory.py` — harness route mount
- **Reproduction:** POST to `/v1/mvp/simulate` or `/v1/mvp/replay` on lab host with harness auth — observe TypeError before CLI logic runs.
- **Impact:** Lab HTTP surface is non-functional despite route exposure delivery; developers cannot use HTTP as an adapter to the same workflow.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-07

- **Severity:** MEDIUM
- **Category:** CONCURRENCY / EVIDENCE DURABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** DX-SURFACE-PERSISTENCE-INTEGRITY
- **Claim falsified:** Canonical lab evidence persistence has explicit concurrency semantics; use provider-neutral persistence ports or explicitly constrain provider to single-process use.
- **Observation:** `FileProductKpiRegistry` and `FileOnlineEvaluationRegistry` use whole-file read-modify-write persistence. No lock/CAS/transaction/version exists. Concurrent writers can read the same state and overwrite each other's updates. Lab/developer stores — not production distributed-store failure — but lost evidence can still falsify experiment results.
- **Location:**
  - `intergrax/runtime/architecture/product_kpi_registry.py` — `FileProductKpiRegistry`
  - `intergrax/runtime/architecture/online_evaluation_registry.py` — `FileOnlineEvaluationRegistry`
- **Reproduction:** Run two concurrent writers updating the same registry file — observe last writer wins with lost updates.
- **Impact:** Concurrent lab workflows can lose KPI or evaluation observations, corrupting comparison evidence.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Experimentation remains lab workflow, not production deployment authority | NOT falsified |
| Experiment evidence ≠ deployment permission | NOT falsified |
| Application Hosting owns production activation | NOT falsified |
| Observability remains canonical execution-evidence owner | NOT falsified |
| Replay CLI reconstructs persisted events; does not re-execute agents/tools | NOT falsified |
| Replay does not silently replay external side effects | NOT falsified |
| CLI remains canonical DX surface | NOT falsified |
| Lab MVP routes mounted only for harness mode and protected by harness auth | NOT falsified |
| G0–G2 script is infrastructure-readiness/file-presence qualification, not semantic product-quality proof | NOT falsified |
| Docs state promotion script is not proven workflow-wired | NOT falsified |
| No central golden-scenario registry falsely claimed | NOT falsified |
| Visual trace UI not falsely claimed shipped | NOT falsified |
| Maturity remains A4/I3/P2/E3 | NOT falsified |
| No E4 public full-harness experiment bundle claimed | NOT falsified |
| Existing PBA-FIX-D remains PLANNED; cross-link, do not duplicate | NOT falsified |
| Findings require hardening existing DX/evaluation stores, not a second experimentation subsystem | NOT falsified |

## Historical MVP-EVOL delivery vs Protocol-v2 residual defects

Historical **MVP-EVOL.1–7 Done** delivery facts remain valid — promotion gate script, simulate/replay CLI, KPI registry, satisfaction bridge, author guide appendix, and lab HTTP route **exposure** were delivered as claimed. The seven accepted Protocol-v2 findings document **residual tenant identity, criteria semantics, evaluation identity, KPI scope, run identity, HTTP/CLI service boundary, and lab persistence concurrency gaps** at `audited_sha`. **MVP-EVOL.7** delivered route mounting and harness auth guard (**route exposure delivered**); **DX-06** records the residual **functional defect** that HTTP wrappers invoke argparse-bound CLI functions without arguments. Remediation hardens the existing DX path; it does **not** reopen closed MVP-EVOL rows or require a second experimentation subsystem.

## Root-cause remediation grouping

### DX-EXPERIMENT-IDENTITY-INTEGRITY — experiment ownership and run linkage

**Findings:** `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-01`, `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-05`

Experiment ownership and run linkage use canonical tenant and execution identity. Cross-link [`IDENTITY_TRUST`](../../project/architecture/IDENTITY_TRUST.md) rather than create DX-specific identity authority.

### DX-EVALUATION-EVIDENCE-INTEGRITY — criteria and product evidence identity

**Findings:** `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-02`, `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-03`, `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-04`

Experiment criteria are real executable semantics; product/satisfaction evidence preserves tenant-scoped identity across bridges.

### DX-SURFACE-PERSISTENCE-INTEGRITY — CLI/HTTP service boundary and lab persistence

**Findings:** `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-06`, `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-07`

CLI and HTTP share a real service boundary; lab evidence stores have explicit safe persistence/concurrency semantics. Cross-link existing **PBA-FIX-D** ([`plan`](../project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md#protocol-v22-pba-fix-d--experiment-persistence-port-2026-08-18)) — do not duplicate persistence architecture.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `84b2477571650ade894f2d52a6b5398aa86922cc`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical MVP-EVOL **Done** plan rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- Finding 02: resolve criteria into canonical evaluation assets vs typed check specification vs clean-cut unsupported field — deferred to remediation design.
- Finding 07: shared concurrency semantics via PBA-FIX-D port vs explicit single-process constraint — deferred to remediation; cross-link PBA-FIX-D.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 7 (`AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-01` … `AUDIT-20260818-EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE-07`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED. **PBA-FIX-D** remains **ACCEPTED / PLANNED**.
