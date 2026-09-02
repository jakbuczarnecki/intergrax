# ADAPTIVE_HARNESS_INTELLIGENCE - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** ADAPTIVE_HARNESS_INTELLIGENCE
- **Tier(s):** Tier-1 `intergrax/runtime/adaptive/` · Tier-1 `intergrax/runtime/architecture/adaptive_governance.py`
- **audited_sha:** `173ec35c50679a352213b9412da46cdf5784f7df`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 2 CRITICAL / 4 HIGH / 0 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- **Scope in:**
  - `AdaptationExecutor` - shadow / canary / apply / rollback promotion boundary
  - `AdaptationGovernancePipeline` - gate evaluation and `passed_all_gates` semantics
  - `AdaptationProposalPackage` / `AdaptationCandidate` promotion lineage
  - `ProfileVersionStore`, `ProfileActivePointerStore`, `ProfileVersionLifecycleManager`
  - `ProfileVersionRecord` scope fields (`tenant_id`, `task_class`, `artifact_type`)
  - `PolicyLearningApprovalStore` and `require_policy_learning_approval()`
  - `evaluate_adaptive_governance` / `AdaptiveLoopEnvelope` authority contracts
  - Historical W-ADAPT **70/70 Done** delivery facts (positive control)
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - TOKEN-AHI full-loop re-qualification
  - ADAS runtime delivery re-audit
  - second adaptive governance pipeline invention
  - production autonomous adaptation claim
- **Prior audit reference(s):** legacy adaptive audits under `docs/audit_results/legacy/` - historical only; Protocol v2 snapshot at pinned SHA supersedes for campaign register
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `c6125b6d09700e948fe7442021756787e4c356bd`

## Executive summary

**Verdict: FAIL.** Two CRITICAL and four HIGH accepted findings show that `AdaptationExecutor.apply()` does not enforce `passed_all_gates` or bind governance results to the exact profile version activated; version resolution is global by `version_id` without tenant/task-class scope verification, enabling cross-tenant active-pointer mutation; policy-learning apply can proceed without authoritative approval when `approval_store=None`; `AdaptationGovernancePipeline` can report `passed_all_gates=True` when mandatory capability-compatibility and golden-scenario evidence are absent; `apply()` performs multi-step lifecycle and pointer mutations without recoverable transactional semantics; and `ProfileActivePointerStore.swap_active()` lacks expected-version CAS fencing under concurrency. Positive controls: AHI remains versioned promotion, not in-place live mutation; proposal engine and mutation executor stay separate; `ProfileVersionRecord` preserves lineage/scope/artifact payload; `ProfileVersionLifecycleManager` enforces explicit state transitions; `AdaptiveLoopEnvelope` has typed authority/max-iteration/max-delta contracts; policy-learning envelope requires approval metadata and bounded delta; observe/recommend/auto-with-human-gate distinction remains architecturally sound; AHI does not install Skills directly; TOKEN-AHI remains partial/recommendation-only; ADAS remains target/planned; production autonomous adaptation is not falsely claimed shipped; current maturity remains A4/I3/P2/E2. Residual defects require hardening the existing promotion boundary - remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** - 2 CRITICAL / 4 HIGH / 0 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-01

- **Severity:** CRITICAL
- **Category:** GOVERNANCE / CONFIGURATION MUTATION BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** AHI-PROMOTION-AUTHORITY-INTEGRITY
- **Claim falsified:** Production profile activation consumes one authoritative promotion artifact binding proposal, exact profile version, tenant, task class, artifact type, governance result, authority level, required approvals, and promotion/verification evidence; failed gates or lineage mismatch fail closed.
- **Observation:** `AdaptationExecutor.shadow()` correctly rejects packages where `passed_all_gates` is false. `AdaptationExecutor.apply()` does not check `package.passed_all_gates`. It also accepts `package` and `version_id` independently and does not prove: `version_id == package.candidate.profile_draft.version_id`; version was materialized from `package.proposal_id`; package and version belong to the same promotion lineage. The executor can therefore perform SHADOW/CANARY → ACTIVE and active-pointer mutation using a failed or unrelated proposal package.
- **Location:**
  - `intergrax/runtime/adaptive/adaptation_executor.py` - `shadow()`, `apply()`, `_require_version()`
  - `intergrax/runtime/adaptive/adaptation_models.py` - `AdaptationProposalPackage`, `passed_all_gates`
- **Reproduction:** Call `apply()` with a package where `passed_all_gates=False` or with a `version_id` unrelated to the package candidate lineage - observe promotion proceeds without gate or lineage binding.
- **Impact:** Failed or unrelated proposals can authorize production configuration activation - undermines governance boundary and audit trust for adaptive promotion.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-02

- **Severity:** CRITICAL
- **Category:** TENANT ISOLATION / CROSS-TENANT CONFIGURATION MUTATION
- **Status at publication:** ACCEPTED
- **Remediation block:** AHI-PROMOTION-AUTHORITY-INTEGRITY
- **Claim falsified:** Every adaptive lifecycle mutation is scope-bound; resolved record must match exact `tenant_id` + `task_class` + `artifact_type` + `version_id`; cross-scope mismatch fails closed.
- **Observation:** `ProfileVersionRecord` stores `tenant_id` and `task_class`. `ProfileVersionStore.get(version_id)` is global by `version_id`. `AdaptationExecutor._require_version()` only resolves by `version_id`. `canary()` and `apply()` accept `tenant_id`/`task_class` separately but never verify that the resolved record belongs to that tenant/task class. `apply()` can transition a version owned by tenant B to ACTIVE and then write an active pointer under tenant A. `ProfileVersionLifecycleManager` also transitions by global `version_id` only.
- **Location:**
  - `intergrax/runtime/adaptive/profile_version_store.py` - `get(version_id)`
  - `intergrax/runtime/adaptive/adaptation_executor.py` - `canary()`, `apply()`, `_require_version()`
  - `intergrax/runtime/adaptive/profile_lifecycle.py` - `ProfileVersionLifecycleManager`
  - `intergrax/runtime/adaptive/adaptation_models.py` - `ProfileVersionRecord`
- **Reproduction:** Materialize a version for tenant B; call `apply(tenant_id="A", task_class=..., version_id=...)` - observe cross-tenant activation and pointer write without scope rejection.
- **Impact:** Cross-tenant profile activation is possible - critical isolation failure on adaptive configuration mutation.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-03

- **Severity:** HIGH
- **Category:** HUMAN APPROVAL / FAIL-OPEN
- **Status at publication:** ACCEPTED
- **Remediation block:** AHI-PROMOTION-AUTHORITY-INTEGRITY
- **Claim falsified:** If effective policy requires human approval, authoritative approval evidence is mandatory; missing approval authority/store is a fail-closed configuration error; evidence is scoped to exact proposal/version/tenant/change.
- **Observation:** Policy-learning governance requires human approval metadata. Actual approval is checked through `PolicyLearningApprovalStore`. `AdaptationExecutor` accepts `approval_store=None`. `apply()` invokes `require_policy_learning_approval()` only when an approval store was injected. Therefore a policy-learning proposal requiring human approval can reach apply without authoritative approval verification when the store is absent. A `proposal.human_approver_id` is not proof that the person approved the action.
- **Location:**
  - `intergrax/runtime/adaptive/adaptation_executor.py` - `apply()`, optional `approval_store`
  - `intergrax/runtime/adaptive/policy_learning_approval.py` - `require_policy_learning_approval()`
  - `intergrax/runtime/architecture/adaptive_governance.py` - policy-learning approval requirements
- **Reproduction:** Configure policy-learning proposal requiring human approval; invoke `apply()` with `approval_store=None` - observe apply proceeds without authoritative approval verification.
- **Impact:** Human-gated policy-learning promotions can bypass approval when store is not injected - fail-open governance on high-risk adaptive changes.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-04

- **Severity:** HIGH
- **Category:** EVIDENCE QUALIFICATION / FAIL-OPEN GOVERNANCE
- **Status at publication:** ACCEPTED
- **Remediation block:** AHI-EVIDENCE-QUALIFICATION-INTEGRITY
- **Claim falsified:** Required evidence depends on authority level, artifact/change class, environment, and promotion stage; missing required production evidence means NOT QUALIFIED, not PASS; `passed_all_gates` means all gates required for that intended action were actually evaluated and passed.
- **Observation:** `AdaptationGovernancePipeline` initializes `capability_gate_passed=True` and runs compatibility only when both capability graphs are supplied. Golden gate returns True when `golden_scenario_pass_rate` is None. Thus `passed_all_gates` can become true when capability-compatibility evidence and golden-scenario evidence are completely absent. This may be acceptable for observe/recommend modes but is not sufficient as a universal production promotion qualification.
- **Location:**
  - `intergrax/runtime/adaptive/governance_pipeline.py` - `AdaptationGovernancePipeline`, gate initialization and golden gate
  - `intergrax/runtime/adaptive/adaptation_models.py` - `passed_all_gates`
- **Reproduction:** Run governance pipeline without capability graphs and without `golden_scenario_pass_rate` - observe `passed_all_gates=True` despite absent mandatory production evidence.
- **Impact:** Production promotion can be qualified without evidence that gates were actually evaluated - undermines promotion integrity for apply-stage actions.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-05

- **Severity:** HIGH
- **Category:** CONSISTENCY / PARTIAL COMMIT DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** AHI-ACTIVATION-CONSISTENCY-INTEGRITY
- **Claim falsified:** Activation is one recoverable logical transaction with `operation_id`, expected active version, new version, explicit state-transition outcome, idempotency, and reconciliation after partial failure.
- **Observation:** `AdaptationExecutor.apply()` sequentially: transitions candidate to CANARY; transitions candidate to ACTIVE; reads current pointer; retires previous ACTIVE version; swaps active pointer. `ProfileVersionStore` and `ProfileActivePointerStore` are separate persistence authorities/stores. No transaction or logical operation protocol spans candidate status, previous status, and active pointer. Failure between those steps can leave contradictory state, e.g.: candidate ACTIVE; previous RETIRED; pointer still referencing previous version. Rollback has the same multi-step consistency risk.
- **Location:**
  - `intergrax/runtime/adaptive/adaptation_executor.py` - `apply()`, `rollback()`
  - `intergrax/runtime/adaptive/profile_version_store.py` - version status persistence
  - `intergrax/runtime/adaptive/profile_pointer_store.py` - active pointer persistence
  - `intergrax/runtime/adaptive/profile_lifecycle.py` - lifecycle transitions
- **Reproduction:** Inject failure between lifecycle transition and pointer swap steps - observe contradictory candidate status, retired predecessor, and stale active pointer.
- **Impact:** Partial activation can leave production configuration in inconsistent state without recoverable reconciliation semantics.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-06

- **Severity:** HIGH
- **Category:** CONCURRENCY / ACTIVE CONFIGURATION RACE
- **Status at publication:** ACCEPTED
- **Remediation block:** AHI-ACTIVATION-CONSISTENCY-INTEGRITY
- **Claim falsified:** Active profile swap is version-fenced: `expected_active_version_id` + `new_active_version_id`; concurrent stale promotion fails with explicit conflict, not silent last-write-wins.
- **Observation:** `ProfileActivePointerStore.swap_active()` has no `expected_active_version`/CAS argument. SQLite implementation: calls `get_pointer()`; derives `previous_version_id` from that stale read; later performs `INSERT ... ON CONFLICT DO UPDATE`. Two concurrent promotions can both read V1, then write V2/V3. Last writer wins and may retain `previous_version_id=V1`, losing V2 from the promotion/rollback chain. Lifecycle status transitions can also diverge from the final pointer.
- **Location:**
  - `intergrax/runtime/adaptive/profile_pointer_store.py` - `swap_active()`, SQLite implementation
  - `intergrax/runtime/adaptive/adaptation_executor.py` - `apply()` pointer swap invocation
- **Reproduction:** Run two concurrent `apply()` promotions against the same tenant/task scope - observe last-write-wins pointer mutation and lost intermediate version from rollback chain.
- **Impact:** Concurrent promotions can silently corrupt active configuration lineage and rollback semantics.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| AHI is versioned promotion, not in-place live mutation | NOT falsified |
| Proposal engine and mutation executor remain separate responsibilities | NOT falsified |
| `ProfileVersionRecord` preserves lineage/scope/artifact payload | NOT falsified |
| `ProfileVersionLifecycleManager` enforces explicit state transitions | NOT falsified |
| `AdaptiveLoopEnvelope` has typed authority/max-iteration/max-delta contracts | NOT falsified |
| Policy-learning envelope requires approval metadata and bounded delta | NOT falsified |
| Observe/recommend/auto-with-human-gate distinction remains architecturally sound | NOT falsified |
| AHI does not install Skills directly | NOT falsified |
| TOKEN-AHI remains partial/recommendation-only | NOT falsified |
| ADAS remains target/planned | NOT falsified |
| Production autonomous adaptation is not falsely claimed shipped | NOT falsified |
| Current maturity remains A4/I3/P2/E2 | NOT falsified |
| Findings require hardening existing promotion boundary, not creating a second adaptive subsystem | NOT falsified - remediation targets existing executor/governance/stores path |

## Historical W-ADAPT vs Protocol-v2 residual defects

Historical **W-ADAPT 70/70 Done** delivery facts remain valid - `SignalCollector`, `AdaptationEngine`, `AdaptationGovernancePipeline`, `AdaptationExecutor`, `VerificationLoop`, profile lifecycle stores, sub-engines, and governance contracts were delivered as claimed. The six accepted Protocol-v2 findings document **residual promotion authority, tenant scope, human approval, evidence qualification, activation consistency, and concurrency gaps** at `audited_sha` - they harden the existing promotion boundary; they do **not** reopen W-ADAPT closeout rows or require a second adaptive subsystem.

## Root-cause remediation grouping

### AHI-PROMOTION-AUTHORITY-INTEGRITY - scope-bound authoritative promotion decision

**Findings:** `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-01`, `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-02`, `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-03`

One scope-bound authoritative promotion decision binds proposal, gates, approval, tenant, and exact profile version. Failed gates, lineage mismatch, or cross-scope resolution fail closed. Human-required promotion fails closed when approval evidence authority is unavailable. Cross-link [`GOVERNED_EXECUTION`](../../project/architecture/GOVERNED_EXECUTION.md) / [`IDENTITY_TRUST`](../../project/architecture/IDENTITY_TRUST.md) rather than duplicate approval/identity infrastructure.

### AHI-EVIDENCE-QUALIFICATION-INTEGRITY - action/stage-aware gate completeness

**Findings:** `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-04`

Gate completeness requirements are action/stage aware; missing mandatory production evidence never silently passes. Distinguish optional recommendation evidence from mandatory production promotion evidence.

### AHI-ACTIVATION-CONSISTENCY-INTEGRITY - recoverable transactional activation and CAS fencing

**Findings:** `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-05`, `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-06`

Active configuration promotion/rollback has recoverable transactional semantics and CAS fencing under concurrency. Reuse existing platform CAS/revision mechanisms where appropriate.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `173ec35c50679a352213b9412da46cdf5784f7df`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical W-ADAPT **Done** plan rows remain valid delivery facts - not rewritten.

## Open questions / blocked items

- Finding 01: exact promotion artifact type (`ProfileVersionRef` vs composite binding) - deferred to remediation design without second governance pipeline.
- Finding 05: provider-neutral atomic/CAS semantic contract surface - deferred to remediation reusing existing platform CAS patterns.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-01` … `AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
