# UNIFIED_CONTEXT_LIFECYCLE — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** UNIFIED_CONTEXT_LIFECYCLE
- **Constituent domains:** UNIFIED_CONTEXT_LIFECYCLE (UCL contracts · Nexus orchestration · Memory/Session revision · Token Optimization executors · CE ContextPlan integration)
- **Tier(s):** Tier-0 `intergrax/context/` · Tier-1 `intergrax/runtime/context_lifecycle/` · Tier-1 `intergrax/runtime/nexus/context/` · Tier-1 `intergrax/runtime/token_optimization/` · Tier-1 `intergrax/runtime/nexus/session/`
- **audited_sha:** `0f6e2d7fe96498346d8ddcc05fe08caa68c00523`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 0 MEDIUM / 0 LOW
- **Operator decision:** all 5 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/UNIFIED_CONTEXT_LIFECYCLE.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md`
- **Scope in:**
  - `ContextOptimizationPolicy` human-review and persistence policy pairing
  - canonical Nexus UCL orchestration ephemeral persistence branches
  - `DurableCompactionPolicy.activation_mode` and `minimum_validation_requirement`
  - `SessionContextRevisionActivationService` CAS activation path
  - durable candidate builder vs durable validation compiler ordering
  - `SQLiteSessionContextRevisionStore` revision-zero bootstrap vs durable source identity contracts
  - `ArtifactLookupKey` / repository lifecycle and reuse-before-create (positive control)
  - CE / Memory / Token Optimization / Nexus ownership boundaries (positive control)
  - historical CTX-UCL-1…6D / CTX-UCL-CLOSEOUT-1 and TOKEN-10E-1…4 / TOKEN-10E **ACCEPTED / CLOSED** delivery states (positive control — not re-audited as failures)
- **Scope out:**
  - remediation implementation
  - second UCL subsystem
  - universal re-qualification of every execution surface beyond documented hot paths
  - silent runtime fixes in production source
  - human-review UX implementation (Application Hosting)
  - rollback execution (Memory/Session deferred scope)
- **Prior audit reference(s):** Protocol v2 [`CONTEXT_ENGINEERING`](CONTEXT_ENGINEERING.md) (CE/UCL boundary — positive control); historical CTX-UCL / TOKEN-10E **ACCEPTED / CLOSED** rows remain valid delivery facts
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `a69129927c80a8ec61d6eec894c63fbb93c6e67b`

## Executive summary

**Verdict: FAIL.** Five accepted HIGH findings show governance bypass on ephemeral human-review policy (transformed context used without review evidence), durable activation ignoring `MANUAL_REVIEW_THEN_COMPARE_AND_SWAP`, paper `minimum_validation_requirement` levels, revision-zero bootstrap contradiction across durable contracts and CAS persistence, and premature repository publication of durable candidates before durable validation completes. Positive controls: UCL cross-domain ownership model is sound; CE remains sole global model-input budget authority; Memory/Session owns durable revision activation; Token Optimization remains transformation executor not repository owner; `ArtifactLookupKey` identity, canonical serialization, payload integrity verification, session-history snapshot binding, `OptimizationExecutionGuard`, reuse-before-create, SQLite transaction-backed reservation, and CAS activation with operation-id idempotency are genuinely implemented; documentation honestly states A4/I3/P2/E3 with no customer production qualification; rollback execution and human-review UX remain not claimed shipped; findings do not require a second UCL subsystem. Residual defects are Protocol-v2 governance and durable-lifecycle integrity gaps distinct from historical CTX-UCL / TOKEN-10E delivery completion — remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** — 0 CRITICAL / 5 HIGH / 0 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-01

- **Severity:** HIGH
- **Category:** GOVERNANCE / HUMAN-REVIEW BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** UCL-GOVERNED-REVIEW-INTEGRITY
- **Claim falsified:** When `require_human_review=True`, unreviewed transformed context cannot become approved model-facing context or reusable durable state; `PERSIST_AFTER_HUMAN_REVIEW` is paired with authoritative review evidence.
- **Substance:** `ContextOptimizationPolicy` exposes `require_human_review` and `EphemeralArtifactPersistencePolicy.PERSIST_AFTER_HUMAN_REVIEW`. The contract requires `PERSIST_AFTER_HUMAN_REVIEW` to be paired with `require_human_review=True`. Canonical Nexus UCL runtime persists only `PERSIST_REUSABLE` and `PERSIST_AFTER_VALIDATION`; `PERSIST_AFTER_HUMAN_REVIEW` falls into the non-persist branch. However the generated summary is still materialized into the current model-facing messages and consumed without any human approval evidence. Only repository persistence is skipped. Existing test `test_non_persist_flow_has_no_persistent_artifact_identity` explicitly proves this behavior for `PERSIST_AFTER_HUMAN_REVIEW`.
- **Evidence:**
  - `intergrax/runtime/context_lifecycle/contracts.py` — `ContextOptimizationPolicy.require_human_review`; `EphemeralArtifactPersistencePolicy.PERSIST_AFTER_HUMAN_REVIEW`
  - `intergrax/runtime/nexus/context/ucl_orchestration.py` — non-persist branch for `PERSIST_AFTER_HUMAN_REVIEW`; model-facing message materialization without review gate
  - `tests/unit/runtime/nexus/context/test_ucl_orchestration.py` — `test_non_persist_flow_has_no_persistent_artifact_identity`
- **Confidence:** HIGH — explicit persist-branch split and test proving use-without-persist semantics.
- **Target invariant:** Human review policy governs the permitted lifecycle transition, not merely storage. When `require_human_review=True`, unreviewed transformed context must not become approved model-facing context or reusable durable state. If the host human-review bridge is unavailable, fail closed rather than silently degrading to "use now, do not persist". Reuse canonical UER/Governed Execution approval evidence if/when wired. Do not introduce caller-controlled approval booleans.

### AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-02

- **Severity:** HIGH
- **Category:** GOVERNANCE / DURABLE ACTIVATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** UCL-GOVERNED-REVIEW-INTEGRITY
- **Claim falsified:** `MANUAL_REVIEW_THEN_COMPARE_AND_SWAP` is an enforceable durable state-transition policy requiring trusted approval evidence before CAS activation.
- **Substance:** `DurableCompactionPolicy` exposes `COMPARE_AND_SWAP` and `MANUAL_REVIEW_THEN_COMPARE_AND_SWAP`. Repository-wide runtime usage of `activation_mode` is absent outside contracts/serialization/tests. `DurableCompactionActivationRequirements` does not carry the required activation mode or review evidence requirement. `SessionContextRevisionActivationService` validates the candidate/outcome and then performs CAS activation without evaluating `DurableCompactionPolicy.activation_mode` or authoritative human-review evidence.
- **Evidence:**
  - `intergrax/runtime/context_lifecycle/contracts.py` — `DurableCompactionPolicy.activation_mode`; `DurableCompactionActivationRequirements`
  - `intergrax/runtime/nexus/session/context_revision.py` — `SessionContextRevisionActivationService` CAS path without activation-mode or review-evidence evaluation
  - `intergrax/runtime/context_lifecycle/serialization.py` — `activation_mode` serialization only
- **Confidence:** HIGH — no runtime consumer of `activation_mode` beyond contracts/tests.
- **Target invariant:** `MANUAL_REVIEW_THEN_COMPARE_AND_SWAP` requires trusted approval evidence scoped to the exact tenant/context scope/candidate/revision before CAS. Application Hosting may own UX; Memory/Session activation remains responsible for rejecting activation whose required review proof is missing. Do not duplicate HITL infrastructure.

### AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-03

- **Severity:** HIGH
- **Category:** VALIDATION POLICY / PAPER CONTROL
- **Status at publication:** ACCEPTED
- **Remediation block:** UCL-DURABLE-VALIDATION-INTEGRITY
- **Claim falsified:** Every supported durable validation level has explicit, deterministic required validation stages and evidence; stronger configured requirements cannot be satisfied by weaker validators.
- **Substance:** `DurableCompactionPolicy` exposes `STRUCTURAL`, `STRUCTURAL_AND_PROTECTED`, and `FULL` through `minimum_validation_requirement`. The field participates in policy serialization/hash identity but is not consumed by the durable validation runtime. `DurableCompactionValidationCompiler` runs one fixed validation path irrespective of the configured requirement.
- **Evidence:**
  - `intergrax/runtime/context_lifecycle/contracts.py` — `DurableCompactionPolicy.minimum_validation_requirement`
  - `intergrax/runtime/token_optimization/durable_compaction_validation.py` — `DurableCompactionValidationCompiler` fixed path
  - `intergrax/runtime/context_lifecycle/serialization.py` — requirement in policy hash identity
- **Confidence:** HIGH — policy field present; runtime path invariant to configured level.
- **Target invariant:** Every supported durable validation level maps to executable validation semantics. A stronger configured requirement cannot be satisfied by a weaker validator. If the platform currently supports only one durable validation level, clean-cut unsupported values instead of retaining paper controls.

### AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-04

- **Severity:** HIGH
- **Category:** DURABILITY / BOOTSTRAP CONTRACT DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** UCL-REVISION-GENESIS-INTEGRITY
- **Claim falsified:** One genesis revision model is consistent across durable source identity, validation eligibility, CAS activation, and persistence.
- **Substance:** `SQLiteSessionContextRevisionStore` represents a new context scope with `active_revision = 0` and its CAS activation mechanism can logically create revision 1 from expected revision 0. `SessionContextRevisionActivationRequest` also accepts expected revision 0. But `DurableCompactionSourceIdentity.expected_active_revision` and `DurableCompactionActivationRequirements.expected_active_revision` require a strict positive integer. `assess_durable_compaction_eligibility` explicitly rejects `<=0` as `MISSING_EXPECTED_ACTIVE_REVISION`. Therefore the canonical durable candidate/validation contracts cannot represent the first `0 → 1` activation of a new revision scope.
- **Evidence:**
  - `intergrax/runtime/nexus/session/context_revision.py` — `SessionContextRevisionActivationRequest`; CAS from expected revision 0
  - `intergrax/runtime/context_lifecycle/contracts.py` — `DurableCompactionSourceIdentity.expected_active_revision`; `DurableCompactionActivationRequirements.expected_active_revision`
  - `intergrax/runtime/token_optimization/durable_compaction_candidate.py` — `assess_durable_compaction_eligibility` rejects `<=0`
- **Confidence:** HIGH — contradictory revision-zero semantics across adjacent contracts.
- **Target invariant:** Define one genesis revision model. Either: (A) `active_revision=0` means "no active compacted revision" and `0 → 1` is a legal first CAS transition throughout all contracts, or (B) bootstrap an explicit baseline revision before durable compaction and prove that invariant at host/session creation. Contract and persistence semantics must agree.

### AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-05

- **Severity:** HIGH
- **Category:** ARTIFACT LIFECYCLE / VALIDATION ORDERING DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** UCL-DURABLE-VALIDATION-INTEGRITY
- **Claim falsified:** Repository lifecycle state distinguishes executor-valid candidate from durable-policy-valid reusable artifact; rejected durable candidates cannot remain canonical lookup-eligible.
- **Substance:** `DurableCompactionCandidateBuilder` executes `MessageSequenceArtifactExecutor`, performs executor/payload validation, and stores the artifact through `store_validated_artifact()` before the separate `DurableCompactionValidationCompiler` performs durable protected-region validation. The repository artifact is therefore lookup-eligible as `ReusableArtifactStatus.VALIDATED` / `ArtifactValidationStatus.PASSED` before durable validation has passed. If durable validation later returns `REJECTED`, the compiler does not invalidate or retire that repository artifact. A future identical durable candidate operation can lookup/reuse the same rejected artifact and reject repeatedly. SQLite additionally permits only one active validated artifact per tenant + lookup key hash, so the prematurely validated artifact can prevent a replacement candidate under the same identity.
- **Evidence:**
  - `intergrax/runtime/token_optimization/durable_compaction_candidate.py` — `DurableCompactionCandidateBuilder`; `store_validated_artifact()` before durable validation
  - `intergrax/runtime/token_optimization/durable_compaction_validation.py` — `DurableCompactionValidationCompiler`; no invalidation on `REJECTED`
  - `intergrax/runtime/context_lifecycle/sqlite_repository.py` — one active validated artifact per tenant + lookup key hash
  - `intergrax/runtime/context_lifecycle/repository.py` — `store_validated_artifact()` lifecycle semantics
- **Confidence:** HIGH — explicit ordering and repository status promotion before durable validation completes.
- **Target invariant:** Repository lifecycle state must distinguish executor-valid candidate from durable-policy-valid reusable artifact. An artifact that fails required durable validation cannot remain eligible for canonical reuse. Define deterministic invalidation/retry/replacement semantics for rejection. Do not require a distributed transaction; make ordering and reconciliation explicit and testable.

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| UCL cross-domain ownership model | NOT falsified — sound |
| CE remains sole global model-input budget authority | NOT falsified |
| Memory/Session remains durable revision owner | NOT falsified |
| Token Optimization remains transformation executor, not repository owner | NOT falsified |
| `ArtifactLookupKey` provides deterministic reuse identity | NOT falsified |
| Canonical serialization/hash is deterministic | NOT falsified |
| `StoredOptimizationArtifact` verifies payload SHA-256 integrity | NOT falsified |
| Session-history snapshot binding validates tenant/context/revision ownership | NOT falsified |
| `OptimizationExecutionGuard` distinguishes primary/internal model calls and blocks recursive same-target execution | NOT falsified |
| Reuse-before-create is genuinely implemented | NOT falsified |
| SQLite repository implements transaction-backed same-key reservation, durable/shared process visibility and bounded wait | NOT falsified |
| Documentation correctly avoids claiming universal distributed locking | NOT falsified |
| `SessionContextRevisionStore` uses transactional CAS activation and operation-id idempotency | NOT falsified |
| Architecture honestly remains A4/I3/P2/E3 and states no customer production qualification | NOT falsified |
| Rollback execution and human-review UX remain explicitly not claimed shipped | NOT falsified |
| Findings do not require a second UCL subsystem | NOT falsified |

## Historical delivery vs Protocol-v2 residual defects

Historical **CTX-UCL-1…6D**, **CTX-UCL-CLOSEOUT-1**, **TOKEN-10E-1…4**, and **TOKEN-10E** **ACCEPTED / CLOSED** delivery facts remain valid — ephemeral integration, bounded durable compaction runtime, SQLite repository, and CAS activation were delivered as claimed. **TOKEN-10E-CLOSEOUT-1** remains **READY_FOR_REVIEW**. The five accepted Protocol-v2 findings document **residual governance and durable-lifecycle integrity gaps** discovered by adversarial falsification at `audited_sha` — they do not reopen or negate historical closeout rows.

## Root-cause remediation grouping

### UCL-GOVERNED-REVIEW-INTEGRITY — authoritative human review for ephemeral use and durable activation

**Findings:** `AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-01`, `AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-02`

Human review becomes an authoritative gate for ephemeral transformed-context use and durable activation where policy requires it. Cross-link canonical Governance/UER approval authority rather than duplicate it.

### UCL-DURABLE-VALIDATION-INTEGRITY — executable validation levels and correct artifact lifecycle ordering

**Findings:** `AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-03`, `AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-05`

Validation policy levels are executable and repository lifecycle does not publish a durable candidate as fully reusable before all required validation has passed.

### UCL-REVISION-GENESIS-INTEGRITY — consistent revision-zero bootstrap model

**Findings:** `AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-04`

One consistent revision-zero/baseline/bootstrap model across durable source identity, validation, and CAS activation.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `0f6e2d7fe96498346d8ddcc05fe08caa68c00523`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical CTX-UCL / TOKEN-10E plan **ACCEPTED / CLOSED** rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- Finding 04: operator choice between genesis model (A) revision-zero legal first CAS vs (B) explicit baseline bootstrap — deferred to remediation design.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 5 (`AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-01` … `AUDIT-20260818-UNIFIED_CONTEXT_LIFECYCLE-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
