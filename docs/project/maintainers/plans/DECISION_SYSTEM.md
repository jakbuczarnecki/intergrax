# Decision System - Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-ROADMAP-REALITY-SYNC (2026-09-04):** Canonical target architecture **FROZEN**. Canonical Decision System runtime is **implemented** and is the **production decision authority**. Legacy Critic production authority has been **fully retired**. Remaining work: trust hardening · durable authority/recovery · CouncilStrategy · Platform Plugins integration · lifecycle observability/diagnostics · real Docker E2E production qualification · final exact-commit audit. **Not** whole-system production-qualified until DS-E2E + DS-FINAL-AUDIT.

> When implementing this layer, read **only** the architecture doc and **this plan hub**.

**Last updated:** 2026-09-04 - DS-ROADMAP-REALITY-SYNC.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Implement / audit default:** architecture frozen banner · Critic disposition · Phase DS-E2E blocking gate summary.
- **Use** `Read` with offset/limit - open **P0/P1** rows with Status ≠ Done in **one** phase section only.
- **Skip** **Done** / closed unless re-validating a cited gap.
- **Architecture hub:** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) read-scope block only.
- **Paired architecture:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) - one per session max.
- **CURRENT implementation:** Decision System runtime — see [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md).
- **Extended depth:** [`architecture/satellites/DECISION_SYSTEM_extended_depth.md`](../../architecture/satellites/DECISION_SYSTEM_extended_depth.md) on demand.

---

## Architecture frozen vs implementation reality

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** - [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) canon |
| **Verification architecture** | **FROZEN** - [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) |
| **Deliberation architecture** | **FROZEN** - [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) |
| **Core Decision runtime** | **DONE** |
| **Decision Revision** | **ENTERPRISE CLOSED** |
| **Decision Verification Pipeline** | **DONE** |
| **Verification production composition** | **ENTERPRISE CLOSED** |
| **Decision Strategy foundation** | **DONE** |
| **Decision Governance** | **DONE** |
| **Decision Human Review** | **DONE** |
| **Execution/Nexus integration** | **DONE** |
| **Critic migration** | **COMPLETE** |
| **Council Strategy** | **PLANNED** |
| **Platform plugin integration** | **PARTIAL** |
| **Durable authority/recovery** | **PARTIAL** |
| **Lifecycle observability** | **PARTIAL** |
| **Production qualification** | **PLANNED** - DS-E2E Docker evidence |
| **Final Decision System audit** | **PLANNED** |

---

## Path to complete

Ordered sequencing labels (existing **DS-\*** IDs remain authoritative):

| # | Group | Outcome |
| - | ----- | ------- |
| 0 | **DS-ROADMAP-REALITY-SYNC** | Roadmap reflects implemented system; no stale Critic/current-runtime claims. |
| 1 | **DS-VER-TRUST-HARDENING** | The verifier cannot be manipulated by the answer it is evaluating. |
| 2 | **DS-DURABLE-AUTHORITY** | Crash or concurrent workers cannot create two final authoritative decisions or reset revision limits. |
| 3 | **DS-COUNCIL** | Multiple models can independently propose, disagree, and synthesize one candidate. |
| 4 | **DS-PLUGIN** | New strategies/verifiers/artifact kinds can be supplied through Platform Plugins without modifying Decision core. |
| 5 | **DS-OBS-DIAG** | Operators can reconstruct why a decision became authoritative. |
| 6 | **DS-E2E** | The architecture is proven with real providers/processes/containers. |
| 7 | **DS-FINAL-AUDIT** | One exact commit is independently certified against architecture and evidence. |

---

## Phase index

| Phase | Status | Detail section |
| ----- | ------ | -------------- |
| **DS-CORE** | **DONE** | [below](#phase-ds-core--decision-lifecycle-foundation) |
| **DS-REV** | **DONE** | [below](#phase-ds-rev--revision-policy-foundation) |
| **DS-VER-PIPE / DS-VER-STAGES** | **DONE** | [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) |
| **DS-DELIB / DS-COUNCIL** | **DONE / PLANNED** | [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md) - DS-DELIB **DONE**; DS-COUNCIL **PLANNED** |
| **DS-MIG** (Critic clean cut) | **COMPLETE** | [below](#phase-ds-mig--critic-clean-cut-migration) |
| **DS-E2E** (Docker qualification) | **PLANNED** | [below](#phase-ds-e2e--docker-production-qualification) |

---

## Phase DS-CORE - Decision Lifecycle foundation (DONE)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-CORE-01 | P0 | Decision ID / Version / scope typed contracts | **Done** - `intergrax/contracts/decision_identity.py`; `tests/unit/contracts/test_decision_identity.py` |
| DS-CORE-02 | P0 | Candidate vs Authoritative Decision records + immutable lineage | **Done** - `intergrax/contracts/decision_record.py`; `tests/unit/contracts/test_decision_record.py` |
| DS-CORE-03 | P0 | Lifecycle state machine hosted by canonical Execution (no second runtime) | **Done** - `intergrax/contracts/decision_lifecycle.py`; `tests/unit/contracts/test_decision_lifecycle.py` |
| DS-CORE-04 | P0 | Resolution semantics (`ACCEPTED` / `REJECTED` / `UNRESOLVED`) | **Done** - `intergrax/contracts/decision_resolution.py`; `tests/unit/contracts/test_decision_resolution.py` |
| DS-CORE-05 | P1 | Finalize guard - one authoritative per decision scope | **Done** - `intergrax/contracts/decision_finalization.py`; `tests/unit/contracts/test_decision_finalization.py` |
| DS-CORE-06 | P1 | Execution-hosted checkpoint persistence for Decision lifecycle state | **Done** - `intergrax/contracts/decision_checkpoint.py`; `intergrax/runtime/execution/decision_checkpoint_persistence.py`; `tests/unit/contracts/test_decision_checkpoint.py` |
| DS-CORE-07 | P1 | Parallel proposal branch lineage | **Done** |
| DS-CORE-08 | P2 | Core typed Decision Artifact kind registration contracts | **Done** - `intergrax/contracts/decision_artifact_registry.py`; `tests/unit/contracts/test_decision_artifact_registry.py` |

### Phase DS-REV - Revision policy foundation (DONE)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-REV-01 | P0 | Decision revision policy foundation (challenge → bounded authorization → revised candidate minting) | **Done / ENTERPRISE CLOSED** - identity-bound `DecisionRevisionState`, policy provenance in `DecisionRevisionDecision`, custom evaluator semantic outputs validated against canonical policy semantics; `intergrax/contracts/decision_revision.py`; `intergrax/runtime/decision_revision.py`; `tests/unit/runtime/test_decision_revision.py` |

### Plugin architecture (PARTIAL)

Domain-owned immutable registries exist. Shared Platform Plugins discovery/admission/config/trust integration remains open.

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-PLUGIN-01 | P1 | Platform Plugins discovery/config integration for `DecisionStrategy` (`DecisionStrategyRegistry` **DONE**; Platform Plugins integration **OPEN**) | **PARTIAL** |
| DS-PLUGIN-02 | P1 | Verification stage registration surface (`VerificationStageRegistry` **DONE**; Platform Plugins integration **OPEN**) | **PARTIAL** |
| DS-PLUGIN-03 | P2 | Plugin/config integration for Decision Artifact kind registration (`DecisionArtifactKindRegistry` **DONE**; plugin/config integration **OPEN**) | **PARTIAL** |

**Architecture:** Platform Plugin System → discovery/admission/config/trust → Decision domain composition adapter → domain-owned immutable registry. Decision contracts do **not** import discovery (source gate on `decision_strategy.py`).

---

## Phase DS-INTEGRATION - Execution host · orchestration · governance · observability · recovery (IMPLEMENTED / PARTIAL)

### Execution-host integration

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-EXEC-00 | P0 | Prove Decision capability is optional: ordinary Execution flows bypass Decision Lifecycle entirely when no authoritative decision is required | **Done** |
| DS-EXEC-01 | P0 | Execution host optional scoped Decision Lifecycle capability (`DecisionLifecycleHost` + active binding in `ExecutionRuntime`) | **Done** |
| DS-EXEC-02 | P1 | Lifecycle stage persistence via canonical Execution checkpoint ports | **Done** |

### DS-EXEC-00 - Decision System optionality / bypass contract (DONE)

Decision System is **optional per flow**. Ordinary Execution work must complete without entering Decision Lifecycle when no authoritative decision is required.

**Acceptance contract (future proof):**

```text
A. Execution without Decision:
Application → Execution → normal execution work → completion

B. Execution with Decision:
Application → Execution → Decision Lifecycle → strategy / verification / resolution → continue execution as required
```

**Required future proofs:**

| Proof | Expectation |
| ----- | ----------- |
| Decision System disabled / absent | Ordinary Execution still works |
| Decision System not selected for a flow | No Decision identity · no Decision lifecycle · no Decision checkpoint · no Decision finalization · no Decision verification |
| Decision System selected | Canonical Decision Lifecycle applies |

**Future invariant:** No Decision artifacts or lifecycle state are created for a flow that does not request Decision capability.

**Future test matrix (runtime - not in DS-DELIB-02 slice):**

| Flow class | Without Decision | With Decision |
| ---------- | ---------------- | ------------- |
| INFERENCE | ordinary inference flow without Decision | Decision-enabled inference flow |
| AGENTIC | ordinary agentic flow without Decision | Decision-enabled agentic flow |
| ORCHESTRATION | ordinary orchestration flow without Decision | Decision-enabled orchestration flow |

Goal: **Decision capability orthogonal to ExecutionStrategy** - none of INFERENCE, AGENTIC, or ORCHESTRATION require Decision System.

**Non-goals for DS-EXEC-00 scoping:** no premature global `DECISION_SYSTEM_ENABLED` flag; no `NoDecisionStrategy` / `NullDecisionStrategy` workaround - absence means Lifecycle is not entered. DS-EXEC-00 does **not** forbid optional Decision host seams in Execution (DS-EXEC-01); it proves ordinary flows do not **require** Decision configuration or lifecycle entry.

### DS-EXEC-01 - Execution-hosted Decision Lifecycle capability (DONE)

`ExecutionRuntime` may accept an optional `decision_lifecycle_host`. When configured, the host is bound for the execution scope around canonical `ExecutionBoundary` work and reset in `finally` - success or failure. Decision-aware delegate code obtains the host via `require_active_decision_lifecycle_host()` and explicitly calls `start(identity)` / `transition(...)`.

**Invariants:** host presence does not create `DecisionIdentity` or lifecycle state; `ExecutionBoundary` and `StrategyExecutionRouter` remain Decision-neutral; canonical lifecycle semantics stay in `intergrax/contracts/decision_lifecycle.py`.

Proof gate: `tests/unit/runtime/execution/test_decision_lifecycle_host.py`.

### DS-EXEC-02 - Execution-hosted Decision checkpoint persistence (DONE)

`ExecutionRuntime` may accept an optional `decision_checkpoint_persistence`. When configured, the port is bound for the execution scope around canonical `ExecutionBoundary` work and reset in `finally` - success or failure. Decision-aware delegate code obtains persistence via `require_active_decision_checkpoint_persistence()` and explicitly calls `save_decision_checkpoint(...)` / `load_decision_checkpoint(...)`.

**Invariants:** persistence presence does not auto-save or auto-load checkpoints; `DecisionLifecycleHost` does not own persistence; `ExecutionBoundary` and `StrategyExecutionRouter` remain Decision-neutral; canonical checkpoint semantics stay in `intergrax/contracts/decision_checkpoint.py`.

Proof gate: `tests/unit/runtime/execution/test_decision_checkpoint_runtime_integration.py`.

### Orchestration-specific integration (Nexus)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-NEXUS-01 | P0 | Canonical Execution work submission for Decision Strategy work (ORCHESTRATION capability via Execution-owned child seam; Nexus remains private) | **Done** |
| DS-NEXUS-02 | P1 | Orchestration checkpoint participation when ORCHESTRATION is selected | **Done** |

#### DS-NEXUS-02 - Orchestration checkpoint/recovery participation (DONE)

Decision semantic checkpoint and physical orchestration recovery remain independently owned. Decision-triggered `ORCHESTRATION` child work participates in canonical Execution checkpoint/resume without Decision importing Nexus or invoking recovery helpers.

Proof gate: `tests/unit/runtime/execution/test_decision_orchestration_recovery.py`.

#### DS-NEXUS-01 - Decision → Execution work seam (DONE)

Decision-aware code submits canonical `ExecutionRequest` work through an optional execution-scoped `ExecutionWorkPort` hosted by `ExecutionRuntime`. Child work is minted via `ChildExecutionRunner` and routed by the wired `StrategyExecutionRouter` - Decision does **not** import Nexus, construct orchestration backends, or select `ExecutionStrategy`.

```text
Decision Strategy (decision-aware delegate)
      ↓ require_active_execution_work_port()
ExecutionWorkPort
      ↓ ChildExecutionRunner (child ExecutionId + parent lineage)
StrategyExecutionRouter
      ├── INFERENCE
      ├── AGENTIC
      └── ORCHESTRATION → private Nexus implementation
```

**Invariants:** no Nexus field on Decision contracts; no global DecisionStrategy → ExecutionStrategy mapping; ordinary flows without work port do not require orchestration backend; missing orchestration backend fails closed via canonical Execution error.

Proof gate: `tests/unit/runtime/execution/test_decision_execution_work.py`.

### Governance / HITL / Execution Authority

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-GOV-01 | P1 | Version-bound authorization handoff to Governed Execution; execution-time current-policy-context validation | **Done** |
| DS-GOV-02 | P1 | HITL invocation for approver / adjudicator (remove L2 Critic) | **Done** |

### Observability / Diagnostics

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-OBS-01 | P1 | Decision lifecycle audit events on observability spine | **PARTIAL** - canonical Verification RuntimeEvent integration exists (`decision.verification.started` · `stage_completed` · `stage_unavailable` · `probabilistic_skipped` · `completed`); full Decision Lifecycle transition coverage not yet implemented/proven |
| DS-OBS-02 | P2 | Diagnostics feed boundaries (no lifecycle ownership) | **PLANNED** - diagnostics must consume/project Decision evidence; must not own Decision Lifecycle |

### Persistence / recovery / concurrency

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-REC-01 | P0 | Finalize idempotency + conflict detection | **PARTIAL** - semantic idempotency/conflict guard **DONE** (`intergrax/contracts/decision_finalization.py`); durable atomic CAS at persistence boundary and cross-process race proof **OPEN** |
| DS-REC-02 | P1 | Crash resume without duplicate authoritative outcome | **PARTIAL** - `DecisionCheckpointState` · `DecisionCheckpointPersistence` · execution-scoped binding · orchestration recovery participation **DONE**; real process/container crash/restart proof and durable duplicate-authority prevention proof **OPEN** |
| DS-REC-03 | P1 | Budget ceiling preserved on resume | **PLANNED** - checkpoint currently carries lifecycle/finalization only; revision/deliberation budget state not durably restored. **Invariant:** resume must never reset semantic revision/deliberation budget ceilings. |

### Failure / security hardening

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-SEC-01 | P0 | Execution identity binding on all decision records | **IMPLEMENTED / CLOSURE GATE OPEN** - `DecisionIdentity` → `DecisionExecutionLineage` → `TaskId` / `RunId` / `AttemptId` / `ExecutionId` propagated through candidate, proposal refs, verification, revision, human review, authorization; dedicated architecture closure gate not yet recorded |
| DS-SEC-02 | P1 | Stale approval protection across revisions | **DONE** - `validate_human_review_decision_for_proposal()` · `proposal_refs_match()` · `validate_execution_authorization_for_decision()` fail closed on exact-version mismatch; v1 approval cannot authorize v2 |

---

## Phase DS-MIG - Critic clean-cut migration (COMPLETE)

Legacy Critic exists only in historical migration evidence / legacy input normalization where explicitly retained (`intergrax/runtime/migration/**`).

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-MIG-01 | P0 | Wire graph/UAEP paths to Decision Lifecycle; Graph/UAEP cutover with governance/lifecycle semantic hardening complete | **Done** |
| DS-MIG-PARITY | P0 | Dual-run Decision/Critic observational parity qualification (`intergrax/runtime/migration/**`) | **Done / READY** |
| DS-MIG-02 | P0 | Retire `CriticOrchestrator` from production authority; remove `CriticProfile` Decision config authority (`ApplicationDecisionWiringSpec`) | **Done** |
| DS-MIG-03 | P1 | Remove L2 from verification model; route HITL via Lifecycle | **Done** |
| DS-MIG-04 | P1 | DELETE CRITIC_VERIFICATION docs + retire `intergrax/runtime/critic/**` | **ENTERPRISE CLOSED** |
| DS-MIG-05 | P2 | Remove legacy `CriticProfile` application configuration; canonical `DecisionProfile` host wire | **ENTERPRISE CLOSED** |

---

## Critic → Decision disposition matrix (historical migration reference)

Audited against legacy Critic and [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md). **Target owner** is post-migration; **Disposition** guided clean-cut slice. **Migration complete** — matrix retained for provenance only.

| Legacy Critic capability | Target owner | Disposition |
| ------------------------- | ------------ | ----------- |
| L0 deterministic (`L0Gateway`, `NexusValidationEngine`) | Decision Verification - structural/deterministic stages | **MOVE/REUSE** |
| L1 semantic (`L1Gateway`, `eval.judge`) | Decision Verification - semantic stage | **MOVE/REUSE** |
| L1 trajectory (`eval.trajectory`, `trajectory_judge_path`) | Decision Verification / evaluation boundary | **MOVE/REUSE** |
| L2 Human (`L2Gateway`, `ESCALATE_HITL`) | Platform HITL | **DELETE** from Critic model |
| `CriticOrchestrator` | Verification Pipeline + Decision Lifecycle orchestration | **REPLACE** |
| `CriticAction.REVISE` | Decision Lifecycle revision | **MOVE** |
| `CriticAction.RETRY` technical semantics | Nexus Reliability | **MOVE** |
| `CriticAction.ESCALATE_HITL` | Decision Lifecycle → HITL invocation | **MOVE** |
| `CriticAction.FAIL` / `CONTINUE` | Decision Lifecycle resolution semantics | **MERGE** |
| `EvaluatorLoopExecutor` / `EvaluatorLoopSpec` | Decision Lifecycle revision policy | **REPLACE/MOVE** |
| `evaluator_loop_metadata` | Decision Lifecycle revision state | **MOVE** |
| `policy_bridge` / `resolve_critic_action` | Policy boundary + Lifecycle routing | **SPLIT/DELETE** |
| `critic_governance_from_fragment` | Policy profile ingestion only | **SPLIT** |
| `guardrail_l0` / `merge_guardrail_l0` | Decision Verification - deterministic stage | **MOVE/REUSE** |
| `CriticGraphHooks` / `critic_wiring` | Nexus graph → Decision Lifecycle hooks | **REPLACE** |
| `CriticTraceEmitter` / `CriticVerdictDiagV1` | Observability decision/verification events | **MOVE** |
| `CriticProfile` / `CriticScope` / `CriticVerdict` contracts | Decision + Verification typed contracts | **REPLACE** |
| `RubricSpec` | Decision Verification semantic stage | **REUSE** |
| Evidence claims integration | Shared evidence contracts | **KEEP** |
| `NexusEvalRunner` / shadow / offline eval | Evaluation / OECP | **KEEP OUTSIDE** |
| `OnlineEvaluationRegistry` | Evaluation / OECP | **KEEP OUTSIDE** |
| `borderline_l1_score` L2 escalation heuristic | HITL policy trigger via Lifecycle | **MOVE** |
| `ToolRegistryCriticEvalClient` | Verification stage tool client | **REUSE** |

**CRITIC_VERIFICATION docs:** **HISTORICAL IMPLEMENTATION SNAPSHOT** — physical DELETE completed in DS-MIG-04 ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) banner).

---

## Decision/Critic parity table (DS-MIG-PARITY)

Migration-only observational comparison in `intergrax/runtime/migration/decision_critic_parity.py` and `critic_shadow_adapter.py`. Scheduled for deletion with legacy Critic retirement.

| Legacy Critic concept | Target Decision owner | Raw parity expected? | Retirement interpretation |
| --------------------- | --------------------- | -------------------- | ------------------------- |
| L0 deterministic | Decision Verification — structural stage | Yes (normalized outcome) | Missing Decision capability blocks retirement |
| L1 semantic | Decision Verification — semantic stage | Yes (normalized outcome) | Provider-unavailable classified explicitly |
| L1 trajectory | Decision Verification — trajectory stage | Yes where configured | Architectural layer mapping only |
| L2 human / `ESCALATE_HITL` | Decision HITL outside Verification | No (expected difference) | Does not block retirement alone |
| `RETRY` | Execution reliability / RetryEngine | No (expected difference) | Does not block retirement |
| `REVISE` | Decision Revision lifecycle | No when revision policy maps | Does not block retirement alone |
| `FAIL` / `CONTINUE` | Decision resolution / host action | Yes (acceptable vs challenged) | Blocking only on capability gap |
| `policy_bridge` | Not invoked in shadow | N/A | Observational projection only |
| `EvaluatorLoopExecutor` | Forbidden in shadow | N/A | Revision assessed separately |

---

## Migrated open requirements (from Critic audit)

Re-owned from [`CRITIC_VERIFICATION` plan](CRITIC_VERIFICATION.md) Protocol v2 findings.

| ID | Owner | Status |
| -- | ----- | ------ |
| DS-VER-RUBRIC-PROVENANCE-INTEGRITY | Verification | **DONE** - `SemanticRubricRef` · `ResolvedSemanticRubric` · `criteria` · `min_score` · `provenance_ref` · resolver exact-ref check · fail-closed unresolved rubric (`intergrax/contracts/semantic_verification.py`) |
| DS-VER-PRODUCER-INDEPENDENCE | Verification · Deliberation | **IMPLEMENTED / QUALIFICATION OPEN** - `SemanticVerificationIndependenceConfig` · `VerifierIndependenceMode` · `producer_profile_id` · `verifier_profile_id`; contract and runtime enforcement exist; real independent-provider/model production E2E remains **DS-E2E-03** |
| DS-VER-ADVERSARIAL-SEMANTIC | Verification | **DONE / ENTERPRISE CLOSED** - `EvalTrustedRubricContext` · `EvalUntrustedCandidateContent` · `build_eval_judge_messages()` · canonical `intergrax.eval.candidate.v1` serialization · adversarial unit tests (`tests/unit/tools/providers/eval/test_judge_trust_boundary.py`) |
| DS-DEC-EXECUTION-IDENTITY-BINDING | Decision System | **IMPLEMENTED / CLOSURE GATE OPEN** - identity chain through authoritative records; dedicated architecture closure gate not yet recorded |
| DS-VER-RESULT-COHERENCE | Verification | **DONE** - `VerificationResult` · `_validate_result_coherence()` · `VerificationStageRecord` coherence · exact `DecisionProposalRef` binding (`intergrax/contracts/decision_verification.py`) |
| DS-DEC-REVISION-LOOP-BOUNDEDNESS | Decision System | **Done** - `intergrax/contracts/decision_revision.py`; `intergrax/runtime/decision_revision.py`; `tests/unit/runtime/test_decision_revision.py` |

---

## Phase DS-E2E - Docker production qualification (PLANNED) - **blocking gate**

Real Docker E2E qualification is the **final gate** before any Decision System production-qualified claim. Unit, integration, and mocked E2E alone are **insufficient**.

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-E2E-01 | P0 | Real single-model Decision System path | **Planned** |
| DS-E2E-02 | P0 | Real multi-model Council | **Planned** |
| DS-E2E-03 | P0 | Real independent semantic verifier | **Planned** |
| DS-E2E-04 | P0 | Real HITL pause/resume | **Planned** |
| DS-E2E-05 | P0 | Governed real side effect: ALLOW and DENY | **Planned** |
| DS-E2E-06 | P1 | Docker process/container crash + resume without duplicate decision | **Planned** |
| DS-E2E-07 | P1 | Concurrent proposal/finalization race test | **Planned** |
| DS-E2E-08 | P1 | Real budget exhaustion / bounded stop | **Planned** |
| DS-E2E-09 | P1 | Real provider outage / fail-closed behavior | **Planned** |
| DS-E2E-10 | P1 | Two-tenant isolation | **Planned** |
| DS-E2E-11 | P1 | Real observability / OTLP evidence reconstruction | **Planned** |
| DS-E2E-12 | P1 | `ai_incident_investigation` full real integration proof | **Planned** |
| DS-E2E-13 | P1 | Cross-scenario qualification proving no scenario-specific Decision runtime branching | **Planned** |
| DS-FINAL-AUDIT | P0 | Independent exact-commit architecture/runtime/docs/E2E audit | **Planned** |

---

## Definition of done - production qualification

The Decision System is **not** production-qualified after unit tests, integration tests, or mocked E2E alone.

**Production qualification** requires:

1. Runtime migration slices through DS-CORE / DS-MIG complete for in-scope capabilities.
2. **Phase DS-E2E** rows executed as **real Docker E2E** - not mocks.
3. **DS-FINAL-AUDIT** passed at an exact commit pin.

Until then, canonical Decision System runtime is **implemented and active**; **production qualification** of the full system remains **PLANNED** (Phase DS-E2E + DS-FINAL-AUDIT). Do **not** claim **DECISION SYSTEM COMPLETE** or whole-system **PRODUCTION QUALIFIED** yet.

---

## Cross-domain references

| Need | Canonical source |
|------|------------------|
| Verification stages | [`DECISION_VERIFICATION.md` plan](DECISION_VERIFICATION.md) |
| Council / strategies | [`DECISION_DELIBERATION.md` plan](DECISION_DELIBERATION.md) |
| HITL | [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) |
| Policy | [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) |
| Historical Critic snapshot | [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) |

---

## Delivery rule

One **DS-\*** ID per PR → update the owning phase row in this hub → documentation gates green → no `shipped` claim until runtime slice lands.
