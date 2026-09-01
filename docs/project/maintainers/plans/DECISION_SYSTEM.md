# Decision System — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC-CLEAN (2026-08-30):** Canonical target architecture **FROZEN**. Implementation **NOT STARTED**. Production remains CVL / Critic until clean-cut migration.

> When implementing this layer, read **only** the architecture doc and **this plan hub**.

**Last updated:** 2026-08-30 — DS-DOC-CLEAN plan consolidation.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Implement / audit default:** architecture frozen banner · Critic disposition · Phase DS-E2E blocking gate summary.
- **Use** `Read` with offset/limit — open **P0/P1** rows with Status ≠ Done in **one** phase section only.
- **Skip** **Done** / closed unless re-validating a cited gap.
- **Architecture hub:** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) read-scope block only.
- **Paired architecture:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) — one per session max.
- **CURRENT implementation:** [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) — migration audit only.
- **Extended depth:** [`architecture/satellites/DECISION_SYSTEM_extended_depth.md`](../../architecture/satellites/DECISION_SYSTEM_extended_depth.md) on demand.

---

## Architecture frozen vs implementation planned

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** — [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) canon |
| **Verification architecture** | **FROZEN** — [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) |
| **Deliberation architecture** | **FROZEN** — [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) |
| **Runtime implementation** | **PLANNED** — no Decision System classes shipped |
| **Production path** | **CURRENT** — `intergrax/runtime/critic/**` until clean cut |
| **Evidence** | **PLANNED** — DS-E2E Docker qualification phase after migration |

---

## Phase index

| Phase | Status | Detail section |
| ----- | ------ | -------------- |
| **DS-CORE** | PLANNED | [below](#phase-ds-core--decision-lifecycle-foundation) |
| **DS-VER-PIPE / DS-VER-STAGES** | PLANNED | [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) |
| **DS-DELIB / DS-COUNCIL** | PLANNED | [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md) |
| **DS-MIG** (Critic clean cut) | PLANNED | [below](#phase-ds-mig--critic-clean-cut-migration) |
| **DS-E2E** (Docker qualification) | PLANNED | [below](#phase-ds-e2e--docker-production-qualification) |

---

## Phase DS-CORE — Decision Lifecycle foundation (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-CORE-01 | P0 | Decision ID / Version / scope typed contracts | **Done** — `intergrax/contracts/decision_identity.py`; `tests/unit/contracts/test_decision_identity.py` |
| DS-CORE-02 | P0 | Candidate vs Authoritative Decision records + immutable lineage | **Done** — `intergrax/contracts/decision_record.py`; `tests/unit/contracts/test_decision_record.py` |
| DS-CORE-03 | P0 | Lifecycle state machine hosted by canonical Execution (no second runtime) | **Done** — `intergrax/contracts/decision_lifecycle.py`; `tests/unit/contracts/test_decision_lifecycle.py` |
| DS-CORE-04 | P0 | Resolution semantics (`ACCEPTED` / `REJECTED` / `UNRESOLVED`) | **Done** — `intergrax/contracts/decision_resolution.py`; `tests/unit/contracts/test_decision_resolution.py` |
| DS-CORE-05 | P1 | Finalize guard — one authoritative per decision scope | **Done** — `intergrax/contracts/decision_finalization.py`; `tests/unit/contracts/test_decision_finalization.py` |
| DS-CORE-06 | P1 | Execution-hosted checkpoint persistence for Decision lifecycle state | **Done** — `intergrax/contracts/decision_checkpoint.py`; `intergrax/runtime/execution/decision_checkpoint_persistence.py`; `tests/unit/contracts/test_decision_checkpoint.py` |
| DS-CORE-07 | P1 | Parallel proposal branch lineage | **Done** |
| DS-CORE-08 | P2 | Core typed Decision Artifact kind registration contracts | **Done** — `intergrax/contracts/decision_artifact_registry.py`; `tests/unit/contracts/test_decision_artifact_registry.py` |

### Plugin architecture (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-PLUGIN-01 | P1 | Platform Plugins discovery/config integration for `DecisionStrategy` (same canonical registry as DS-DELIB-01) | **Planned** |
| DS-PLUGIN-02 | P1 | Verification stage registration surface | **Planned** |
| DS-PLUGIN-03 | P2 | Plugin/config integration for Decision Artifact kind registration | **Planned** |

---

## Phase DS-INTEGRATION — Execution host · orchestration · governance · observability · recovery (PLANNED)

### Execution-host integration

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-EXEC-00 | P0 | Prove Decision capability is optional: ordinary Execution flows bypass Decision Lifecycle entirely when no authoritative decision is required | **Done** |
| DS-EXEC-01 | P0 | Execution host / strategy-routing hooks → Decision Lifecycle | **Planned** |
| DS-EXEC-02 | P1 | Lifecycle stage persistence via canonical Execution checkpoint ports | **Planned** |

### DS-EXEC-00 — Decision System optionality / bypass contract (DONE)

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

**Future test matrix (runtime — not in DS-DELIB-02 slice):**

| Flow class | Without Decision | With Decision |
| ---------- | ---------------- | ------------- |
| INFERENCE | ordinary inference flow without Decision | Decision-enabled inference flow |
| AGENTIC | ordinary agentic flow without Decision | Decision-enabled agentic flow |
| ORCHESTRATION | ordinary orchestration flow without Decision | Decision-enabled orchestration flow |

Goal: **Decision capability orthogonal to ExecutionStrategy** — none of INFERENCE, AGENTIC, or ORCHESTRATION require Decision System.

**Non-goals for DS-EXEC-00 scoping:** no premature global `DECISION_SYSTEM_ENABLED` flag; no `NoDecisionStrategy` / `NullDecisionStrategy` workaround — absence means Lifecycle is not entered. DS-EXEC-00 does **not** forbid optional Decision host seams in Execution (DS-EXEC-01); it proves ordinary flows do not **require** Decision configuration or lifecycle entry.

### Orchestration-specific integration (Nexus)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-NEXUS-01 | P0 | Graph / UAEP hooks for ORCHESTRATION-backed Decision Strategy work | **Planned** |
| DS-NEXUS-02 | P1 | Orchestration checkpoint participation when ORCHESTRATION is selected | **Planned** |

### Governance / HITL / Execution Authority

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-GOV-01 | P1 | Version-bound authorization handoff to Governed Execution | **Planned** |
| DS-GOV-02 | P1 | HITL invocation for approver / adjudicator (remove L2 Critic) | **Planned** |

### Observability / Diagnostics

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-OBS-01 | P1 | Decision lifecycle audit events on observability spine | **Planned** |
| DS-OBS-02 | P2 | Diagnostics feed boundaries (no lifecycle ownership) | **Planned** |

### Persistence / recovery / concurrency

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-REC-01 | P0 | Finalize idempotency + conflict detection | **Planned** |
| DS-REC-02 | P1 | Crash resume without duplicate authoritative outcome | **Planned** |
| DS-REC-03 | P1 | Budget ceiling preserved on resume | **Planned** |

### Failure / security hardening

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-SEC-01 | P0 | Execution identity binding on all decision records | **Planned** |
| DS-SEC-02 | P1 | Stale approval protection across revisions | **Planned** |

---

## Phase DS-MIG — Critic clean-cut migration (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-MIG-01 | P0 | Wire graph/UAEP paths to Decision Lifecycle | **Planned** |
| DS-MIG-02 | P0 | Retire `CriticOrchestrator` after pipeline parity | **Planned** |
| DS-MIG-03 | P1 | Remove L2 from verification model; route HITL via Lifecycle | **Planned** |
| DS-MIG-04 | P1 | DELETE CRITIC_VERIFICATION docs + retire `intergrax/runtime/critic/**` | **Planned** |
| DS-MIG-05 | P2 | Update application CriticProfile/CVL references | **Planned** |

---

## Critic → Decision disposition matrix

Audited against `intergrax/runtime/critic/**` and [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md). **Target owner** is post-migration; **Disposition** guides clean-cut slice.

| Current Critic capability | Target owner | Disposition |
| ------------------------- | ------------ | ----------- |
| L0 deterministic (`L0Gateway`, `NexusValidationEngine`) | Decision Verification — structural/deterministic stages | **MOVE/REUSE** |
| L1 semantic (`L1Gateway`, `eval.judge`) | Decision Verification — semantic stage | **MOVE/REUSE** |
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
| `guardrail_l0` / `merge_guardrail_l0` | Decision Verification — deterministic stage | **MOVE/REUSE** |
| `CriticGraphHooks` / `critic_wiring` | Nexus graph → Decision Lifecycle hooks | **REPLACE** |
| `CriticTraceEmitter` / `CriticVerdictDiagV1` | Observability decision/verification events | **MOVE** |
| `CriticProfile` / `CriticScope` / `CriticVerdict` contracts | Decision + Verification typed contracts | **REPLACE** |
| `RubricSpec` | Decision Verification semantic stage | **REUSE** |
| Evidence claims integration | Shared evidence contracts | **KEEP** |
| `NexusEvalRunner` / shadow / offline eval | Evaluation / OECP | **KEEP OUTSIDE** |
| `OnlineEvaluationRegistry` | Evaluation / OECP | **KEEP OUTSIDE** |
| `borderline_l1_score` L2 escalation heuristic | HITL policy trigger via Lifecycle | **MOVE** |
| `ToolRegistryCriticEvalClient` | Verification stage tool client | **REUSE** |

**CRITIC_VERIFICATION docs:** **CURRENT IMPLEMENTATION SNAPSHOT** — physical **DELETE** planned in clean-cut slice ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) banner).

---

## Migrated open requirements (from Critic audit)

Re-owned from [`CRITIC_VERIFICATION` plan](CRITIC_VERIFICATION.md) Protocol v2 findings.

| ID | Owner | Status |
| -- | ----- | ------ |
| DS-VER-RUBRIC-PROVENANCE-INTEGRITY | Verification | ACCEPTED / PLANNED |
| DS-VER-PRODUCER-INDEPENDENCE | Verification · Deliberation | ACCEPTED / PLANNED |
| DS-VER-ADVERSARIAL-SEMANTIC | Verification | ACCEPTED / PLANNED |
| DS-DEC-EXECUTION-IDENTITY-BINDING | Decision System | ACCEPTED / PLANNED |
| DS-VER-RESULT-COHERENCE | Verification | ACCEPTED / PLANNED |
| DS-DEC-REVISION-LOOP-BOUNDEDNESS | Decision System | ACCEPTED / PLANNED |

---

## Phase DS-E2E — Docker production qualification (PLANNED) — **blocking gate**

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

## Definition of done — production qualification

The Decision System is **not** production-qualified after unit tests, integration tests, or mocked E2E alone.

**Production qualification** requires:

1. Runtime migration slices through DS-CORE / DS-MIG complete for in-scope capabilities.
2. **Phase DS-E2E** rows executed as **real Docker E2E** — not mocks.
3. **DS-FINAL-AUDIT** passed at an exact commit pin.

Until then, production correctness remains **CVL / Critic** ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md)).

---

## Cross-domain references

| Need | Canonical source |
|------|------------------|
| Verification stages | [`DECISION_VERIFICATION.md` plan](DECISION_VERIFICATION.md) |
| Council / strategies | [`DECISION_DELIBERATION.md` plan](DECISION_DELIBERATION.md) |
| HITL | [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) |
| Policy | [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) |
| CURRENT Critic | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |

---

## Delivery rule

One **DS-\*** ID per PR → update the owning phase row in this hub → documentation gates green → no `shipped` claim until runtime slice lands.
