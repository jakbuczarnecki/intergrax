# Decision System — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC (2026-08-30):** Canonical target architecture **FROZEN**. Implementation **NOT STARTED**. Production remains CVL / Critic until clean-cut migration slice.

> When implementing this layer, read **only** the architecture doc and **this plan hub**.

**Last updated:** 2026-08-30 — DS-DOC documentation foundation.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Implement / audit default:** §Critic disposition · open DS-* rows · architecture frozen banner.
- **Use** `Read` with offset/limit — open Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** **Done** / closed unless re-validating a cited gap.
- **Architecture hub:** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) read-scope block only.
- **Paired docs:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) — on demand, one per session max.
- **CURRENT implementation:** [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) — migration audit only.

---

## Architecture frozen vs implementation planned

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** — [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) canon |
| **Verification architecture** | **FROZEN** — [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) |
| **Deliberation architecture** | **FROZEN** — [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) |
| **Runtime implementation** | **PLANNED** — no Decision System classes shipped |
| **Production path** | **CURRENT** — `intergrax/runtime/critic/**` until clean cut |
| **Evidence** | **PLANNED** — Docker E2E after migration |

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

**CRITIC_VERIFICATION docs:** **CURRENT IMPLEMENTATION SNAPSHOT** — physical **DELETE** planned in clean-cut slice after runtime migration ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) banner).

---

## Phase DS-CORE — Decision Lifecycle foundation (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-CORE-01 | P0 | Decision ID / Version / scope typed contracts | **Planned** |
| DS-CORE-02 | P0 | Candidate vs Authoritative Decision records + immutable lineage | **Planned** |
| DS-CORE-03 | P0 | Lifecycle state machine executed by Nexus (no second runtime) | **Planned** |
| DS-CORE-04 | P0 | Resolution semantics incl. UNRESOLVED | **Planned** |
| DS-CORE-05 | P1 | Finalize guard — one authoritative per decision scope | **Planned** |
| DS-CORE-06 | P1 | Nexus checkpoint persistence for lifecycle state | **Planned** |
| DS-CORE-07 | P1 | Parallel proposal branch lineage | **Planned** |
| DS-CORE-08 | P2 | Decision Artifact kind registration | **Planned** |

---

## Phase DS-MIG — Critic clean cut (PLANNED)

| ID | Priority | Item | Status |
|----|----------|------|--------|
| DS-MIG-01 | P0 | Wire graph/UAEP paths to Decision Lifecycle | **Planned** |
| DS-MIG-02 | P0 | Retire `CriticOrchestrator` after pipeline parity | **Planned** |
| DS-MIG-03 | P1 | Remove L2 from verification model; route HITL via Lifecycle | **Planned** |
| DS-MIG-04 | P1 | DELETE [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) docs + retire `intergrax/runtime/critic/**` | **Planned** |
| DS-MIG-05 | P2 | Update application CriticProfile/CVL references | **Planned** |

---

## Migrated open requirements (from Critic audit)

Source: [`CRITIC_VERIFICATION` plan](CRITIC_VERIFICATION.md) Protocol v2 findings — re-owned with Decision-oriented IDs.

### DS-VER-RUBRIC-PROVENANCE-INTEGRITY

**Priority:** P0/P1 · **Status:** `ACCEPTED / PLANNED`
**Owner:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md)
**Successor to:** CRITIC-SEMANTIC-AUTHORITY-INTEGRITY (rubric portion)

Named rubric refs resolve to versioned criteria with provenance before semantic stages; unresolvable rubric fails closed.

### DS-VER-PRODUCER-INDEPENDENCE

**Priority:** P0/P1 · **Status:** `ACCEPTED / PLANNED`
**Owner:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md)
**Successor to:** CRITIC-SEMANTIC-AUTHORITY-INTEGRITY (independence portion)

Independent verification profiles prove producer/verifier separation or explicitly label non-independent modes.

### DS-VER-ADVERSARIAL-SEMANTIC

**Priority:** P1 · **Status:** `ACCEPTED / PLANNED`
**Owner:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md)
**Successor to:** CRITIC-SEMANTIC-AUTHORITY-INTEGRITY (adversarial portion)

Judge construction isolates trusted rubric from untrusted candidate; adversarial tests for high-assurance profiles.

### DS-DEC-EXECUTION-IDENTITY-BINDING

**Priority:** P0/P1 · **Status:** `ACCEPTED / PLANNED`
**Owner:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md)
**Successor to:** CRITIC-EXECUTION-IDENTITY-INTEGRITY

Decision / verification / approval records bind canonical tenant/task/run/attempt/execution identity — no `"default"` fallbacks.

### DS-VER-RESULT-COHERENCE

**Priority:** P1/P2 · **Status:** `ACCEPTED / PLANNED`
**Owner:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md)
**Successor to:** CRITIC-CONTRACT-BOUNDEDNESS-INTEGRITY (verdict portion)

Verification Result enforces constructional consistency across pass/stage/challenge fields.

### DS-DEC-REVISION-LOOP-BOUNDEDNESS

**Priority:** P1/P2 · **Status:** `ACCEPTED / PLANNED`
**Owner:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md)
**Successor to:** CRITIC-CONTRACT-BOUNDEDNESS-INTEGRITY (evaluator-loop portion)

Revision loop validates non-negative iteration, identity consistency, exhausted semantics; resume cannot expand budget.

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

One **DS-\*** ID per PR → update this table → documentation gates green → no `shipped` claim until runtime slice lands.
