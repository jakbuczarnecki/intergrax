# Decision System — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC-HARDEN (2026-08-30):** Canonical target architecture **FROZEN**. Implementation **NOT STARTED**. Production remains CVL / Critic until clean-cut migration slice.

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`satellites/` on demand).

**Last updated:** 2026-08-30 — DS-DOC-HARDEN documentation architecture.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Implement / audit default:** architecture frozen banner · Critic disposition · Phase DS-E2E blocking gate summary.
- **Use** `Read` with offset/limit — open **P0/P1** rows with Status ≠ Done in **one** implementation satellite only.
- **Skip** **Done** / closed unless re-validating a cited gap.
- **Architecture hub:** [`architecture/DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) read-scope block only.
- **Paired architecture:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](../../architecture/DECISION_DELIBERATION.md) — one per session max.
- **CURRENT implementation:** [`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) — migration audit only.
- **Satellites:** at most **one** [`satellites/`](satellites/) file per session unless RESUME cites more.

---

## Implementation satellites (read on demand)

| Satellite | Contents |
| --------- | -------- |
| [`satellites/DECISION_SYSTEM_implementation_core.md`](satellites/DECISION_SYSTEM_implementation_core.md) | DS-CORE lifecycle foundation · plugin architecture |
| [`satellites/DECISION_SYSTEM_implementation_integration.md`](satellites/DECISION_SYSTEM_implementation_integration.md) | Nexus · governance/HITL · observability · recovery · Critic migration · security hardening |
| [`satellites/DECISION_VERIFICATION_implementation_pipeline.md`](satellites/DECISION_VERIFICATION_implementation_pipeline.md) | Verification pipeline + stage migration rows |
| [`satellites/DECISION_DELIBERATION_implementation_strategies.md`](satellites/DECISION_DELIBERATION_implementation_strategies.md) | DecisionStrategy · Council strategy rows |

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

| Phase | Status | Detail satellite |
| ----- | ------ | ---------------- |
| **DS-CORE** | PLANNED | [`implementation_core`](satellites/DECISION_SYSTEM_implementation_core.md) |
| **DS-VER-PIPE / DS-VER-STAGES** | PLANNED | [`DECISION_VERIFICATION_implementation_pipeline`](satellites/DECISION_VERIFICATION_implementation_pipeline.md) |
| **DS-DELIB / DS-COUNCIL** | PLANNED | [`DECISION_DELIBERATION_implementation_strategies`](satellites/DECISION_DELIBERATION_implementation_strategies.md) |
| **DS-MIG** (Critic clean cut) | PLANNED | [`implementation_integration`](satellites/DECISION_SYSTEM_implementation_integration.md) |
| **DS-E2E** (Docker qualification) | PLANNED | **below — blocking production gate** |

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

Re-owned from [`CRITIC_VERIFICATION` plan](CRITIC_VERIFICATION.md) Protocol v2 findings. Full rows: [`DECISION_VERIFICATION_implementation_pipeline.md`](satellites/DECISION_VERIFICATION_implementation_pipeline.md) · [`implementation_integration`](satellites/DECISION_SYSTEM_implementation_integration.md).

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

One **DS-\*** ID per PR → update the owning satellite row → documentation gates green → no `shipped` claim until runtime slice lands.
