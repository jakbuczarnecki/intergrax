# CRITIC_VERIFICATION — §11+ scenarios & control

**Parent hub:** [`CRITIC_VERIFICATION.md`](../CRITIC_VERIFICATION.md)

## 11. Policy and governance

`critic_governance` fragment in `RuntimePolicyBundle` (Tier-3 merge):

| Policy key | Effect |
|------------|--------|
| `require_l0_on_all_nodes` | Default true |
| `semantic_judge_min_risk` | Enable L1 only above risk tier |
| `block_complete_on_critic_fail` | Terminal gate |
| `critic_cost_budget_tokens` | Cap L1 spend per run |
| `human_review_on_borderline` | Score in [0.6, threshold) → HITL |

Integrates with existing `RuntimePolicyEngine` — no agent-specific branches.

---

## 12. Observability

| Event | Trace step | Runtime event |
|-------|------------|---------------|
| L0 fail | `critic.l0_failed` | `VALIDATION_ERROR` |
| L1 judge | `critic.l1_judge` | `LLM_CALL` (critic profile) |
| Trajectory eval | `critic.trajectory` | `STEP_COMPLETED` |
| Evaluator-loop iteration | `critic.evaluator_loop` | custom tag |
| Final verdict | `critic.final_verdict` | maps to task lifecycle |

---

## 13. Maturity model (target)

| Level | CVL capability |
|-------|----------------|
| **L0** | Structural validation only (current baseline) |
| **L1** | L0 + registry wiring (Phase EVAL — Done) |
| **L2** | L1 + `eval.judge` + `CriticProfile` + partial hooks |
| **L3** | L2 + trajectory eval + evaluator-loop executor + semantic offline runner |
| **L4** | L3 + adaptive critic threshold proposals + human-calibrated judge baseline in CI |

**Current:** **L3+** (CRIT-V-0…7 + FOLLOWUP complete, 2026-06-13 layer completion audit). **Next:** L4 adaptive critic thresholds (deferred — AHIA / product gate).

---

## 14. Non-goals (Phase CRIT-V)

- Universal mandatory LLM-judge on every production run.
- Domain rubric library in Tier-0/Tier-1.
- Replacing human compliance workflows.
- Second evaluation registry or trace system.
- FLOW-8 reference product app (remains §6.3 deferred) — CRIT-V may use lab harness only.

---

## 15. Relationship to industry patterns

| Pattern | CVL mapping |
|---------|-------------|
| **LangGraph checkpoint/validation nodes** | `CriticOrchestrator` + graph hooks |
| **LangSmith evaluators + datasets** | `OnlineEvaluationRegistry` + `NexusEvalRunner` |
| **PEV (Plan-Execute-Verify)** | CVL = Verify phase infrastructure |
| **Agent-as-judge / LLM-as-judge** | `eval.judge` + ValidatorAgent nodes |
| **Trajectory evaluation** | `eval.trajectory` + ReplayEngine |

---

## 16. Implementation tracking

See [`plan/CRITIC_VERIFICATION.md) — **Phase CRIT-V**.

| Wave | Focus | Status |
|------|-------|--------|
| CRIT-V-0 | This document, ADR-CRITIC-001, canon §55, README | **Done** |
| CRIT-V-1 | Contracts + `CriticProfile` | **Done** |
| CRIT-V-2 | Tier-0 tools `eval.judge`, `eval.trajectory` | **Done** |
| CRIT-V-3 | `CriticOrchestrator` + graph hooks + UAEP step hook | **Done** |
| CRIT-V-4 | `EvaluatorLoopExecutor` | **Done** |
| CRIT-V-5 | `NexusEvalRunner` semantic mode | **Done** |
| CRIT-V-6 | Tier-3 wiring + policy bundle + CI | **Done** |
| CRIT-V-7 | FAUDIT-EVAL.1 baseline gate + docs Appendix W | **Done** |
| CRIT-V-FOLLOWUP | L1 tool client, L2 HITL, UAEP hook, policy bridge | **Done** |

---

## 17. Forbidden patterns

- **Fat Critic Nexus** — domain rubrics or revise logic in Tier-1.
- **Self-judge** — same LLM profile for producer and critic without policy override.
- **L1-only verification** — skipping L0 for speed.
- **Silent pass** — critic disabled but terminal `COMPLETED` on high-risk tasks when `require_critic_on_completion=true`.
- **Duplicate registry** — critic scores stored outside `OnlineEvaluationRegistry`.

---

## 18. References

- Canon §29 Validation Model · §42.43 Multi-Agent Flow · §53.10 Coordination patterns
- [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §18 Evaluation hooks
- [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) Appendix U (Evaluation) · Appendix W (Critic)
- [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) — L4 verify loop consumes CVL signals

---

*Maintainer: update this file when Phase CRIT-V deliverables land; sync canon §55 and plan register.*
