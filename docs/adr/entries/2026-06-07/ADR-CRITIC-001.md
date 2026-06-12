# ADR-CRITIC-001: Critic & Verification Layer — tier-separated PEV verify stack

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-07 |
| **Deciders** | Harness platform architecture |
| **Related** | [`architecture/CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) · canon §55 · Phase CRIT-V |

## Context

Intergrax needs a production-grade mechanism to verify partial and final agent outputs (LLM-as-judge, rule-based checks, trajectory evaluation, human gates). An architecture audit (2026-06-07) found:

- Strong L0 structural validation (`NexusValidationEngine`) and evaluation **wiring** (Phase EVAL Done).
- No platform primitive for semantic judging or trajectory evaluation.
- LLM-as-judge explicitly documented as opt-in with scores supplied externally.
- Evaluation audit layer at **L2** maturity — closeout ≠ execution depth.

Alternatives considered:

1. **Monolithic platform critic in Nexus** — single LLM call in `NexusLoop` for all domains.
2. **Application-only critics** — each Tier-3 app implements full verify stack ad hoc.
3. **Tier-separated CVL (chosen)** — Harness provides orchestration + Tier-0 tools; agents/apps provide rubrics and policy.

## Decision

Adopt the **Critic & Verification Layer (CVL)** with a **three-layer stack (L0/L1/L2)** and **strict tier separation**:

| Tier | Owns |
|------|------|
| **Tier-0** | `eval.judge`, `eval.trajectory` tools; contracts; registry integration |
| **Tier-1** | `CriticOrchestrator`, `EvaluatorLoopExecutor`, hooks in graph/UAEP; policy bridge |
| **Tier-2** | Rubrics, ValidatorAgents, `Agent.validate()`, domain tests |
| **Tier-3** | `CriticProfile`, thresholds, golden datasets, when to require L1/L2 |

**Rejected:**

- **Monolithic Nexus critic** — violates fat-nexus anti-pattern; cannot encode domain rubrics correctly.
- **Application-only** — duplicates infrastructure; breaks observability and release gates.

**LLM-as-judge is opt-in** via `CriticProfile` — not mandatory on every run (consistent with [`architecture/NEXUS_EXECUTION_FLOW.md`](../../architecture/NEXUS_EXECUTION_FLOW.md) §18).

**L0 always runs before L1** when L1 is enabled.

## Consequences

### Positive

- Clear author map for partial vs final verification.
- Reuses `NexusValidationEngine`, `OnlineEvaluationRegistry`, `ReplayEngine`, `EvaluationProfile`.
- Aligns with PEV and industry harness patterns (LangGraph verify nodes + eval observability).
- Enables Evaluation layer uplift L2→L3 without new universal stores.

### Negative

- Additional Tier-1 module surface (`runtime/critic/`).
- Authors must configure `CriticProfile` for semantic verification — not zero-config.
- Judge calibration against human baseline remains operational work (documented, not automated in v1).

## Compliance

- Tier boundaries preserved — no domain logic in Tier-1; agents use ToolRuntime for L1.
- Extends existing evaluation control plane (Appendix U), does not fork it.
- ADR linked from canon §55 and plan Phase CRIT-V.

## Implementation notes

- Architecture: [`architecture/CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md)
- Plan: Phase CRIT-V (Band 2ak) — waves CRIT-V-0 through CRIT-V-7
- Verification: `uv run pytest -m gate -q` after each wave; critic-specific tests under `tests/unit/runtime/critic/`
