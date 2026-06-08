**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix K — Adaptive Harness Intelligence traceability (Phase W-ADAPT)

**Purpose:** 100% mapping from [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) (AHIA) to concrete **W-ADAPT.\*** IDs. **Canonical phase narrative:** [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime).

**Status:** **70/70 Done** (Band 2y closed 2026-06-05) — Waves W-ADAPT-0 through W-ADAPT-7 complete.

### K.1 AHIA component → W-ADAPT ID matrix

| AHIA component (§9) | Existing module to reuse | W-ADAPT ID |
|---------------------|--------------------------|------------|
| SignalCollector | `metrics/export.py`, `execution_guard.py`, `online_evaluation_registry.py` | W-ADAPT-1.4–1.11 |
| HarnessOutcomeSignal + utility | — (new) | W-ADAPT-1.1, W-ADAPT-1.8 |
| SignalStore | — (new SQLite) | W-ADAPT-1.3 |
| BanditStateStore | — (new) | W-ADAPT-2.1 |
| RoutingTuningEngine | `rag/routing/query_router.py`, LLM profiles | W-ADAPT-2.2, W-ADAPT-3.7, W-ADAPT-4.10 |
| ExecutionStrategyEngine | `history_evaluator.py`, `nexus_factory.py` | W-ADAPT-2.3, W-ADAPT-4.10 |
| PolicyLearningEngine | `adaptive_governance.py`, `tool_security.py` | W-ADAPT-2.4, W-ADAPT-4.6, W-ADAPT-4.9 |
| EvaluationFeedbackEngine | `evaluation_registry_trends.py` | W-ADAPT-2.5, W-ADAPT-5.3 |
| ProposalBuilder | `adaptive_governance.py` (`AdaptiveLoopProposal`) | W-ADAPT-2.6 |
| AdaptationEngine facade | — (new) | W-ADAPT-2.7 |
| Governance gate | `adaptive_governance.py`, `capability_graph_compatibility.py` | W-ADAPT-2.8–2.9 |
| ProfileVersionStore | — (new; pattern from `agent_promotion.py`) | W-ADAPT-3.1–3.2, W-ADAPT-3.5 |
| AdaptationExecutor | `runtime_governance_bridge.py` (extend) | W-ADAPT-3.3–3.4, W-ADAPT-4.4–4.5, W-ADAPT-4.8 |
| VerificationLoop | `evaluation_registry_trends.py`, `execution_guard.py` | W-ADAPT-5.1–5.5 |
| ProcessPatternMiner | trace persistence | W-ADAPT-6.* |
| AdaptationScheduler | Celery/message bus pattern from W-ML | W-ADAPT-2.12, W-ADAPT-5.12, W-ADAPT-6.5 |
| AdaptiveProfile (Tier-3) | `environment_profile.py` | W-ADAPT-4.1, W-ADAPT-7.1–7.2 |
| Ops reports / CI | `phase_v_governance_report.py` pattern | W-ADAPT-1.12, W-ADAPT-2.11, W-ADAPT-5.6–5.8 |
| Runtime L4 evidence | `maturity_gate_evidence.py` | W-ADAPT-5.7, W-ADAPT-5.11 |
| Author docs | AGENT_CREATION_GUIDE appendices | W-ADAPT-7.3–7.4 |

### K.2 Adaptive loop kind → implementation wave

| `AdaptiveLoopKind` | Engine | Apply wave | Authority default |
|--------------------|--------|------------|-------------------|
| `ROUTING_TUNING` | W-ADAPT-2.2 | W-ADAPT-4.10 | RECOMMEND |
| `EXECUTION_STRATEGY_TUNING` | W-ADAPT-2.3 | W-ADAPT-4.10 | RECOMMEND |
| `POLICY_LEARNING` | W-ADAPT-2.4 | W-ADAPT-4.6, W-ADAPT-4.9 | AUTO_WITH_HUMAN_GATE |
| `EVALUATION_FEEDBACK` | W-ADAPT-2.5 | observe only (W-ADAPT-5.3) | OBSERVE_ONLY |

### K.3 Lifecycle mode → task coverage

| Mode | Code | Primary tasks |
|------|------|---------------|
| Observe | L4-O | W-ADAPT-1.* |
| Recommend | L4-R | W-ADAPT-2.* |
| Shadow | L4-S | W-ADAPT-3.* |
| Canary | L4-C | W-ADAPT-4.3 |
| Apply | L4-A | W-ADAPT-4.4–4.10 |
| Verify | L4-V | W-ADAPT-5.* |

### K.4 Paydown log

| Date | W-ADAPT ID | Summary |
|------|------------|---------|
| 2026-06-05 | W-ADAPT-1.1–1.12 | Observe (L4-O): contracts, SignalStore, SignalCollector, Nexus/Runtime hooks, `phase_w_adapt_report.py` |
| 2026-06-05 | W-ADAPT-0.2–0.5 | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold + gate import tests |
| 2026-06-05 | W-ADAPT-0.1 | Phase W-ADAPT register + §6.1t + §6.2ac + Appendix K + Band 2y |
| 2026-06-02 | W-ADAPT-2.1–2.12 | Recommend (L4-R): AdaptationEngine, ProposalBuilder, bandit store, proposal report |
| 2026-06-02 | W-ADAPT-3.1–3.7 | Shadow (L4-S): ProfileVersionStore, shadow executor, integration tests |
| 2026-06-02 | W-ADAPT-4.1–4.10 | Apply (L4-A): canary, apply, rollback, policy-learning HITL |
| 2026-06-02 | W-ADAPT-5.1–5.12 | Verify (L4-V): VerificationLoop, auto-rollback, L4 runtime closeout gate, runbooks |
| 2026-06-02 | W-ADAPT-6.1–6.5 | ProcessPatternMiner, trace sequence reader, pattern report export |
| 2026-06-02 | W-ADAPT-7.1–7.7 | Tier-3 AdaptiveProfile wiring, debug routes, business outcome webhook, acceptance E2E |
| 2026-06-02 | W-ADAPT-OPS | Lab L4-O observe default (`LAB_ADAPTIVE_OBSERVE`); CI/release `--enforce-l4-runtime`; canon §54 + AHIA sync |

---
