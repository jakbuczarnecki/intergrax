# Adaptive Harness Intelligence

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Audit layers:** L4 AHI  
**Audit instruction:** [`audit/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../maintainers/audit/ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Last updated:** 2026-06-22 — **AHI-ADAS-00** ADAS canonical section + ADR-ADAPT-002; **P2-ARCH-10** AHI governance boundary; **Full Harness LC** (re-validates W-ADAPT); **70/70 Done**

### L4 Frozen cross-domain index (AHI-MAINT-04)

| Item | Owner domain | Plan row | Notes |
|------|--------------|----------|-------|
| GAP-CTX-12 adaptive context ranking | AHI (Frozen) | AHI-MAINT-04 | No CE-owned auto-ranking |
| M-RAG.58 / GAP-RAG-15 adaptive retriever selection | AHI (Frozen) | [`RAG-MAINT-04`](../maintainers/plans/RAG.md#61av-harness-implementation-queue--rag-audit-maintenance-planned) | No RAG-owned implementation |
| CVL L4 adaptive critic thresholds | AHI (Frozen) | CVL-MAINT-02 | Product gate before auto-apply |

**Product gate (AHI-MAINT-01):** L4 threshold auto-apply requires explicit product decision — evidence bundle via `phase_w_adapt_report.py`.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ADAPTIVE_HARNESS_INTELLIGENCE canon).

- **Implement / audit default:** L4 adaptive loop contracts (§1–§7). Extended §8+: [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md). ADAS sub-capability: [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md) (on demand).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../technical/guides/audit_slices/ADAPTIVE_HARNESS_INTELLIGENCE.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---
## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_extended_depth.md) | extended depth |
| [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md) | ADAS / Agent Design Search sub-capability |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


## Table of contents

1. [Executive summary](.#1-executive-summary)
2. [Strategic business case](.#2-strategic-business-case)
3. [Problem statement and market gap](.#3-problem-statement-and-market-gap)
4. [Terminology — Adaptive Harness Intelligence vs classical RL](.#4-terminology--adaptive-harness-intelligence-vs-classical-rl)
5. [Audit of current Intergrax state](.#5-audit-of-current-intergrax-state)
6. [Gap analysis](.#6-gap-analysis)
7. [Vision and design principles](.#7-vision-and-design-principles)
8. [Target architecture overview](.#8-target-architecture-overview)
9. [Adaptive Control Plane — component specification](.#9-adaptive-control-plane--component-specification)
10. [Signal model and utility function](.#10-signal-model-and-utility-function)
11. [Adaptation loops — four canonical kinds](.#11-adaptation-loops--four-canonical-kinds)
12. [Lifecycle modes — Observe through Verify](.#12-lifecycle-modes--observe-through-verify)
13. [Process pattern intelligence](.#13-process-pattern-intelligence)
14. [Integration with existing Intergrax subsystems](.#14-integration-with-existing-intergrax-subsystems)
15. [Tier placement and dependency rules](.#15-tier-placement-and-dependency-rules)
16. [Security, governance, and human-in-the-loop](.#16-security-governance-and-human-in-the-loop) *(planned — see [Governance Boundary](.#governance-boundary))*
28. [Governance Boundary](.#governance-boundary)
29. [Allowed AHI actions](.#allowed-ahi-actions)
30. [Disallowed AHI actions](.#disallowed-ahi-actions)
31. [AHI change lifecycle](.#ahi-change-lifecycle)
32. [Change risk classes](.#change-risk-classes)
33. [Production auto-apply rule](.#production-auto-apply-rule)
34. [Cursor review checklist](.#cursor-review-checklist)
17. [Data contracts (Pydantic reference)](.#17-data-contracts-pydantic-reference)
18. [End-to-end flow diagrams](.#18-end-to-end-flow-diagrams)
19. [Phased implementation roadmap — Phase W-ADAPT](.#19-phased-implementation-roadmap--phase-w-adapt)
20. [KPIs, acceptance gates, and L4 evidence](.#20-kpis-acceptance-gates-and-l4-evidence)
21. [Operational model](.#21-operational-model)
22. [Risks, anti-patterns, and mitigations](.#22-risks-anti-patterns-and-mitigations)
23. [Competitive differentiation summary](.#23-competitive-differentiation-summary)
24. [Conclusions and recommendations](.#24-conclusions-and-recommendations)
25. [Appendix A — Mapping to existing code](.#appendix-a--mapping-to-existing-code)
26. [Appendix B — Proposed implementation plan task IDs](.#appendix-b--proposed-implementation-plan-task-ids)
27. [Appendix C — ADR decision record](.#appendix-c--adr-decision-record)
35. [ADAS — Agent Design Search (sub-capability)](.#adas--agent-design-search-sub-capability)

---

## 1. Executive summary

Intergrax is a **Harness AI platform** — the durable product is the runtime, not any single agent. Most industry harnesses (Cursor-class IDE agents, Codex-style coding harnesses, Viktor-style automation, Google ADK-style labs) optimize for **run → trace → manual tuning**. They do not provide a **governed, auditable, closed-loop path** from production telemetry to bounded runtime improvement.

**Adaptive Harness Intelligence (AHI)** is Intergrax's proposed answer: a Tier-1 **Adaptive Control Plane** that:

1. **Observes** execution outcomes from trace, metrics, evaluation, cost, and human-in-the-loop signals.
2. **Proposes** bounded configuration changes (routing, orchestration, RAG, policy fragments) as versioned artifacts.
3. **Validates** proposals through existing governance envelopes, capability-graph impact analysis, and regression suites.
4. **Applies** approved changes through shadow → canary → production promotion — never bypassing `PolicyEngine`.
5. **Verifies** measurable improvement over baseline before declaring L4 readiness.

This is **not classical reinforcement learning** (policy-gradient training, unconstrained reward maximization, black-box model updates). It is **evidence-driven harness adaptation**: contextual bandits, rule-based tuning, statistical regression gates, and human-governed policy learning — aligned with IDEAL §25 and the L4 maturity model.

**Strategic verdict:** AHI is **justified, feasible, and differentiated** if implemented as a governed platform capability (Phase W-ADAPT), not as an opaque "self-learning agent."

**Current state:** Intergrax has **L3 production harness** plus **L4 governance contracts** (`adaptive_governance.py`, maturity gates, evaluation registry). It does **not** yet have a runtime **AdaptationExecutor** that closes the loop with measurable improvement.

---

## 2. Strategic business case

### 2.1 Core business question

> Can Intergrax become a harness that **gets measurably better at running agents** — discovering efficient paths, reducing cost, improving quality, surfacing hidden workflow patterns — without sacrificing auditability or human control?

### 2.2 Value proposition

| Stakeholder | Value |
|-------------|-------|
| **Platform team** | Reduced manual tuning; evidence-based promotion of profile changes |
| **Agent authors** | Faster time-to-quality via recommended skill/routing profiles |
| **Operations / SRE** | Regression detection + bounded auto-remediation within policy |
| **Security / compliance** | Every adaptation is versioned, gated, rollback-ready |
| **Product leadership** | Defensible differentiator vs commodity agent runtimes |
| **Business applications (Tier-3)** | Optional business-outcome signals feed harness utility without polluting Tier-1 |

### 2.3 Alignment with Intergrax strategic lock

From [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md):

- **Harness is the durable product; agents are replaceable execution units.**
- **Laboratory** optimizes hypothesis speed; **production harness** optimizes governed repeatability.
- Evolution is **evidence-driven** (L0–L4), not declaration-driven.

AHI extends the laboratory workflow (`ExperimentSession` → KEEP/DISCARD) into **continuous production learning** with the same evidence discipline.

### 2.4 What success looks like (12-month horizon)

1. **≥ 10% improvement** in composite utility `U` (§10) on golden scenarios after adaptive routing vs static baseline.
2. **Zero unapproved policy mutations** in production (100% envelope compliance).
3. **≥ 3 process patterns** per active tenant surfaced monthly with human-reviewed skill proposals.
4. **Rollback time < 5 minutes** for any applied profile version via registry pointer swap.
5. **L4 gate evidence** satisfies Phase V criteria: closed-loop improvement documented in evaluation registry trends.

### 2.5 What AHI is not (scope boundary)

| Out of scope | Reason |
|--------------|--------|
| Training foundation models inside Intergrax | Tier-0 LLM adapters remain replaceable providers |
| Autonomous business strategy (Problem Radar product logic) | Tier-2 agent / Tier-3 application scope |
| Unsupervised policy drift | Violates policy-first architecture |
| Multi-tenant SaaS billing for "AI learning" | Future product layer |
| Deep RL GPU training pipelines | Wrong tool; governance and auditability fail |

---

## 3. Problem statement and market gap

### 3.1 Industry default harness loop

```text
Static config → Run agent → Trace/logs → Human edits config → Repeat
```

Pain points:

- **Configuration debt** accumulates across agents, skills, prompts, and RAG profiles.
- **Hidden inefficiencies** (wrong model tier, excessive tool fanout, suboptimal RAG depth) are invisible until cost incidents.
- **Process knowledge** stays in traces nobody mines.
- **No safe automation** for tuning — teams either don't tune or tune without regression gates.

### 3.2 Intergrax opportunity

Intergrax already invested in the **prerequisites** competitors often lack together:

| Capability | Intergrax artifact |
|------------|-------------------|
| Full execution trace | `RunTraceWriter`, `RuntimeEvent` |
| Policy-first execution | `PolicyEngine`, `RuntimePolicyBundle` |
| Evaluation registry | `online_evaluation_registry`, `evaluation_registry_trends` |
| Agent lifecycle promotion | `agent_promotion.py` |
| Capability graph impact | `capability_graph_*` |
| Bounded adaptive envelopes | `adaptive_governance.py` |
| Shadow evaluation hook | `RuntimeArchitectureGovernanceBridge.record_shadow_run_evaluation` |
| Cost governance | `cost_budget.py`, `cost_optimization.py` |
| Experiment lab | `ExperimentSession` |

The gap is the **missing middle layer** that turns signals into **approved, versioned, verified** harness mutations.

### 3.3 Three intelligence problems (kept separate)

AHI addresses three related but distinct problems:

| # | Problem | Primary owner | AHI component |
|---|---------|---------------|---------------|
| P1 | **Operational adaptation** — better routing, retry, RAG, cost | Tier-1 Adaptive Control Plane | `AdaptationEngine`, `AdaptationExecutor` |
| P2 | **Process pattern discovery** — recurring tool/agent/HITL sequences | Tier-1 `ProcessPatternMiner` | Emits proposals; Tier-2 implements |
| P3 | **Strategic market intelligence** — forces, pains, opportunities | Tier-2 agents (e.g. Problem Radar) | Consumes AHI outputs optionally |

**Architectural rule:** Tier-1 MUST remain domain-agnostic. P3 never lives inside Nexus core loops.

---

## 4. Terminology — Adaptive Harness Intelligence vs classical RL

### 4.1 Definitions

| Term | Definition |
|------|------------|
| **Adaptive Harness Intelligence (AHI)** | The Intergrax platform capability that improves harness behavior over time through governed closed loops |
| **Adaptive Control Plane (ACP)** | Tier-1 subsystem implementing observe → propose → gate → apply → verify |
| **Harness Outcome Signal** | Normalized post-run measurement bundle feeding adaptation decisions |
| **Utility function U** | Weighted composite score for comparing candidate vs baseline configurations |
| **Adaptive Loop** | A bounded category of harness changes (`AdaptiveLoopKind` in existing contracts) |
| **Profile Version** | Immutable snapshot of a tunable configuration artifact with rollback pointer |
| **Classical RL** | MDP-based learning with policy parameters updated via reward gradients — **not the AHI implementation model** |

### 4.2 Comparison table

| Dimension | Classical RL | Adaptive Harness Intelligence |
|-----------|--------------|-------------------------------|
| Optimization target | Expected cumulative reward | Bounded utility improvement on golden + online eval |
| Action space | Continuous / high-dimensional policy | Discrete profile versions in registries |
| Exploration | Epsilon-greedy, entropy bonus | Shadow runs + canary traffic |
| Safety | Reward shaping, constraints | `PolicyEngine` + `AdaptiveLoopEnvelope` + human gate |
| Auditability | Often opaque | Full trace + proposal ID + version lineage |
| Update frequency | Every episode / batch | Async scheduler; cooldown per envelope |
| Failure mode | Reward hacking | Regression suite block + rollback |

### 4.3 Acceptable learning algorithms inside AHI

| Algorithm class | Use case | Tier |
|-----------------|----------|------|
| **Contextual bandits (Thompson sampling, UCB)** | Model routing, RAG tier selection | Tier-1 `AdaptationEngine` |
| **Rule-based thresholds** | Step explosion, cost spike response | Tier-1 (extends `HistoryAwareEvaluator`) |
| **Statistical process control** | Anomaly-triggered recommendations | Tier-1 (extends `cost_forecast.py`) |
| **Offline policy improvement** | Batch analysis → proposal | Tier-1 scheduler job |
| **Frequent sequence mining** | Process patterns in traces | Tier-1 `ProcessPatternMiner` |
| **LLM-as-judge** | Quality signal input only | Tier-1 eval subsystem (existing) |
| **Deep RL / neural policy** | — | **Rejected for AHI v1** |

---

## 5. Audit of current Intergrax state

### 5.1 Maturity model recap

From [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md):

| Level | Meaning |
|-------|---------|
| L0 | Fragmented — no consistent model |
| L1 | Operational MVP |
| L2 | Scalable harness — modular, registered, testable |
| L3 | Production harness OS — policy, SLOs, runbooks, evaluation ops |
| **L4** | **Adaptive Agent OS — closed feedback loops, bounded self-tuning** |

### 5.2 Layer-by-layer readiness for AHI

| Layer | Current score | Evidence | AHI readiness |
|-------|---------------|----------|---------------|
| Observability / trace | L3 | `RunTraceWriter`, OTLP, modality metrics | ✅ Signal source ready |
| Evaluation | L3 | `evaluation_automation.py`, online registry, trends | ✅ Baseline/candidate compare ready |
| Policy / governance | L3–L4 contracts | `PolicyEngine`, `adaptive_governance.py` | ⚠️ Gate ready; executor missing |
| Cost governance | L3 | Budget, quota, forecast, optimization report | ⚠️ Recommendations only |
| Orchestration | L3–L4 | Nexus loop, graphs, coordination catalog | ⚠️ Static profiles |
| RAG routing | L2 | `QueryRouter` heuristics | ⚠️ Not learned |
| Memory | L3 | `MemoryView`, task memory | ⚠️ No org-level pattern store |
| Agent promotion | L3 | `agent_promotion.py` | ✅ Pattern reusable for profiles |
| **Adaptive Harness (IDEAL §25)** | **L1–L2** | Contracts + CI gate evidence | ❌ Runtime loop not closed |

### 5.3 Existing artifacts directly reusable

```
intergrax/runtime/architecture/
├── adaptive_governance.py          # AdaptiveLoopEnvelope, gate rules
├── maturity_gate_evidence.py       # L4 evidence aggregator
├── runtime_governance_bridge.py    # Shadow eval, trace metadata
├── online_evaluation_registry.py   # Observation persistence
├── evaluation_registry_trends.py     # Release comparisons
├── cost_optimization.py            # Recommendation generator
├── agent_promotion.py              # Promotion evidence pattern
└── capability_graph_*.py           # Impact analysis

intergrax/runtime/governance/
├── execution_guard.py              # Post-run governance
└── history_evaluator.py            # Historical regression signals

intergrax/experiments/workflow.py     # Lab session pattern
intergrax/rag/routing/query_router.py # Tunable RAG tier target
```

### 5.4 What Phase V L4 closeout actually validates today

`scripts/release/phase_v_closeout_gate.py --enforce-l4` confirms:

- Adaptive **proposal envelopes** pass governance rules (`build_default_adaptive_proposals()`).
- Maturity gate inputs include `adaptive_governance_passed`.

It does **not** confirm:

- Measurable quality/cost improvement over baseline in production.
- Automatic application of approved adaptations.
- Continuous signal collection driving proposals.

**Interpretation:** Intergrax achieved **L4 governance readiness** (Phase V) and **L4 adaptive runtime readiness** (Phase W-ADAPT, 70/70 Done).

---

## 6. Gap analysis

> **Historical audit (2026-06-05).** All gaps below were closed by Phase W-ADAPT (Wave 0–7). For current delivery status see [§19](.#19-phased-implementation-roadmap--phase-w-adapt) and canon [§54.3](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#543-implementation-state-phase-w-adapt--done).

### 6.1 Missing components (must build) — **Done (W-ADAPT)**

| ID | Component | Purpose |
|----|-----------|---------|
| G1 | `HarnessOutcomeSignal` + `SignalCollector` | Normalize post-run inputs to adaptation engine |
| G2 | `AdaptationEngine` | Generate `AdaptiveLoopProposal` from signals + bandit state |
| G3 | `ProfileVersionStore` | Immutable versioned profiles with rollback pointers |
| G4 | `AdaptationExecutor` | Apply approved changes to registry-bound profiles |
| G5 | `AdaptationScheduler` | Async worker — never blocks hot execution path |
| G6 | `VerificationLoop` | Post-apply regression + SLO window check |
| G7 | `ProcessPatternMiner` | Trace sequence mining → `ProcessPatternProposal` |
| G8 | Ops/report API | `phase_w_adapt_report.py` + optional HTTP debug routes |

### 6.2 Partial components (must extend) — **Done (W-ADAPT)**

| Component | Gap |
|-----------|-----|
| `RuntimeArchitectureGovernanceBridge` | Add `submit_proposal`, `apply_approved`, `rollback` |
| `cost_optimization.py` | Wire into `AdaptationEngine` as proposal source |
| `QueryRouter` | Accept bandit-weighted tier overrides from profile version |
| `ApplicationEnvironmentProfile` | Add `AdaptiveProfile` section (weights, enabled loops, authority) |
| `ExecutionGuard` | Emit structured signals into `SignalCollector` |

### 6.3 Documentation / planning gaps — **Done**

| Gap | Resolution |
|-----|------------|
| IDEAL L4 policy learning marked out of scope in plan | Add Phase W-ADAPT from this RFC |
| No single AHI architecture doc | This document |
| §34 Evaluation Model lacks closed-loop semantics | Extended in canon §54 |

---

## 7. Vision and design principles

### 7.1 Vision statement

> **Intergrax harness learns from every governed run — not by mutating opaque models, but by proposing auditable profile improvements that make agents cheaper, safer, and more effective over time.**

### 7.2 Design principles (normative)

1. **Policy-first adaptation** — no change bypasses `PolicyEngine`.
2. **Reuse Tier-0** — one trace system, one eval registry, one policy stack (canon §5.2, §8.8).
3. **Hot path stays deterministic** — adaptation runs async; per-run bandit choices read precomputed weights only.
4. **Human-governed autonomy** — authority matrix defaults conservative; `POLICY_LEARNING` requires approver.
5. **Evidence over declaration** — L4 requires measured improvement, not passing unit tests on sample proposals.
6. **Rollback by default** — every apply creates rollback pointer; auto-rollback on verification failure.
7. **Tier-1 domain-agnostic** — business semantics enter via optional Tier-3 outcome webhooks only.
8. **Capability graph before apply** — blast-radius analysis mandatory for skill/policy changes.
9. **Shadow before production** — no skip of shadow/canary except OBSERVE mode.
10. **Extend, don't duplicate** — evolve `adaptive_governance.py`, don't create parallel governance stack.

---

## ADAS — Agent Design Search (sub-capability)

**ADAS / Agent Design Search** is a governed **AHI sub-capability** for systematic discovery, evaluation, archival, and promotion of agent candidates — aligned with meta-agent-building trends without bypassing harness governance.

It extends (does not replace) the profile adaptation loop: same **observe → propose → validate → shadow → canary → apply/promote → verify** lifecycle, but targets **agent candidates** rather than runtime profile versions.

| Rule | Constraint |
|------|------------|
| Placement | Inside AHI Tier-1 — **not** a separate top-level harness layer |
| Scope | **Not** Tier-3-only; optional Tier-3 ADAS Lab is operator UI only |
| Mutation | **No** direct production agent mutation — scaffold → static gate → evaluation → archive → governed promotion |
| Strategy | MAS (Meta Agent Search) is one replaceable `AgentDesignStrategy`; it does not own the control plane |

**Canonical detail:** [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)  
**ADR:** [ADR-ADAPT-002](../technical/adr/entries/2026-06-22/ADR-ADAPT-002.md) — ADAS inside AHI, not separate layer
**Plan:** Phase **AHI-ADAS** in [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md)

---
