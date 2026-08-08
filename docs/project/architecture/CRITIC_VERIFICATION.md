# Critic Verification

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/CRITIC_VERIFICATION.md`](../maintainers/plans/CRITIC_VERIFICATION.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 25 (verify depth)  
**Audit instruction:** [`audit/CRITIC_VERIFICATION.md`](../maintainers/audit/CRITIC_VERIFICATION.md)  
**Last updated:** 2026-06-20 — **P2-ARCH-08** Verification Safety Boundaries; **CRIT-V-0…7 + CVL-LC-1…4 Done (L3+)**
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CRITIC_VERIFICATION canon).

- **Implement / audit default:** CVL contracts + orchestrator + wiring (§1–§6). Extended §7+: [`satellites/CRITIC_VERIFICATION_extended_depth.md`](satellites/CRITIC_VERIFICATION_extended_depth.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/CRITIC_VERIFICATION.md`](../maintainers/plans/CRITIC_VERIFICATION.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/CRITIC_VERIFICATION.md`](../technical/guides/audit_slices/CRITIC_VERIFICATION.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/CRITIC_VERIFICATION_extended_depth.md`](satellites/CRITIC_VERIFICATION_extended_depth.md) | extended depth |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

## 1. Purpose

Define the **Critic & Verification Layer (CVL)** — the Harness AI subsystem that answers:

> **Is this partial or final agent output actually correct — structurally, procedurally, and (when configured) semantically?**

CVL completes the **Plan → Execute → Verify (PEV)** loop that leading Harness AI systems use in production. It **does not** embed domain business rules in Nexus. It provides **typed primitives, orchestration hooks, telemetry, and policy gates** so agents and applications can compose domain-specific critics safely.

**Strategic positioning:** The Harness owns **how** verification runs; agents and applications own **what** is verified.

---

## Boundary with Observability & Evaluation Control Plane (OECP)

| Concern | CVL | OECP ([`OBSERVABILITY.md`](OBSERVABILITY.md#observability--evaluation-control-plane)) |
|---------|-----|------|
| Scope | Per-run / per-step correctness verification | Continuous measurement, eval datasets, metrics, regression gates, perturbations, long-term measurement |
| Output | Verdicts and observations (`CriticVerdict`, validation results) | Evidence records, eval snapshots, metric results, regression views |
| Persistence | Emits through HOS; no private verification stores | Evidence Ledger + Eval Registry v2 (references canonical trace/journal) |
| Gating | L0/L1/L2 PEV gates within a run | Trace completeness, eval regression, release/canary gates across runs |

**CVL emits** verdicts and observations. **OECP stores** evidence, **compares** results, builds **regression views**, and supplies the basis for **gating** and **adaptation**. CVL does **not** own OECP; OECP does **not** replace critic orchestration. Extended contracts: [`satellites/OBSERVABILITY_extended_depth.md`](satellites/OBSERVABILITY_extended_depth.md).

---

## 2. Problem statement

Intergrax already had strong **structural validation** (`NexusValidationEngine`), **evaluation infrastructure** (registry, shadow eval, offline runner contracts), and **adaptive verification** (`VerificationLoop` for profile promotion). Before Phase CRIT-V (2026-06-07…2026-06-08), production-grade PEV **Verify** depth was missing:

| Gap (pre-CRIT-V) | Impact | Status |
|------------------|--------|--------|
| No universal **semantic judge primitive** (`eval.judge`) | LLM-as-judge scores supplied ad hoc | **Done** — `tools/providers/eval/judge.py` + `L1Gateway` |
| No **trajectory evaluation** contract | Process quality invisible to gates | **Done** — `eval.trajectory` (heuristic process scoring) |
| **Evaluator-loop** catalog pattern only | No critique→revise→re-evaluate executor | **Done** — `EvaluatorLoopExecutor` + graph wiring |
| **NexusEvalRunner** exact-match only | Offline benchmarks miss semantic equivalence | **Done** — optional `semantic_match_enabled` + `eval.judge` |
| **L0→L1→L2 stack** not explicit | Authors unclear where to put rubrics vs hooks | **Done** — §6 stack + `CriticOrchestrator` |
| Evaluation layer maturity **L2** (FAUDIT-32) | Closeout wiring ≠ execution depth | **Done** — CRIT-V uplift to **L3** |

CVL closed these gaps **without** violating tier boundaries or creating a second evaluation system parallel to `OnlineEvaluationRegistry`.

**Remaining depth (not blocking L3):** L4 adaptive critic thresholds (AHIA), LLM-based trajectory judge (`eval.trajectory_judge` skill), FLOW-8 product reference host — see plan backlog §CVL-Backlog.

---

## 3. Terminology

| Term | Meaning in Intergrax |
|------|----------------------|
| **Critic** | Any component that produces a scored verdict on output or trajectory |
| **Verification** | Harness-orchestrated application of critics with policy consequences (retry, revise, HITL, fail) |
| **L0 critic** | Deterministic — schema, rules, contract, executable tests |
| **L1 critic** | Probabilistic semantic — LLM-as-judge, secondary model, self-consistency |
| **L2 critic** | Authoritative — human expert, compliance sign-off, audit gate |
| **Partial verification** | After a graph node, subtask, or UAEP step milestone |
| **Final verification** | Before task terminal state (`COMPLETED`, `PARTIALLY_COMPLETED`) |
| **Evaluator-loop** | Multi-iteration critique→revise pattern until pass or budget exhausted |
| **CVL** | Critic & Verification Layer — platform subsystem (this document) |

**Not CVL:** Adaptive profile promotion (`VerificationLoop` in `runtime/adaptive/`) — complementary; consumes CVL/registry signals but serves L4 adaptation, not per-run correctness.

---

## 4. Design principles

1. **Reuse before create** — extend `NexusValidationEngine`, `ValidationResult`, `OnlineEvaluationRegistry`, `EvaluationProfile`, `ReplayEngine`; no parallel eval store.
2. **L0 before L1** — semantic judges run only after deterministic gates pass (cost + safety). Vendor **llm_guardrail** scans compose into L0 via `merge_guardrail_l0` when `guardrail_scan` is present in critic context ([`INTEGRATIONS.md`](INTEGRATIONS.md) §47).
3. **Judge separation** — critic LLM profile MUST differ from producer agent profile (model, temperature, prompt registry id).
4. **Opt-in by policy** — LLM-judge never mandatory on every run; `CriticProfile` + `EvaluationProfile` control activation.
5. **Trace everything** — every critic invocation emits trace + optional `OnlineEvaluationObservation`.
6. **Tier discipline** — Nexus orchestrates; Tier-2 supplies rubrics and ValidatorAgents; Tier-3 selects profiles and datasets.
7. **Fail closed on high risk** — when `require_critic_on_completion` is set and critic unavailable → `FAILED` or HITL, not silent pass.

---

## 5. Separated competencies (tier model)

### 5.1 Responsibility matrix

| Concern | Tier-0 Platform | Tier-1 Nexus / CVL | Tier-2 Agent | Tier-3 Application |
|---------|-----------------|-------------------|--------------|-------------------|
| `ValidationResult` contract | defines | consumes | extends via `validate()` | — |
| Structural validation (L0) | rules engine | `NexusValidationEngine` per node | `AgentContract.validation_rules` | `NexusPlan.validation_criteria` |
| Semantic judge primitive (L1) | `eval.judge` tool, rubric schema | `CriticOrchestrator` hook | rubric content, ValidatorAgent | enable + thresholds |
| Trajectory evaluation (L1) | `eval.trajectory` tool | hook after step/graph | domain step expectations | scenario definitions |
| Evaluator-loop execution | pattern + budget types | `EvaluatorLoopExecutor` | revise logic in worker agent | graph_spec nodes |
| Registry & trends | `OnlineEvaluationRegistry` | post-run bridge | — | `EvaluationProfile`, CI baselines |
| Release / adaptive gates | closeout scripts | `VerificationLoop` (L4) | — | `require_baseline_for_release` |
| HITL escalation (L2) | policy primitives | `HitlRunner` | interrupt reasons | approval policy |
| Golden datasets | runner contracts | `NexusEvalRunner` | eval cases | asset paths |
| Domain correctness | — | — | **primary owner** | orchestration + policy |

### 5.2 What Harness MUST NOT do

- Encode domain rubrics (“is this legal clause acceptable?”) in Tier-1.
- Force LLM-judge on every run regardless of `CriticProfile`.
- Replace ValidatorAgents with a monolithic platform critic.
- Bypass `NexusValidationEngine` for graph nodes.

### 5.3 What agents/applications MUST NOT do

- Implement parallel verification stores outside `OnlineEvaluationRegistry`.
- Call vendor LLM SDKs directly for judging (use `eval.judge` / `ToolRuntime`).
- Skip L0 validation and rely solely on LLM self-assessment.

---

## 6. Three-layer critic stack (L0 / L1 / L2)

```text
┌─────────────────────────────────────────────────────────────────┐
│ L2 — Authoritative                                              │
│ Human review · compliance sign-off · policy INTERRUPT → HITL    │
├─────────────────────────────────────────────────────────────────┤
│ L1 — Semantic (probabilistic)                                   │
│ eval.judge · eval.trajectory · ValidatorAgent · secondary model │
├─────────────────────────────────────────────────────────────────┤
│ L0 — Deterministic (always cheap, always first)                 │
│ schema · NexusValidationEngine · Agent.validate() · exec tests  │
└─────────────────────────────────────────────────────────────────┘
     fail fast ──► retry / revise          fail ──► HITL or FAILED
```

| Layer | Typical latency | Typical cost | When required |
|-------|-----------------|--------------|---------------|
| L0 | ms | negligible | Every graph node (default) |
| L1 | seconds | LLM tokens | When `CriticProfile.semantic_judge_enabled` |
| L2 | minutes–hours | human | High-risk policy or L1 borderline |

**Combined verdict:** `CriticVerdict.passed = L0.passed ∧ (L1.passed if enabled) ∧ (L2.passed if required)`.

---
