# Experimentation Workflow and Developer Experience

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 25–27, 30  
---

---

# 39. Implementation Rules For Cursor AI

When Cursor AI or an LLM coding agent implements Intergrax, it MUST follow these rules.

## 39.1 Always Preserve Layer Boundaries

Do not put orchestration logic into adapters.

Do not put business agent logic into Nexus.

Do not put platform lifecycle logic into agents.

---

## 39.2 Prefer Contracts Over Hardcoding

Use contracts, registries and schemas.

Avoid direct hardcoded branching such as:

```text
if task contains "vendor": run VendorAgent
```

Prefer capability matching.

---

## 39.3 Build Minimal Useful Runtime First

Initial implementation should focus on:

- AgentContract
- AgentRegistry
- Task object
- Nexus execution loop
- basic ToolRegistry
- basic TraceLogger
- simple adapter model
- one or two example agents

Do not build the entire platform prematurely.

---

## 39.4 Every New Agent Must Be Runnable Through Nexus

Agents should not be executed as standalone scripts except for isolated unit tests.

The normal path is:

```text
Task -> Nexus -> Agent -> Result -> Nexus
```

---

## 39.5 Every Agent Must Produce Structured Output

Agents must not return only raw text.

Raw text may exist as summary, but structured data is required for evaluation.

---

## 39.6 Every Execution Must Be Traceable

No hidden execution.

Every meaningful decision should produce a trace event or structured log.

---

## 39.7 Prefer Simple Internal UI

If a UI is needed, build a minimal debug/inspection surface.

Do not build a polished SaaS frontend at this stage.

---

## 39.8 Reuse Tier-0 — Never Duplicate Universal Mechanisms

Before writing code, Cursor AI and implementation agents MUST:

1. Identify whether the needed capability **already exists** in Tier-0 (§5.2.2).
2. Use the **canonical entry point** (LLM adapters, logging, tools, RAG, trace, memory, queues).
3. Implement **orchestration and domain logic only** in Tier-1 / Tier-2 / Tier-3.
4. **STOP and ask the human** if a new universal Tier-0 mechanism appears necessary (§5.2.4).

Cursor AI MUST NOT:

- add parallel LLM client wrappers,
- create agent-local logging or tracing systems,
- introduce duplicate tool registries or adapter facades,
- add new PostgreSQL/Redis/file clients in agents when Tier-0 adapters exist,
- implement §42 scaffold as standalone replacements for existing Nexus trace/tool/LLM paths.

When wiring §42 (events, hooks, UAEP), **integrate with** existing `RunTraceWriter`, `ToolRuntime`, `RuntimeEngine` — do not fork them.

---


---

# 40. Recommended Minimal First Implementation

The first implementation milestone should include:

```text
core/
    AgentContract
    AgentRegistry
    Task
    TaskState
    NexusRuntime
    ExecutionContext
    AgentExecutionResult
    ValidationResult
    TraceLogger

components/
    LlmProviderAdapter
    SlackAdapter interface placeholder
    TeamsAdapter interface placeholder
    StorageAdapter
    QueueAdapter placeholder

agents/
    EchoAgent
    ResearchAgent prototype
    ProblemRadarAgent prototype

applications/
    legal_application/          # host + serving + env config (composes agents/legal)
    <name>_application/         # future execution environments

runtime/
    NexusLoop
    TaskClassifier
    Planner
    AgentRouter
    ExecutionGraph
```

This is enough to validate the architecture.

Do not start with too many agents.

---


---

# 41. Minimal Runtime Flow

The first usable flow should be:

```text
1. User submits task
2. Nexus creates Task object
3. Nexus classifies task
4. Nexus creates simple plan
5. Nexus selects agent from registry
6. Nexus executes agent
7. Agent returns structured result
8. Nexus validates result
9. Nexus logs full trace
10. Nexus returns final response
```

This validates the entire skeleton.

---

---

---

# 42. Evaluation and Benchmarking Operations

Evaluation is a **first-class runtime subsystem**, not a post-hoc script.

## 42.1 Modes

| Mode | Purpose |
|------|---------|
| Offline | Golden datasets, regression before merge |
| Online | Production sampling, score trends |
| Shadow | Compare candidate path without user impact |
| Human | HITL rubric scoring |

## 42.2 Components

| Module | Role |
|--------|------|
| `runtime/architecture/evaluation_modes.py` | Mode contracts |
| `evaluation_automation.py` | Runner automation |
| `evaluation_registry_trends.py` | Score history / trends |
| `online_evaluation_registry.py` | Live eval registry |
| `evaluation_assets.py` | Golden asset catalog |
| `runtime/eval/` | NexusEvalRunner integration |

Evaluators: rule-based, schema, LLM-judge (see [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md)).

**Plan:** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) Phase EVAL · [`plan/CRITIC_VERIFICATION.md`](../plan/CRITIC_VERIFICATION.md) CRIT-V.

---

# 44. MVP-to-Product Evolution Layer

Every Intergrax product starts as a **prototype or MVP** on the same Harness stack. The platform MUST provide **systematic tools** for iterative design, evaluation, real-life and simulated testing, and evidence-based promotion to production — not ad-hoc scripts per team.

This layer is a **competitive differentiator**: developers ship fast; the Harness supplies feedback, gates, and promotion discipline.

## 44.1 Product maturity stages

```text
PROTOTYPE → MVP → BETA → PRODUCTION → OPTIMIZE (L4 / AHI)
```

| Stage | Goal | Harness posture | Evidence required |
|-------|------|-----------------|-------------------|
| **PROTOTYPE** | Validate idea in Nexus | `execution_mode=EXPLORATORY`, shadow eval, lab host | Smoke run + trace |
| **MVP** | First real users, narrow scope | `BALANCED`, offline + online eval sampling | Baseline eval scores, TTFRun |
| **BETA** | Scale testing, feedback loops | Stricter policy, HITL on risky paths | KPI trends, satisfaction samples |
| **PRODUCTION** | SLO-backed operation | `STRICT`, critic gates, reliability profiles | SLO window, incident budget, PRR |
| **OPTIMIZE** | Closed-loop improvement | AHI proposals, bounded policy learning | Eval deltas + human approval |

Promotion between stages is **evidence-driven** — see §44.5 and [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

## 44.2 Developer toolchain (as-built + target)

| Tool / surface | Role in MVP evolution | Status |
|----------------|----------------------|--------|
| `intergrax/scaffold` (`new-agent`, `new-stack`, `--minimal`) | Zero-to-first-run in minutes | **Done** — DX phase |
| `intergrax doctor` | Harness health before iterate | **Done** |
| `intergrax run` / lab `POST /v1/lab/run` | Fast local and harness validation | **Done** |
| **Agent Lab** (`lab_application`) | Compose and probe agents without product polish | **Done** |
| **Evaluation subsystem** (§42) | Offline / online / shadow / human modes | **Done** — EVAL phase |
| **Shadow workspace** | Compare candidate path without user impact | **Done** — REL Phase F |
| **Replay environment** | Deterministic re-run from trace store | **Partial** — `intergrax mvp replay` CLI (MVP-EVOL.3); no Tier-3 HTTP router |
| **Agent simulator** | Multi-agent contention and failure injection | **Partial** — `intergrax mvp simulate` CLI + `test_orchestration_cfg_simulation.py`; not wired to product hosts |
| **Trace Explorer** | Decision / tool / context visibility | **Partial** — lab debug APIs; UI deferred (GOV-PROD.1 §6.3) |
| **Promotion gates** | MVP → Beta evidence | **Done** — `scripts/check_mvp_promotion_gates.py` (MVP-EVOL.1) |
| **Product KPI / satisfaction** | Tenant metrics + CSAT bridge | **Done** — `product_kpi_registry.py`, `user_satisfaction.py` (MVP-EVOL.4–5); export surfaces CLI-only |

**IDEAL reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §22 (Developer Experience Layer).

## 44.3 Evaluation and testing strategy

| Mode | When | Mechanism |
|------|------|-----------|
| **Unit / contract** | Every PR | `pytest -m gate`, agent contract tests |
| **Golden offline eval** | Pre-merge / nightly | `EvaluationProfile` + `evaluation_assets` |
| **Shadow production** | MVP → Beta | `online_evaluation_registry` — candidate vs baseline |
| **Simulation** | Before real users | Harness CFG matrix, orchestration sim tests, future agent simulator |
| **Real-life pilot** | Beta | Sampled online eval + observability SLOs |
| **Human rubric** | Regulated / subjective quality | CVL + HITL scoring |

Raw text outputs are insufficient — structured results feed evaluators (§39.5, §42).

## 44.4 KPI, metrics, and user satisfaction

Platform and products SHOULD declare measurable outcomes. Intergrax provides **hooks and registries**; product teams define domain KPIs.

| Signal class | Examples | Harness hook |
|--------------|----------|----------------|
| **Technical KPIs** | Latency p95, success rate, cost per task, retry rate | Observability spine, `TASK_COMPLETED` payloads |
| **Quality KPIs** | Eval score trends, critic pass rate, schema validity | `evaluation_registry_trends`, CVL |
| **Product KPIs** | Task completion, time-to-value, feature adoption | Tier-3 app metrics export (product-owned) |
| **User satisfaction** | CSAT, NPS, thumbs up/down on responses | `feedback.*` integration pattern; online eval human mode |
| **Architecture health** | Debt index, gate coverage | §43 architecture metrics pipeline |

```text
Run → trace + eval score → trend registry → promotion gate / AHI proposal
User feedback → online eval registry → dashboard + optional HITL review
```

**Rule:** satisfaction and product KPIs are **not** inferred silently — explicit capture adapters or UI events with tenant scope.

## 44.5 Promotion gates (prototype → product)

| Gate | Checks |
|------|--------|
| **G0 — Runnable** | Scaffold smoke, `doctor` clean, one Nexus path green |
| **G1 — Eval baseline** | Offline golden set registered; score recorded |
| **G2 — Policy** | `ReliabilityProfile` + autonomy ceiling documented |
| **G3 — Multi-agent** | If N>1: `graph_spec` + merge + CFG proof |
| **G4 — Ops** | SLO catalog, runbook stub, checkpoint/resume if long-running |
| **G5 — Production PRR** | Phase V evidence, compatibility graph, owner sign-off |

Gates G0–G2 are **platform-enforced** via CI scripts; G3–G5 are product checklists in Tier-3 `ARCHITECTURE.md`.

## 44.6 Feedback into platform improvement

MVP iteration MUST feed the Harness — not only the product:

| Feedback source | Consumer |
|-----------------|----------|
| Eval regression | Block merge; CVL rubric updates |
| Trace anomalies | Observability alerts; optional AHI pattern detection |
| User dissatisfaction | Online eval + HITL queue; autonomy downgrade |
| Cost overrun | Cost profile tuning; model routing (AHI) |
| Failure patterns | Resilience policy proposals (REL-ADV) |

**Cross-ref:** L4 adaptive loop [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md) · agent lifecycle [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §20.

**Plan:** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) Phase MVP-EVOL.

---

# 43. Architecture Metrics, Debt, and CI Gates

Architecture health MUST be measured, not inferred.

## 43.1 Metric families

- modularity and coupling indicators,
- dependency graph health and incompatibility rate,
- observability and governance coverage on critical paths,
- policy / context / prompt / test coverage,
- architecture debt index with trend tracking.

**Code:** `runtime/architecture/architecture_metrics.py`, `architecture_metrics_pipeline.py`, `debt_governance.py`, `architecture_coverage.py`, `maturity_gate_evidence.py`.

## 43.2 Developer experience surface

| Surface | Role |
|---------|------|
| `intergrax/scaffold/` | `new-agent`, `new-application`, `new-skill` |
| `intergrax/cli/doctor.py` | Harness health checks |
| `scripts/test.bat` / `pytest -m gate` | Mandatory merge gates |
| `guides/AGENT_CREATION_GUIDE.md` | Author workflow |

**TTFRun** (idea → first Nexus run) is the primary DX metric. **Plan:** Phase DX, AA, W-OPS in [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).

## 43.3 Operational L3 evidence

Release cycles, SLO snapshots, and ops sign-off are tracked via `scripts/phase_w_ops_evidence.py` and release cycle artifacts under `build/architecture_hardening/`.

---
