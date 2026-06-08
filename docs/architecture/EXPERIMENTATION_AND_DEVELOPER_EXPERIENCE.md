# Experimentation Workflow and Developer Experience

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 35. Experimentation Workflow

The expected workflow for a new idea:

```text
1. Define hypothesis
2. Define agent capability
3. Define expected output
4. Define validation criteria
5. Implement minimal agent
6. Register agent
7. Run through Nexus
8. Observe execution trace
9. Compare outputs
10. Decide: keep, improve, pause or delete
```

Example hypothesis:

> ProblemRadarAgent can discover repeated user complaints from Reddit and Hacker News and cluster them into potential product ideas.

This should become an agent experiment, not a full product.

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
