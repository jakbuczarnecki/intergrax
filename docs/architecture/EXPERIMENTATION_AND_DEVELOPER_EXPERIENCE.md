# Experimentation Workflow and Developer Experience

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 25–27, 30  
**Audit instruction:** [`audit/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../audit/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Last updated:** 2026-06-20 — **P2-ARCH-13** Experimentation/DX architecture vs implementation rules boundary

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE canon).

- **Implement / audit default:** §39–§41 DX + minimal runtime flow. Extended §42+: [`arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md`](arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md). §43+: [`arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md`](arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md`](arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_extended_depth.md) | extended depth |
| [`arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md`](arch/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Experimentation / DX architecture owns

This layer **may** describe:

- experiment definitions,
- evaluation scenarios,
- developer feedback loops,
- local/lab execution ergonomics,
- smoke/e2e evidence collection,
- harness playgrounds,
- trace review workflows,
- test data and scenario catalogs,
- comparison of runs,
- documentation of evidence,
- developer-facing observability views,
- repeatable validation loops.

---

## Experimentation / DX architecture does not own

This architecture **MUST NOT** own:

- Tier-0/Tier-1/Tier-2/Tier-3 responsibility boundaries,
- agent runtime lifecycle,
- Nexus orchestration semantics,
- production policy decisions,
- HITL authority,
- tool side-effect gateway,
- integration access paths,
- context assembly rules,
- memory/RAG ownership,
- CodeCraft safety rules,
- AHI auto-apply decisions,
- ECP production scaling decisions.

It **may reference** those documents, but **must not redefine** them.

---

## Cursor / implementation rules placement

Cursor-specific implementation rules **SHOULD** live in:

- [`AGENTS.md`](../../AGENTS.md) — repo-wide coding agent behavior,
- [`LAYER_COMPLETION_MODE.md`](../guides/LAYER_COMPLETION_MODE.md) — layer completion workflow,
- [`AGENT_AUTHOR_MINIMAL_PATH.md`](../guides/AGENT_AUTHOR_MINIMAL_PATH.md) — agent authoring,
- [`TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md`](../guides/TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md) — Tier-3 product hypothesis,
- [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) — cross-layer invariants.

This architecture document **may link** to these guides, but **should not duplicate** their full content.

---

## Recommended document placement

| Content type | Canonical location |
|---|---|
| Cross-layer invariants | [`docs/guides/SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) |
| Maturity wording | [`docs/guides/MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md) |
| Cursor layer workflow | [`docs/guides/LAYER_COMPLETION_MODE.md`](../guides/LAYER_COMPLETION_MODE.md) |
| Repo-wide coding agent behavior | [`AGENTS.md`](../../AGENTS.md) |
| Agent authoring shortcut | [`docs/guides/AGENT_AUTHOR_MINIMAL_PATH.md`](../guides/AGENT_AUTHOR_MINIMAL_PATH.md) |
| Tier-3 product hypothesis | [`docs/guides/TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md`](../guides/TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md) |
| Experiment definitions/evidence loops | [`docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Subsystem architecture | [`docs/architecture/*.md`](.) |
| Implementation plan | [`docs/plan/*.md`](../plan/) |

---

## Cursor review checklist

Before modifying Experimentation / DX documentation, Cursor **must** verify:

- Is this architecture or implementation workflow guidance?
- If it is repo-wide coding behavior, should it be in [`AGENTS.md`](../../AGENTS.md)?
- If it is layer-completion process, should it be in [`LAYER_COMPLETION_MODE.md`](../guides/LAYER_COMPLETION_MODE.md)?
- If it is a subsystem rule, should it stay in the subsystem architecture document?
- Does this document redefine rules already owned by [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md)?
- Does this document accidentally override Nexus, ToolRuntime, Context, Memory, RAG, CVL, CodeCraft, AHI, or ECP boundaries?
- Are maturity claims expressed through [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md)?
- Are implementation examples clearly marked as examples, not architecture mandates?

---

## Migration note (§39–§41 legacy placement)

Sections **§39–§41** below predate this boundary split. They contain **Cursor implementation rules**, **minimal first implementation**, and **minimal runtime flow** — operational guidance, not Experimentation/DX subsystem architecture.

**TODO (future doc pass):** migrate §39–§41 to [`AGENT_INSTRUCTIONS.md`](../guides/AGENT_INSTRUCTIONS.md) / [`AGENTS.md`](../../AGENTS.md) without losing cross-refs from [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) and other domain pairs. Until then, treat §39–§41 as **legacy canonical copies**; do not add new Cursor workflow rules here.

---

# 39. Implementation Rules For Cursor AI

> **Legacy placement** — see [Architecture vs Implementation Rules Boundary](#architecture-vs-implementation-rules-boundary) and [Migration note (§39–§41 legacy placement)](#migration-note-3941-legacy-placement). Prefer [`AGENTS.md`](../../AGENTS.md) and [`AGENT_INSTRUCTIONS.md`](../guides/AGENT_INSTRUCTIONS.md) for repo-wide coding agent behavior.

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

Do not build a polished SaaS frontend at this stage. **DX-MAINT-04:** this remains an explicit harness non-goal — product UI belongs to Tier-3 hosts or Phase K, not the DX control plane.

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

When wiring §42 (events, hooks, UAEP), **integrate with** existing `RunTraceWriter`, `ToolRuntime`, `AgentEngine` — do not fork them.

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
