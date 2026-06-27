# INTERGRAX HARNESS AUDIT MAP

**Status:** audit control document
**Purpose:** structured audit map for comparing Intergrax against the target Harness AI architecture
**Primary reference:** `IDEAL_HARNESS_AI_ARCHITECTURE.md`
**Current architecture hub:** `intergrax_runtime_architecture.md`
**Current architecture detail:** `architecture/` (domain documents)
**Implementation hub:** `intergrax_runtime_architecture.md`
**Implementation detail:** `plan/`

---

# 1. Purpose

This document defines a structured audit map for Intergrax.

The goal is to prevent broad, shallow architecture audits where the implementation agent claims that the entire architecture is complete, only to rediscover new gaps in the next iteration.

Instead of auditing the whole platform at once, Intergrax should be audited layer by layer.

Each layer must have:

* a clear scope,
* target-state criteria,
* current-state assessment,
* gap list,
* risk level,
* required architecture updates,
* required implementation-plan updates,
* required code changes,
* required tests,
* Definition of Done,
* evidence required to mark the layer as complete.

The audit must always compare:

1. `IDEAL_HARNESS_AI_ARCHITECTURE.md`
2. `intergrax_runtime_architecture.md` + relevant `architecture/<domain>.md`
3. `intergrax_runtime_architecture.md` + relevant `plan/<domain>.md`
4. source code
5. tests
6. documentation

---

# 2. Core Problem

Intergrax is a large Harness AI / Agent OS platform.

A general audit of the whole project is too broad for a coding agent.

When the agent reviews everything at once, it tends to produce shallow results:

```text
reviewed everything
found all problems
updated the plan
implemented fixes
declared complete
```

In practice, the next iteration often discovers additional issues because the previous audit did not isolate individual architectural layers.

This document solves that problem by dividing the platform into auditable Harness AI layers.

---

# 3. Audit Philosophy

**Per-domain copy-paste prompts:** For deep single-domain audits (RAG, Tools, Memory, UAEP, …), use [`../audit/README.md`](../audit/README.md) — 22 prompts aligned 1:1 with `architecture/<DOMAIN>.md` ↔ `plan/<DOMAIN>.md`. **Multi-domain orchestration** (all 22 pairs in one Cursor session): [`../bootstrap/`](../bootstrap/README.md). **Idea intake before build (Mode I):** [`../bootstrap/idea_audit.txt`](../bootstrap/idea_audit.txt) · [`../audit/IDEA_AUDIT_ORCHESTRATOR.md`](../audit/IDEA_AUDIT_ORCHESTRATOR.md) — live verdict in chat; on operator approval update the affected domain-layer pair or multi-layer feature pair under `docs/features/` (no `audit_results/` folder). Use this document for layer map, scoring, and output format; use `../audit/<DOMAIN>.md` for the runnable agent instruction.

Do not audit the entire system at once.

Audit one layer at a time.

For each layer, the agent must:

1. Read the ideal target architecture.
2. Read the current Intergrax architecture.
3. Read the implementation plan.
4. Inspect the relevant source code.
5. Inspect the relevant tests.
6. Identify gaps.
7. Propose architecture updates if needed.
8. Propose implementation-plan updates if needed.
9. Implement only scoped fixes.
10. Add or update tests.
11. Run verification.
12. Update documentation.
13. Provide evidence.

The agent must never declare the entire platform complete after auditing one layer.

---

# 4. Standard Audit Output Format

For every audited layer, the implementation agent must produce the following output:

```md
# Audit Result: <Layer Name>

## 1. Scope

What was audited.

## 2. Target State

What the ideal architecture requires.

## 3. Current State

What Intergrax currently implements.

## 4. Gap List

Concrete gaps between target and current state.

## 5. Risk Assessment

Impact of the gaps.

Risk levels:

- Critical
- High
- Medium
- Low

## 6. Required Architecture Updates

Changes required in `intergrax_runtime_architecture.md` (hub) and/or the relevant `architecture/<domain>.md`.

## 7. Required Implementation Plan Updates

Changes required in `intergrax_runtime_architecture.md`.

## 8. Required Code Changes

Concrete implementation work.

## 9. Required Tests

Unit, integration, contract, acceptance, evaluation, or architecture-boundary tests.

## 10. Definition of Done

Precise criteria for marking this layer complete.

## 11. Evidence

Files changed, tests added, gates run, documentation updated.

## 12. Remaining Risks

Known limitations after the iteration.

## 13. Next Recommended Layer

What should be audited next.
```

---

# 5. Maturity Scoring Model

Each layer must be scored independently.

```text
L0 — Fragmented
Local functionality only, no consistent model, no governance, no telemetry.

L1 — Operational MVP
Basic mechanism exists but is incomplete, partially manual, and weakly tested.

L2 — Scalable Harness
Mechanism is modular, reusable, registered, testable, and works across multiple agents.

L3 — Production Harness OS
Full policy enforcement, telemetry, fallback, test coverage, documentation, SLOs, runbooks, and operational readiness.

L4 — Adaptive Agent OS
Closed feedback loops, automated optimization, adaptive routing, evaluation-driven improvement, and bounded self-tuning.
```

For every layer:

```md
Score before: L?
Score after: L?
Target score for current milestone: L?
Evidence supporting score: ...
```

A layer must not be marked as complete without evidence.

---

# 6. Global Audit Rules

## 6.1 No Global Done

The implementation agent must not write:

```text
The entire architecture is complete.
All issues have been resolved.
The platform is now fully aligned.
```

Unless a full layer-by-layer audit has been completed and verified.

Allowed wording:

```text
Layer audited: <name>
Layer score before: L?
Layer score after: L?
Evidence: ...
Remaining risks: ...
Next recommended layer: ...
```

## 6.2 Evidence Required

Every "Done" status requires evidence.

Evidence can include:

* files changed,
* tests added,
* tests run,
* gates passed,
* documentation updated,
* ADR updated,
* implementation-plan entry updated,
* architecture section updated,
* benchmark result,
* evaluation result.

## 6.3 No Out-of-Scope Fixes

When auditing one layer, do not rewrite unrelated layers.

If an issue is discovered outside the current scope, record it as:

```md
Out-of-scope finding:
- Area:
- Risk:
- Suggested next audit layer:
```

## 6.4 Target vs Current vs Plan vs Implementation

The agent must always distinguish:

* **Target architecture:** `IDEAL_HARNESS_AI_ARCHITECTURE.md`
* **Current architecture:** `intergrax_runtime_architecture.md` + `architecture/`
* **Implementation plan:** `intergrax_runtime_architecture.md` + `plan/`
* **Actual implementation:** source code
* **Verification:** tests and gates

Do not confuse these layers.

---

# 7. Recommended Audit Phases

## Phase 1 — Core Harness Integrity

Audit the minimum set of layers that determine whether Intergrax is truly a Harness AI system.

1. Strategic Harness Model
2. Tier Model and Dependency Boundaries
3. Execution Runtime and Agent OS
4. Policy and Governance
5. Tool / Skill / Integration Separation
6. Context Engineering
7. Observability and Telemetry
8. Error Handling and Reliability
9. Evaluation and Benchmarking
10. Architecture Governance and Documentation Loop

## Phase 2 — Capability Platform

Audit the reusable capability infrastructure.

1. LLM and Model Adapter Layer
2. Prompt Engineering and Prompt Registry
3. RAG and Retrieval Layer
4. Memory Layer
5. Registry Architecture
6. Capability Graph Architecture
7. Modality, Vision, Audio and Dedicated ML
8. Developer Experience, Scaffold and Lab

## Phase 3 — Production Readiness

Audit production-level maturity.

1. Identity, Trust and Tenancy
2. Security and Data Governance
3. Cost and Resource Governance
4. Product Environment and Tier-3 Applications
5. Testing, CI and Architecture Gates
6. Agent Lifecycle Governance
7. Operational Excellence and SLOs
8. Incident Management and Runbooks

---

# 8. Full Audit Layer Map

Each layer below maps to a **domain pair** (`docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md`). Routing index: [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) (Audit map routing table).

## 1. Strategic Harness Model

### Purpose

Verify that Intergrax is still being developed as a Harness AI / Agent OS platform, not as a collection of isolated agents.

### Audit Questions

* Is the Harness treated as the durable product?
* Are agents treated as replaceable execution units?
* Is the relationship clear?

```text
Harness -> Runtime -> Agents -> Applications -> Products
```

* Does the current architecture support long-term Harness AI evolution?
* Does the implementation plan still serve the strategic objective?
* Is Intergrax avoiding the "single super-agent" trap?

### Typical Gaps

* Architecture optimized for one agent instead of the runtime.
* Product logic mixed into runtime.
* Implementation plan focused on features rather than Harness maturity.
* Missing strategic alignment between ideal architecture and current architecture.

### Score

```text
Strategic Alignment Score: L0-L4
```

---

## 2. Tier Model and Dependency Boundaries

### Purpose

Verify that the four-tier Intergrax model is respected.

```text
Tier-0 Platform
Tier-1 Nexus Runtime
Tier-2 Agents
Tier-3 Applications
```

### Audit Questions

* Does Tier-0 contain only universal platform mechanisms?
* Does Tier-1 remain domain-agnostic?
* Do Tier-2 agents avoid vendor SDKs and direct integrations?
* Do Tier-3 applications only compose runtime, agents, tools, policies and integrations?
* Are dependency directions respected?
* Are there circular dependencies between tiers?
* Are new components placed in the correct tier?

### Typical Gaps

* Business logic inside Nexus.
* Vendor SDK imports in agents.
* Agent-specific branches in `NexusLoop`.
* Tier-3 application containing agent pipeline logic.
* Tier-0 duplicating mechanisms already existing elsewhere.

### Score

```text
Layer Boundary Integrity Score: L0-L4
```

---

## 3. Interface and Task Intake

### Purpose

Verify that all inputs enter the system through a normalized, auditable intake model.

### Audit Questions

* Is there a common input envelope such as `TaskEnvelope`?
* Do API, CLI, worker, queue and webhook inputs converge into the same runtime path?
* Are inputs contract-validated?
* Are tenant, user, SLA, constraints and risk metadata captured?
* Are sync, async and streaming variants consistent?
* Are task and run identifiers stable and traceable?

### Typical Gaps

* Multiple independent intake paths.
* API bypassing runtime lifecycle.
* Worker path not equivalent to HTTP path.
* Missing tenant or user metadata.
* Inconsistent validation between entrypoints.

### Score

```text
Interface Normalization Score: L0-L4
```

---

## 4. Identity, Trust and Tenancy

### Purpose

Verify that every execution is tied to identity, scope and data boundaries.

### Audit Questions

* Does the system distinguish user identity, service identity and agent identity?
* Does every run have tenant context?
* Are scopes and permissions propagated to tools and subagents?
* Is delegation auditable?
* Are secrets retrieved from a secrets layer?
* Are critical actions cryptographically signed or audit-protected where appropriate?
* Is tenant isolation enforced and tested?

### Typical Gaps

* Missing tenant ID in events.
* Tools invoked without permission context.
* Subagents inheriting unrestricted parent permissions.
* Secrets stored in application config or agent code.
* No audit trail for delegated execution.

### Score

```text
Identity and Trust Score: L0-L4
```

---

## 5. Policy and Governance

### Purpose

Verify that Intergrax is policy-first.

Nothing should execute without appropriate policy checks, permission checks and constraint checks.

### Audit Questions

* Is there a central `PolicyEngine`?
* Are policies applied pre-run, pre-plan, pre-LLM, pre-tool, post-tool and pre-output?
* Does `RuntimePolicyBundle` cover all relevant runtime paths?
* Does `ToolRuntime` enforce tool access policies?
* Are policy decisions traceable?
* Are execution modes supported?

  * strict
  * balanced
  * exploratory
* Are policy failures typed, explainable and recoverable?
* Are policies versioned and testable?

### Typical Gaps

* Policy exists in documentation but is not enforced.
* Policy checks only exist for tools.
* LLM calls bypass policy.
* Context assembly bypasses policy.
* HITL is not connected to policy decisions.
* Policy decisions are not traced.

### Score

```text
Policy Enforcement Score: L0-L4
```

**Guardrails vector (M.12):** Vendor LLM scanners are **not** a separate tier — they extend Policy & Governance via `IntegrationProfile.llm_guardrail` + `GuardrailProfile` + `LlmGuardrailMiddleware`. Canon: UAEP [§42.11.6](architecture/UNIFIED_EXECUTION_RUNTIME.md) · Integration [§47](architecture/INTEGRATIONS.md) · [ADR-GR-001](adr/entries/2026-06-09/ADR-GR-001.md).

**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) (control plane map, security profile, policy bundle read order); canon [§42.11](architecture/UNIFIED_EXECUTION_RUNTIME.md#4211-policy-engine); [`guides/EXTENSION_AUTHOR_GUIDE.md` §10](guides/EXTENSION_AUTHOR_GUIDE.md#10-policy-rule-handler-plugins-phase-dx-58) (`intergrax.policy_rules`). **Closeout:** [Phase GOV-AUDIT](plan/UNIFIED_EXECUTION_RUNTIME.md) **Done** (GOV-DOC.*) · [Phase M.12](plan/INTEGRATIONS.md) guardrails **Done**.

---

## 6. LLM and Model Adapter Layer

### Purpose

Verify that LLMs are treated as replaceable providers, not hardcoded system dependencies.

### Audit Questions

* Do all LLM calls go through `LLMAdapter`?
* Are there direct OpenAI, Anthropic, Gemini or other SDK imports in agents or runtime?
* Is there a formal `LLMProfile`?
* Can model selection consider cost, latency, quality, risk and capability?
* Are structured outputs validated?
* Is usage metered by tokens, cost, model and tenant?
* Are retries and fallbacks defined?
* Are models selected by policy/profile instead of hardcoded per agent?

### Typical Gaps

* Model name hardcoded in agent.
* Direct vendor SDK use.
* No fallback model.
* No token/cost telemetry.
* Same model used for planning, execution and evaluation without explicit reason.
* Structured output parsed manually without schema validation.

### Score

```text
LLM Abstraction Score: L0-L4
```

---

## 7. Reasoning, Planning and Cognition

### Purpose

Verify that reasoning is explicit, observable and separated from execution.

### Audit Questions

* Is planning represented as a structured contract?
* Are decisions recorded as `DecisionRecord` or equivalent?
* Is reasoning separated from side-effectful execution?
* Are planning strategies explicit?

  * no planner
  * deterministic planner
  * LLM planner
  * graph planner
* Are outputs validated against typed contracts?
* Is prompt compilation layered?

  * system
  * task
  * policy
  * context
  * memory
* Are reasoning failures classified separately from dependency/runtime failures?

### Typical Gaps

* Plan exists only as free text.
* Agent performs reasoning and tool execution in the same method.
* No decision record.
* No validation of model decisions.
* Reasoning path is not traceable.
* Prompt assembled ad hoc.

### Score

```text
Reasoning Architecture Score: L0-L4
```

**Domain pair (canon):** [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) ↔ [`plan/REASONING_AND_COGNITION.md`](plan/REASONING_AND_COGNITION.md)  
**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix I §I.4](guides/AGENT_CREATION_GUIDE.md#i4-planning-strategies-explicit-customizable) (planning strategies); canon [§42.5](architecture/UNIFIED_EXECUTION_RUNTIME.md#425-unified-agent-execution-protocol) (UAEP separation); **flow narrative:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §4–§18. **Historical closeout:** ORCH-1, FLOW-1/11/12 — **Done**; maturity uplift: [Phase COG-DEPTH](plan/REASONING_AND_COGNITION.md).

---

## 8. Execution Runtime and Agent OS

### Purpose

Verify that Intergrax has one coherent runtime path and a domain-agnostic Agent OS.

### Audit Questions

* Is there a single canonical execution path?
* Does execution flow through `UnifiedTaskRunner`, `NexusLoop`, `AgentEngine` and UAEP consistently?
* Is `NexusLoop` domain-agnostic?
* Is run and step lifecycle explicit?
* Are pause, resume, interrupt and checkpoint supported?
* Is retry handled by runtime rather than agents?
* Are checkpoints complete and recoverable?
* Does execution produce trace and runtime events?
* Can agents run through Nexus without an HTTP server?

### Typical Gaps

* Parallel execution engines.
* Legacy execution path still active.
* Agent-specific conditionals in `NexusLoop`.
* Retry logic implemented inside agents.
* Incomplete checkpointing.
* Missing lifecycle events.

### Score

```text
Execution Runtime Score: L0-L4
```

**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix I §I.2](guides/AGENT_CREATION_GUIDE.md#i2-orchestration-control-plane-map) (NexusLoop → AgentEngine stack); canon [§42.44](architecture/UNIFIED_EXECUTION_RUNTIME.md#4244-agentengine-as-universal-executor-summary); **flow narrative:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §4–§6.

---

## 9. Orchestration, Scheduler and Execution Graph

### Purpose

Verify that planning, scheduling, graph execution and routing are formal runtime responsibilities.

### Audit Questions

* Is there a formal `ExecutionGraph`?
* Do graph nodes and edges have typed contracts?
* Is graph execution observable?
* Is scheduling priority-aware?
* Are retry budgets tied to graph/step execution?
* Are fan-out/fan-in patterns supported?
* Are concurrency limits enforced?
* Is backpressure supported?
* Are execution strategies explicit?

  * single-agent
  * orchestrator-worker
  * supervisor-worker
  * evaluator loop
* Are merge policies deterministic?

### Typical Gaps

* Graph is implicit in code.
* Subtasks executed as normal function calls.
* No concurrency limits.
* No backpressure.
* No deterministic merge policy.
* Scheduler and graph concerns mixed with agent code.

### Score

```text
Orchestration Score: L0-L4
```

**Authoring reference:** [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §50–§52 (strategy catalog, parallelism, resilience); [`guides/AGENT_CREATION_GUIDE.md` Appendix I §I.5](guides/AGENT_CREATION_GUIDE.md#i5-graph-execution-and-merge) (batches, retry, merge); `ExecutionGraph` / `GraphExecutor`; **flow narrative:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §9, §14. **Closeout:** [Phase ORCH](plan/ORCHESTRATION.md) ORCH-2, ORCH-3 · [Phase ORCH-STRAT](plan/ORCHESTRATION.md) — **Done**.

---

## 10. Subagents and Multi-Agent Coordination

### Purpose

Verify that subagents are isolated delegated executions, not nested uncontrolled agents.

### Audit Questions

* Is there a formal `SubtaskContract`?
* Does each subagent have a constrained scope?
* Does each subagent have its own context namespace?
* Does each subagent have its own memory namespace?
* Does the parent retain policy control?
* Are budgets split or delegated explicitly?
* Is delegation traced?
* Are outputs merged through a defined merge policy?
* Are subagent tools restricted by scope and policy?

### Typical Gaps

* Subagent is just a function call.
* No memory isolation.
* No budget delegation.
* No input/output contract.
* Parent cannot explain why delegation happened.
* Subagent inherits all parent tools.

### Score

```text
Subagent Architecture Score: L0-L4
```

**Authoring reference:** [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §50, §53 (patterns, delegation, collaboration); [`guides/AGENT_CREATION_GUIDE.md` Appendix I §I.6](guides/AGENT_CREATION_GUIDE.md#i6-subagent--delegation-semantics-r-delegate--done); canon [§42.14.3](architecture/UNIFIED_EXECUTION_RUNTIME.md#42143-graph-delegation-subagent-equivalent); **flow narrative:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §13.

---

## 11. Tool Layer

**Authoring map (control plane):** `guides/AGENT_CREATION_GUIDE.md` **Appendix J** · implementation closeout: plan **Phase TOOL-ENG** (**Done** 2026-06-12).

### Purpose

Verify that tools are atomic, policy-governed, observable operations.

### Audit Questions

* Does every tool define:

  * `tool_id`
  * input schema
  * output schema
  * risk level
  * timeout
  * retry policy
  * telemetry?
* Do all tool calls go through `ToolRuntime`?
* Are tools atomic rather than workflow-sized?
* Are tool handlers backed by Integration Contracts rather than direct vendor SDKs?
* Are tools exportable to MCP or function schemas?
* Are tool calls policy-checked?
* Are tool calls traced?
* Do tools have contract tests?

### Typical Gaps

* Oversized tools performing full workflows.
* Agent-local tool registries.
* Tool without risk metadata.
* Tool without trace.
* Tool bypassing integration catalog.
* Boolean flags such as `use_rag` instead of `tool_ids`.

### Score

```text
Tool Layer Score: L0-L4
```

**Authoring reference:** [`architecture/TOOLS.md`](architecture/TOOLS.md) — [Tool execution pipeline](architecture/TOOLS.md#tool-execution-pipeline) (select → orchestrate → invoke → log) + [Invocation patterns](architecture/TOOLS.md#tool-invocation-patterns-production-orchestration) + [Tool engine component map](architecture/TOOLS.md#tool-engine-implemented-today); runtime narrative [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §15–§17; enforcement [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.12; author control plane [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) Appendix J. **Closed:** [Phase TOOL-ENG](plan/TOOLS.md) (2026-06-12) — 36/36 deliverables including TOOL-ENG-16–30.

---

## 11b. Ephemeral Code Craft Layer

**Authoring map:** [`architecture/CODE_CRAFT.md`](architecture/CODE_CRAFT.md) · [`plan/CODE_CRAFT.md`](plan/CODE_CRAFT.md) · audit prompt [`../audit/CODE_CRAFT.md`](../audit/CODE_CRAFT.md)

### Purpose

Verify that dynamic codegen runs through harness orchestration (`CodeCraftOrchestrator`), not agent-local subprocess loops; that ephemeral tools stay task-scoped; and that sandbox substrate is reused.

### Audit Questions

* Is `CodeCraftProfile` wired on the host (`wire_application_codecraft`)?
* Do `codecraft.*` tools route through `ToolRuntime` / `SANDBOX_REQUIRED_TOOLS`?
* Does `StaticCodeGate` run before exec in autonomous modes?
* Are craft modes enforced (`disabled`, `dry_run`, `assist_only`, `supervised`, `autonomous`)?
* Is promotion typed (`CraftResult`) rather than stdout-only?
* Are `CODECRAFT_*` trace steps emitted and correlated with `craft_id` / `sandbox_session_id`?
* Does supervised mode use HITL before exec?
* Are ephemeral tools registered only in `EphemeralToolRegistry`, not global `ToolRegistry`?

### Typical Gaps

* Craft loop implemented inside Tier-2 agent code.
* Missing sandbox session → host subprocess fallback (must fail closed).
* Local sandbox treated as production security boundary without `isolation_tier=cloud`.
* Global catalog pollution from generated helpers.

### Score

```text
Ephemeral Code Craft Score: L0-L4
```

**Authoring reference:** [`architecture/CODE_CRAFT.md`](architecture/CODE_CRAFT.md) · substrate [`RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) · verification [`CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md). **Closed:** [Phase ECC](plan/CODE_CRAFT.md) (2026-06-13) — ECC-0…ECC-6 + S7–S10 Done (L3+); depth = metrics dashboards (§10.2) + container isolation tier + codegen LLM profile wiring.

---

## 12. Skill Layer

**Authoring map (control plane):** `guides/AGENT_CREATION_GUIDE.md` **Appendix J** · implementation closeout: plan **Phase TS** (**Done**).

### Purpose

Verify that skills are composable capability packs, not tools and not agents.

### Audit Questions

* Is skill separate from tool?
* Does `SkillManifest` include:

  * `skill_id`
  * version
  * description
  * `tool_ids`
  * prompt instruction references
  * policy fragment
  * risk tier?
* Are skills registered in Skill Registry?
* Do agents declare `skill_ids` instead of copying tool/prompt lists?
* Are skill dependencies validated?
* Are skills tested?
* Are external skills imported through validation?
* Do skills avoid vendor SDKs and runtime logic?

### Typical Gaps

* Markdown instruction pack registered as a tool.
* Skills without versions.
* Skills without policy fragments.
* Prompt duplication between agents.
* Skills bypassing tool runtime.
* Skill-specific runtime logic.

### Score

```text
Skill Layer Score: L0-L4
```

---

## 13. Integration Layer

**Authoring map (control plane):** `guides/AGENT_CREATION_GUIDE.md` **Appendix K** · implementation closeout: plan **Phase INT** (**Done**).

### Purpose

Verify that integrations are backend/provider adapters, not agent tools or business logic.

### Audit Questions

* Are integrations registered in `intergrax/integrations`?
* Does every integration belong to a category contract?
* Does each provider implement a stable contract?
* Is vendor SDK usage isolated to boundary modules?
* Does Tier-3 resolve integrations through `IntegrationProfile`?
* Are agents vendor-agnostic?
* Do integrations have health checks?
* Do integrations have smoke tests?
* Do integrations define retry/rate-limit behavior?
* Does adding a new integration category require architecture review?

### Typical Gaps

* Direct imports of `redis`, `boto3`, `psycopg`, etc. in agents.
* Provider without contract.
* Integration config stored in agent directory.
* No health checks.
* LLM provider treated as Integration Library provider instead of LLM Adapter.

### Score

```text
Integration Layer Score: L0-L4
```

---

## 14. RAG and Retrieval Layer

**Canonical architecture:** [`architecture/RAG.md`](architecture/RAG.md) · implementation: [`plan/RAG.md`](plan/RAG.md) (**Phase M-RAG** **Done** · **M-RAG-DEPTH** planned)  
**Authoring map (control plane):** `guides/AGENT_CREATION_GUIDE.md` **Appendix K** §K.5 · runtime bridge closeout: plan **Phase RAG** (**Done**)

### Purpose

Verify that retrieval is a full architecture layer, not a vector-search shortcut.

### Audit Questions

* Is there one canonical `RetrievalService`?
* Is RAG invoked through catalog tools such as `rag.retrieve`?
* Are there no direct `vectorstore.query` shortcuts in agents?
* Is there a formal `RagProfile`?
* Are retrieval modes supported?

  * vector
  * keyword
  * hybrid
  * graph
  * reranking
  * agentic retrieval
* Is retrieval telemetry emitted?
* Are citations preserved?
* Are retrieval results tenant-aware?
* Is chunking/ingest configurable?
* Are there golden retrieval tests?
* Is retrieval poisoning defense considered?

### Typical Gaps

* Multiple RAG paths.
* Dense-only retrieval.
* RAG logic inside agent.
* Missing citations.
* Missing recall/MRR benchmarks.
* Retrieval without tenant boundary.
* No reranking or graph strategy.

### Score

```text
RAG Architecture Score: L0-L4
```

---

## 15. Memory Layer

**Canonical architecture:** [`architecture/MEMORY.md`](architecture/MEMORY.md) · implementation: [Phase MEM](plan/MEMORY.md) (**Done**) · [Phase MEM-DEPTH](plan/MEMORY.md) (**Done**) · [Phase MEM-VEC](plan/MEMORY.md) (**P0–P1 Done**, MEM-VEC-3 backlog)

### Purpose

Verify that memory is explicit, scoped, governed and observable.

### Audit Questions

* Are memory types clearly separated?

  * run-local memory
  * task memory
  * session memory
  * user long-term memory
  * tenant memory
  * procedural memory
  * shared multi-agent context
* Does every memory read/write have scope?
* Is memory access policy-controlled?
* Does memory have lineage?
* Does memory have retention policy?
* Are memory artifacts traced?
* Is memory isolated between subagents?
* Is there a forget/delete mechanism?
* Does memory avoid direct DB access from agents?

### Typical Gaps

* One global memory store.
* Missing namespace.
* No retention model.
* Agent writes directly to database.
* No memory provenance.
* No separation between context and memory.

### Score

```text
Memory Architecture Score: L0-L4
```

---

## 16. Context Engineering Layer

**Domain pair:** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) · [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md) · **ADR-CTX-001**  
**Authoring map (control plane):** `guides/AGENT_CREATION_GUIDE.md` **Appendix L** · CTX control plane **Done** · plugin engine **CE-EXT Planned**

### Purpose

Verify that context is built, scored, budgeted, compressed and traced as a first-class runtime concern.

### Audit Questions

* Is there a `ContextManager`?
* Is there a `ContextBudgetPolicy`?
* Is context assembly a deterministic pipeline?
* Can context sources include:

  * task
  * memory
  * RAG
  * tools
  * policies
  * runtime state?
* Are context fragments scored by relevance, freshness and confidence?
* Is trimming traceable?
* Is compression controlled?
* Does every context fragment have provenance?
* Can output be linked back to context sources?
* Are context regression tests present?

### Typical Gaps

* Prompt assembled manually.
* No token budget.
* No context source lineage.
* No record of why a fragment was included or excluded.
* RAG and memory mixed without governance.
* No context quality tests.

### Score

```text
Context Engineering Score: L0-L4
```

---

## 17. Prompt Engineering and Prompt Registry

**Authoring map (control plane):** `guides/AGENT_CREATION_GUIDE.md` **Appendix M** · implementation closeout: plan **Phase PE** (**Done**).

### Purpose

Verify that prompts are managed architectural assets, not hidden strings in code.

### Audit Questions

* Is there a Prompt Registry?
* Does every managed prompt have:

  * id
  * version
  * owner
  * risk class
  * changelog?
* Is prompt composition layered?
* Are policy overlays applied deterministically?
* Are prompts tested against golden cases?
* Are prompt changes reviewed?
* Can prompt versions be compared?
* Are prompts linked to skills, agents or policies?

### Typical Gaps

* Prompt as multiline string in code.
* No versioning.
* No prompt tests.
* Policy copied manually into prompts.
* Prompt duplication across agents.
* No rollback path.

### Score

```text
Prompt Governance Score: L0-L4
```

---

## 18. Agent Assembly and Agent Contracts

### Purpose

Verify that agents are composable units built from profiles and capabilities, not monolithic systems.

### Audit Questions

* Does `AgentContract` contain enough metadata?
* Does the agent declare capabilities?
* Does the agent declare `skill_ids`?
* Are allowed tools resolved from skills and policies?
* Does the agent have a bounded local loop?
* Does the agent avoid global orchestration?
* Can the agent run through Nexus without HTTP?
* Is the agent reusable across applications?
* Does the agent have lifecycle state?

  * draft
  * experimental
  * certified
  * deprecated
  * retired

### Typical Gaps

* Agent god object.
* Agent depends on one application.
* Agent owns its own tool registry.
* Agent owns its own LLM stack.
* Agent lacks contract tests.
* Agent cannot be registered without runtime changes.

### Score

```text
Agent Assembly Score: L0-L4
```

---

## 19. Registry Architecture

### Purpose

Verify that registries are core runtime primitives and not optional documentation.

### Audit Questions

* Does every artifact type have a registry?

  * Agent
  * Tool
  * Skill
  * Policy
  * Prompt
  * Integration
  * Evaluation
* Does each registry support:

  * discovery
  * versioning
  * lifecycle state
  * dependency tracking
  * compatibility validation?
* Does runtime resolution go through registries?
* Is capability negotiation supported?
* Are plugin packages supported?
* Can registries produce impact analysis?

### Typical Gaps

* Hardcoded imports instead of registry resolution.
* Enum slugs as central bottleneck.
* Registry without lifecycle state.
* No version compatibility.
* No dependency graph.

### Score

```text
Registry Architecture Score: L0-L4
```

---

## 20. Capability Graph Architecture

### Purpose

Verify that dependencies between capabilities are explicit, analyzable and enforceable.

### Target Model

```text
Integration -> Tool -> Skill -> Policy -> Agent -> Application -> Product
```

### Audit Questions

* Is there a formal capability graph?
* Are graph nodes typed?

  * integration
  * tool
  * skill
  * policy
  * prompt
  * agent
  * application
  * product
* Are edges typed?

  * uses
  * requires
  * enables
  * restricts
  * depends_on
* Can the system perform blast-radius analysis?
* Can a tool change reveal affected skills, agents and applications?
* Can a policy change reveal affected tools, agents and applications?
* Does compatibility validation operate on dependency edges?
* Is the graph used in audit and release decisions?

### Typical Gaps

* Dependencies only visible by reading code.
* No impact analysis.
* Tool/skill/policy changes made blindly.
* No compatibility graph.
* Release risk not measurable.

### Score

```text
Capability Graph Score: L0-L4
```

---

## 21. Observability and Telemetry

### Purpose

Verify that every important runtime decision and action is traceable, measurable and diagnosable.

### Audit Questions

* Does every execution have `trace_id` and `run_id`?
* Are lifecycle events emitted for:

  * run
  * step
  * tool invocation
  * policy decision
  * LLM call
  * context assembly
  * memory read/write
  * subagent delegation?
* Are metrics emitted for:

  * latency
  * tokens
  * cost
  * retries
  * error classes
  * quality?
* Are logs structured?
* Is telemetry correlated by tenant/run/step?
* Is there an event journal?
* Can historical runs be replayed or inspected?
* Is observability mandatory rather than best-effort?

### Typical Gaps

* Happy-path-only tracing.
* Missing events for policy/context/memory.
* Cost only in logs.
* Missing correlation IDs.
* No dashboard-ready metrics.
* No event journal.

### Score

```text
Observability Score: L4
```

**Authoring reference:** [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) (Harness Observability Spine — canonical deep dive); [`guides/AGENT_CREATION_GUIDE.md` Appendix H §H.5](guides/AGENT_CREATION_GUIDE.md#h5-observability--what-is-mandatory-vs-optional) (mandatory vs optional signals); [`guides/AGENT_CREATION_GUIDE.md` Appendix Q](guides/AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) (wire-time profile bridge + assembly validation); [`guides/HARNESS_ENVIRONMENT.md`](guides/HARNESS_ENVIRONMENT.md#otlp--observability-s-ops2) (OTLP / lab debug APIs); canon [§42.1](architecture/UNIFIED_EXECUTION_RUNTIME.md#421-runtimeevent-contract) (event catalog). **Closeout:** [Phase OBS](plan/OBSERVABILITY.md) **Done** (wiring); [Phase OBS-BUS](plan/OBSERVABILITY.md) **Done** (spine L4). **Gate evidence:** `scripts/maintenance/check_observability_gates.py` (emission coverage, payload registry, persistence conformance, trace bridge catalog, L4 depth gate).

---

## 22. Error Handling and Reliability

### Purpose

Verify that failures are anticipated, classified and handled safely.

### Audit Questions

* Is there a central error taxonomy?
* Are errors classified?

  * user error
  * policy error
  * dependency error
  * runtime error
  * quality error
* Is retry policy defined?
* Are retry budgets enforced?
* Are backoff and jitter used?
* Are circuit breakers supported?
* Are model/tool fallbacks available?
* Do side-effectful operations use idempotency keys?
* Are compensation flows defined?
* Can the system return partial results?
* Are high-risk failures escalated to HITL?

### Typical Gaps

* Catch-all exceptions.
* Retry without limits.
* No idempotency.
* No fallback.
* No distinction between quality and dependency failures.
* Errors not traced.

### Score

```text
Reliability Score: L0-L4
```

**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix R](guides/AGENT_CREATION_GUIDE.md#appendix-r--reliability-control-plane-closeout) (wire-time idempotency bridge + circuit breaker assembly); H-APP `ReliabilityProfile` in [`environment_profile.py`](../intergrax/applications/contracts/environment_profile.py). **Closeout:** [Phase REL](plan/OBSERVABILITY.md) **Done** (REL-DOC.*).

---

## 23. Security and Data Governance

### Purpose

Verify that data and execution safety are first-class platform properties.

### Audit Questions

* Is there data classification?

  * public
  * internal
  * confidential
  * restricted
* Is PII or secret detection implemented?
* Is prompt injection defense implemented?
* Is tool injection defense implemented?
* Is retrieval poisoning defense implemented?
* Is tenant isolation tested?
* Is retention policy defined?
* Is audit trail immutable?
* Are secrets rotated?
* Is output sanitization implemented?

### Typical Gaps

* All data treated equally.
* No prompt injection tests.
* No tenant isolation tests.
* No source trust scoring.
* No retention policy.
* Secrets passed through agent code.

### Score

```text
Security and Data Governance Score: L0-L4
```

**LLM guardrails (M.12):** Complements V-SEC native defenses — `GuardrailProfile` + `IntegrationProfile.llm_guardrail` → `LlmGuardrailMiddleware` (no vendor SDK in Tier-2). Canon: [`INTEGRATIONS.md` §47](architecture/INTEGRATIONS.md) · UAEP §42.11.6.

**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H §H.3](guides/AGENT_CREATION_GUIDE.md#h3-security-profile-per-application) (V-SEC toggles); [`guides/AGENT_CREATION_GUIDE.md` Appendix S](guides/AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout) (wire-time middleware bridge + assembly validation). **Closeout:** [Phase SEC](plan/UNIFIED_EXECUTION_RUNTIME.md) **Done** (SEC-DOC.*) · [Phase M.12](plan/INTEGRATIONS.md) guardrails **Done**.

---

## 24. Cost and Resource Governance

### Purpose

Verify that cost and resource consumption are explicit, enforceable and optimizable.

### Audit Questions

* Are cost budgets defined?
* Are token budgets defined?
* Are tool budgets defined?
* Are model budgets defined?
* Is cost calculated per:

  * run
  * tenant
  * agent
  * product
  * model
  * tool?
* Is anomaly detection supported?
* Does model routing consider cost?
* Does context budgeting control cost?
* Is cost forecasting supported?
* Are quotas enforced?

### Typical Gaps

* No cost attribution.
* No budget enforcement.
* Premium model usage uncontrolled.
* Token growth not detected.
* No per-tenant cost metrics.
* No anomaly detection.

### Score

```text
Cost Governance Score: L0-L4
```

**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix T](guides/AGENT_CREATION_GUIDE.md#appendix-t--cost-governance-control-plane-closeout) (wire-time budget bridge + policy bundle merge); V-COST contracts in `runtime/architecture/cost_*.py`. **Closeout:** [Phase COST](plan/UNIFIED_EXECUTION_RUNTIME.md) **Done** (COST-DOC.*).

---

## 25. Evaluation and Benchmarking

### Purpose

Verify that intelligent behavior is measured, compared and regression-tested.

### Audit Questions

* Is there an Evaluation Registry?
* Are golden datasets defined?
* Are scenario benchmarks defined?
* Are quality regression tests present?
* Is LLM-as-a-Judge supported where appropriate?
* Are deterministic validators used where possible?
* Are versions compared?

  * prompt version
  * model version
  * skill version
  * tool version
* Does every major change have baseline vs post-change evaluation?
* Can evaluation results block release?
* Are human review samples supported?

### Typical Gaps

* Tests only prove that code runs.
* No quality regression.
* No benchmark delta.
* No evaluation history.
* No context/RAG evaluation.
* No release gate based on quality.

### Score

```text
Evaluation Score: L0-L4
```

**Authoring reference:** [`guides/AGENT_CREATION_GUIDE.md` Appendix U](guides/AGENT_CREATION_GUIDE.md#appendix-u--evaluation-control-plane-closeout) (wire-time evaluation bridge + policy bundle merge); V-EVAL contracts in `runtime/architecture/online_evaluation*.py`, `eval/nexus_eval_runner.py`; **CVL depth:** [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · Phase CRIT-V. **Closeout:** [Phase EVAL](plan/CRITIC_VERIFICATION.md) **Done** (EVAL-DOC.*); execution depth → [Phase CRIT-V](plan/CRITIC_VERIFICATION.md).

---

## 26. Testing, CI and Architecture Gates

### Purpose

Verify that architecture rules are enforced automatically.

### Audit Questions

* Are there unit tests?
* Are there integration tests?
* Are there contract tests?
* Are there acceptance tests?
* Are there architecture-boundary tests?
* Are import gates enforced?
* Are no-network tests present?
* Are regression gates present?
* Are simulation or chaos tests present?
* Does CI block architecture violations?

### Typical Gaps

* Manual architecture review only.
* No import boundary tests.
* No contract tests for providers/tools/skills.
* No regression gate.
* Tests not mapped to architecture layers.
* CI allows runtime duplication.

### Score

```text
Testing and Gate Score: L0-L4
```

---

## 27. Developer Experience, Scaffold and Lab

### Purpose

Verify that Intergrax enables fast creation, testing and iteration of agents and capabilities.

### Audit Questions

* Does scaffold generate correct agents, skills and applications?
* Can a new agent run without runtime changes?
* Is there a local lab environment?
* Is there a debug API?
* Is there a Trace Explorer or equivalent?
* Is replay supported?
* Are sample agents available?
* Is onboarding documentation clear?
* Is agent creation workflow documented in one canonical place?
* Is "idea to first run" measurable?

### Typical Gaps

* Scaffold incomplete or stale.
* New agent requires runtime edits.
* No replay or debug workflow.
* Documentation duplicated across files.
* Agent creation guide does not match code.
* Local lab does not match production path.

### Score

```text
Developer Experience Score: L0-L4
```

---

## 28. Product Environment and Tier-3 Applications

### Purpose

Verify that applications are deployable environments, not agents.

### Audit Questions

* Are applications self-contained?
* Do applications have their own `.env.example`?
* Do applications define host, factory, settings and wiring?
* Do applications configure:

  * agents
  * skills
  * tools
  * integrations
  * policies?
* Do applications avoid agent business logic?
* Can the same agent be reused in multiple applications?
* Is deployment documented?
* Are product-specific policies explicit?
* Are application environments testable independently?

### Typical Gaps

* Application contains agent pipeline logic.
* Agent depends on one application.
* Environment variables only exist in root `.env`.
* No application manifest.
* Deployment not self-contained.
* Tier-3 wiring duplicated.
* Flat `ApplicationEnvironmentProfile` namespace growth — mitigated by hierarchical bundles (§22.6 · APP-EVOL-8 · ADR-APP-003).

### Score

```text
Tier-3 Product Environment Score: L0-L4
```

---

## 29. Modality, Vision, Audio and Dedicated ML

### Purpose

Verify that non-text modalities are first-class and separated from the core text LLM path.

### Audit Questions

* Are modality planes separated?

  * generative LLM
  * ingest/RAG
  * dedicated inference
* Are vision, audio and classical ML invoked through tools?
* Do heavy models run outside the core runtime OS?
* Are agents forbidden from importing `torch`, `onnxruntime`, `ultralytics`, etc. directly?
* Do modality tools have policy and risk metadata?
* Are media size limits enforced?
* Is inference telemetry emitted?
* Do modality outputs have schemas?
* Are modality profiles available?

### Typical Gaps

* Vision/audio handled ad hoc.
* Heavy inference embedded into agent code.
* No media limits.
* No telemetry for inference.
* No distinction between generative multimodal LLM and deterministic CV model.
* Vendor SDK imports in agents.

### Score

```text
Modality Architecture Score: L0-L4
```

---

## 30. Operational Excellence and SLOs

### Purpose

Verify that the system can be operated as a production platform.

### Audit Questions

* Are SLOs defined?
* Are SLIs measured?
* Are dashboards defined?
* Are alerts defined?
* Are runbooks available?
* Is there incident classification?
* Are postmortems required?
* Are rollback procedures documented?
* Are release readiness checks defined?
* Is on-call ownership clear for production components?

### Typical Gaps

* No SLOs.
* Metrics exist but are not tied to SLOs.
* No operational runbooks.
* No release readiness process.
* No incident budget.
* No ownership model.

### Score

```text
Operational Excellence Score: L0-L4
```

**Domain pair (canon):** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](architecture/ELASTIC_CAPACITY_AND_SCALING.md) ↔ [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](plan/ELASTIC_CAPACITY_AND_SCALING.md) · [ADR-SCALE-001](adr/entries/2026-06-08/ADR-SCALE-001.md) — closed-loop **elastic capacity** (Harness ECP) extends this layer; SLO/runbook baseline: Phase W-OPS **Done**.

---

## 31. Agent Lifecycle Governance

### Purpose

Verify that agents have a managed lifecycle from experiment to retirement.

### Audit Questions

* Are agent lifecycle states defined?

  * draft
  * experimental
  * candidate
  * certified
  * production
  * deprecated
  * retired
* Is there an agent certification process?
* Are promotion criteria defined?
* Are rollback criteria defined?
* Are ownership and responsibility defined?
* Are version compatibility rules defined?
* Are deprecation policies defined?
* Are retired agents blocked from production routing?
* Are evaluation results required before promotion?

### Typical Gaps

* Agent marked ready without certification.
* No retirement process.
* No owner.
* No versioning.
* Old agents still routeable.
* Agent promotion not tied to tests/evals.

### Score

```text
Agent Lifecycle Governance Score: L0-L4
```

---

## 32. Architecture Governance and Documentation Loop

### Purpose

Verify that architecture, implementation plan and documentation evolve together.

### Audit Questions

* Does every major change go through:

  * architecture review
  * implementation-plan review
  * implementation
  * verification
  * documentation update?
* Is `IDEAL_HARNESS_AI_ARCHITECTURE.md` treated as target architecture?
* Is `intergrax_runtime_architecture.md` treated as current architecture?
* Is `intergrax_runtime_architecture.md` treated as current execution plan?
* Are ADRs used for significant decisions?
* Are architecture risks tracked?
* Is architecture debt tracked?
* Does "Done" require evidence?
* Are out-of-scope findings recorded instead of silently ignored?

### Typical Gaps

* Documentation updated after implementation only.
* Architecture and plan diverge.
* Cursor declares complete without evidence.
* ADRs missing.
* Implementation plan says done while tests are missing.
* Target and current architecture are confused.

### Score

```text
Architecture Governance Score: L0-L4
```

---

# 9. Recommended Cursor Prompt Template

Use this prompt for focused layer audits.

```md
# TASK: Focused Intergrax Harness Audit

You are auditing exactly one Harness AI architecture layer.

Do not audit the whole system.

Do not implement unrelated changes.

Do not declare the entire platform complete.

## Layer to audit

<INSERT LAYER NAME>

## References

Compare:

1. `IDEAL_HARNESS_AI_ARCHITECTURE.md`
2. `intergrax_runtime_architecture.md` + relevant `architecture/<domain>.md`
3. `intergrax_runtime_architecture.md` + relevant `plan/<domain>.md`
4. source code
5. tests
6. documentation

## Required output

Produce:

1. Scope
2. Target state
3. Current state
4. Gap list
5. Risk assessment
6. Required architecture updates
7. Required implementation-plan updates
8. Required code changes
9. Required tests
10. Definition of Done
11. Evidence required
12. Remaining risks
13. Next recommended layer

## Scoring

Use the maturity model:

- L0 Fragmented
- L1 Operational MVP
- L2 Scalable Harness
- L3 Production Harness OS
- L4 Adaptive Agent OS

Provide:

- Score before
- Target score
- Score after, only if implementation is performed
- Evidence supporting the score

## Rules

- If you find issues outside this layer, record them as out-of-scope findings.
- Do not fix them unless explicitly asked.
- Do not create duplicate Tier-0 mechanisms.
- Reuse existing platform mechanisms.
- Verify all claims with code/tests/docs.
- Never mark the layer as complete without evidence.
```

---

# 10. Recommended Completion Statement

Every Cursor iteration should end with this exact type of statement:

```md
# Completion Summary

Layer audited: <Layer Name>

Score before: L?
Score after: L?
Target score for current milestone: L?

## Evidence

- Files changed:
- Tests added:
- Tests run:
- Gates passed:
- Documentation updated:
- ADR updated:
- Implementation plan updated:

## Remaining Risks

- ...

## Out-of-Scope Findings

- ...

## Next Recommended Layer

- ...
```

---

# 11. Final Rule

Intergrax should not be evaluated by the number of implemented features.

Intergrax should be evaluated by whether each Harness AI layer is:

* correctly designed,
* correctly implemented,
* policy-governed,
* observable,
* testable,
* documented,
* reusable,
* verifiable.

A layer is not complete because the implementation agent says it is complete.

A layer is complete only when the architecture, implementation plan, code, tests and documentation all provide evidence.
