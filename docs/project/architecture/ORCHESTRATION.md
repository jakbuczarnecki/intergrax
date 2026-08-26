# Orchestration

**Intergrax Orchestration** is the platform configuration and topology layer that defines how independently executable work is **structured and coordinated** — planner/classifier selection, accepted graph topology, parallelism caps, merge posture, resilience configuration, and delegation structure — before Nexus interprets that topology at runtime.

> **Orchestration answers: HOW IS WORK STRUCTURED?**
> **Nexus answers: WHAT EXECUTES NEXT?**
> **Unified Execution Runtime answers: HOW DOES AN EXECUTION BEHAVE?**

**Semantic authority:** Subordinate to frozen [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) (UEA). Where Orchestration docs and UEA conflict, **UEA wins**.

Orchestration is **not** a second runtime, a private agent thread mechanism, or a place for product-specific orchestration code. Tier-3 hosts declare an `OrchestrationProfile` and optional `ApplicationGraphSpec`; Nexus consumes **accepted** topology on orchestration-strategy Executions.

## Why it matters

Without a central orchestration model, each application could implement its own planner selection, topology, parallelism, delegation, merge, and resilience posture. That leads to custom pipelines, graph-node-as-Execution confusion, weak governance, and incomparable behavior.

Intergrax moves collaboration **structure** into platform contracts so every host configures the same dimensions — and Nexus executes **interpretation** of accepted topology, not ad-hoc agent graphs.

> [!NOTE]
> **Maturity boundary:** Phase ORCH, ORCH-STRAT, ORCH-CONFIG, ORCH-5, and ORCH-6 are **Done** on the **CURRENT** harness path. That is **not** proof that `OrchestrationDefinition` / child Execution admission is implemented, **not** universal production qualification, and **not** proof that multi-agent == orchestration or single-agent requires orchestration.

**Primary audience:** Principal / Staff engineers and Tier-3 host authors configuring collaboration structure.

## At a glance

| Concern | Summary |
| -------- | -------- |
| **Core question** | How is work **structured**? (Nexus = what next; UER = Execution behavior) |
| **Configuration** | `OrchestrationProfile`; optional `ApplicationGraphSpec` — **CURRENT** wiring surfaces |
| **Target concept** | **OrchestrationDefinition** — accepted topology/structure (conceptual; exact class name not frozen) |
| **Topology vs runtime** | `NodeId` ≠ `ExecutionId` — one node may instantiate many Executions |
| **Planner boundary** | `PlannerProposal` ≠ accepted topology until validation + governance + acceptance |
| **Strategies** | Topology may coordinate agents, inference Executions, nested orchestrations, mixed strategies |
| **Single-agent** | Direct agentic Execution does **not** require orchestration merely because one agent is involved |
| **Parallelism** | Graph-level caps (`max_parallel_nodes`, `max_inflight_nodes`) — not infra scaling |
| **Nexus relation** | Nexus **interprets** accepted topology and schedules child Executions |
| **UER relation** | UER owns **how** each child Execution behaves |
| **Maturity** | Four-axis statement in [Current maturity](#current-maturity) |
| **Go deeper** | [Engineering canon](#engineering-canon) · [production gates satellite](satellites/ORCHESTRATION_production_gates.md) · [plan](../maintainers/plans/ORCHESTRATION.md) · [Nexus](NEXUS_EXECUTION_FLOW.md) · [UEA](UNIFIED_EXECUTION_ARCHITECTURE.md) |

## Flagship architecture visual

Domain control-plane view — distinguish **CURRENT** diagram terminology from **TARGET** semantics in surrounding copy.

<a href="assets/fullsize/orchestration-control-plane.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/orchestration-control-plane-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/orchestration-control-plane-light.svg">
  <img
    alt="Conceptual diagram: application needs flow into OrchestrationProfile configuring planner, graph, parallelism, and merge; Nexus executes orchestration control-flow; UER supplies Execution behavior."
    src="assets/orchestration-control-plane-light.svg"
  >
</picture>
</a>

Tier-3 profiles shape orchestration configuration; Orchestration owns the **structure** contract. Agents do not create private orchestration runtimes.

## OrchestrationDefinition (TARGET)

**OrchestrationDefinition** represents **accepted** topology and structure — declarative configuration and/or an authorized plan that passed structural validation, governance, and acceptance.

Provenance must be preserved:

| Source | Meaning |
| ------ | ------- |
| **configured** | Explicit application/platform configuration (`ApplicationGraphSpec`, profile) |
| **planned / proposed** | `PlannerProposal` or planner output — **not** automatically executable |
| **accepted** | Validated, governable topology Nexus may interpret |

**StrategyResolver** and planners **MUST NOT** silently invent arbitrary topology. No “magic AI router” without explicit acceptance.

**CURRENT IMPLEMENTATION:** `ApplicationGraphSpec` → `graph_spec_to_plan()` → `NexusPlan` seed → `GraphExecutor` — migration wiring; nodes still map to agent execution units on harness paths.

## Topology is not runtime identity

**ORCH-INV-001 / NEXUS-INV-003:** Topology `NodeId` is a **definition slot**. `ExecutionId` is a **runtime instance**.

| Topology node | Execution |
| ------------- | --------- |
| Stable logical position in accepted definition | Independently schedulable work unit |
| May instantiate 0, 1, or many times | Unique in Attempt context (+ Run/Attempt lineage) |
| Not retry/cancel identity | Canonical retry/cancel tree member |

Fan-out: `NodeId = researcher` → `Execution E2`, `E3`, `E4`.

<a href="UNIFIED_EXECUTION_ARCHITECTURE.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/unified-execution-topology-vs-execution-tree-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/unified-execution-topology-vs-execution-tree-light.svg">
  <img
    alt="Definition topology NodeId versus runtime Execution Tree ExecutionId."
    src="assets/unified-execution-topology-vs-execution-tree-light.svg"
  >
</picture>
</a>

**TARGET:** Nexus schedules **child Executions** from validated topology — does not equate nodes to runtime instances.

<a href="UNIFIED_EXECUTION_ARCHITECTURE.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/unified-execution-orchestration-nexus-flow-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/unified-execution-orchestration-nexus-flow-light.svg">
  <img
    alt="Orchestration strategy: Nexus schedules child Executions from OrchestrationDefinition at the execution boundary."
    src="assets/unified-execution-orchestration-nexus-flow-light.svg"
  >
</picture>
</a>

**CURRENT IMPLEMENTATION:** Nexus `GraphExecutor` remains agent-centric on harness paths — see [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md).

## How configuration reaches Nexus (CURRENT harness path)

```text
Application requirements
        ↓
OrchestrationProfile (+ optional ApplicationGraphSpec)
        ↓
planner_kind · classifier_kind · graph topology · parallelism · merge · resilience
        ↓
build_nexus_loop_from_environment() wires planner/classifier
        ↓
UnifiedTaskRunner → NexusLoop (CURRENT entry for orchestrated harness tasks)
        ↓
classify → plan → graph execution → TaskResult
```

**TARGET path** (orchestration-strategy Execution): accepted `OrchestrationDefinition` → parent Execution → Nexus → child Executions — same Run/Attempt unless lifecycle boundary changes.

1. **Profile resolution** — `ApplicationEnvironmentProfile` carries `OrchestrationProfile` and optional `graph_spec`.
2. **Kind wiring** — `planner_kind` / `classifier_kind` resolve at Nexus bootstrap (**fail-fast** on unknown kinds).
3. **Plan seeding** — `ApplicationGraphSpec` can seed `NexusPlan` via `GraphSpecSeedingPlanner` (ORCH-2) — **CURRENT**.
4. **Graph execution** — **CURRENT:** `GraphExecutor` honors topology caps; **TARGET:** child Execution scheduling.
5. **Completion** — merge via `FinalResponseComposer` / `merge_strategy`; resilience events follow REL/UER.

Step-by-step **CURRENT** narrative: [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md).

## Orchestration vs Nexus vs UER

| Layer | Core question | Owns |
| ----- | ------------- | ---- |
| **Orchestration** | How should work be **structured**? | Accepted topology, profile configuration, planner proposal → acceptance |
| **Nexus** | What executes **next** for this orchestration Execution? | Readiness, scheduling child Executions, fan-in, orchestration-level failure coordination |
| **UER** | How does each **Execution** behave? | Lifecycle, attempts, events, retry/HITL/cancel identity |

```text
Orchestration → structure (definition)
Nexus         → control-flow over accepted structure
UER           → Execution semantics
```

## Orchestration vs Reasoning / Planning

| Concern | Owner |
| ------- | ----- |
| Planner/classifier **kind** selection | Orchestration profile |
| Planner/classifier **implementation** | [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) |
| **PlannerProposal** | Reasoning/planner output — proposal until accepted |
| **Accepted topology** | Orchestration + validation/governance |
| Tool planning inside a step | Third plane — [`TOOLS.md`](TOOLS.md) |

**Rule:** config selection ≠ reasoning implementation. Orchestration chooses **which** planner/classifier; Reasoning owns **how** plans are produced.

## OrchestrationProfile (CURRENT contract)

Real configurable dimensions on the platform profile:

| Dimension | Profile surface | Notes |
| --------- | --------------- | ----- |
| Planner strategy | `planner_kind` | `default` → `TaskPlanner`; `engine` → engine-backed planner |
| Classifier strategy | `classifier_kind` | `default`, `rules`, `llm` + `IntentRoute` |
| Graph topology | `ApplicationGraphSpec` / `graph_spec` | Declarative nodes, edges, capabilities |
| Parallel cap (batch) | `max_parallel_nodes` | Semaphore per topological batch (ORCH-3) |
| Global inflight cap | `max_inflight_nodes` | Cross-graph semaphore; `GRAPH_BACKPRESSURE` |
| Merge | `merge_strategy` | `concat`, `last_wins`, `structured_json`, `citation_preserving` |
| Resilience / partial | `allow_partial_result`, retry policy fields | Configuration — REL owns semantics |
| Long-running | `long_running_enabled` + helpers | Checkpoint/schedule posture |
| Delegation depth | `max_delegation_depth` | Limits nested expansion (**CURRENT** graph model) |
| Intent routing | `intent_routes` | Classifier-driven orchestration tokens |

Orchestration configuration **cannot expand** effective authority or budget beyond Run/root allowances.

## Strategy catalog (summary)

Public summary — full catalog in [satellite §50](satellites/ORCHESTRATION_production_gates.md#50-orchestration-strategies-catalog). **Done** in plan ≠ production-qualified at scale.

| Pattern | Structure | Notes |
| ------- | --------- | ----- |
| **Direct agentic** | No orchestration required | `Execution` → agentic → `AgentEngine` — not “single-agent orchestration” |
| **Orchestration — one logical node** | Minimal topology | Still orchestration strategy — distinct from direct agentic |
| **Sequential pipeline** | `depends_on` chain | May mix executor strategies per node (target) |
| **Parallel fan-out / fan-in** | Independent logical work + merge | Nexus schedules concurrent child Executions subject to caps |
| **Declarative graph** | `ApplicationGraphSpec` seeds plan | **CURRENT** seeding path |
| **Delegation** | Bounded sub-work in topology | **TARGET:** child Execution; **CURRENT:** `DELEGATES_TO` / `DelegationSpec` |
| **Handoff** | Runtime topology continuation | **TARGET:** child Execution route; **CURRENT:** `HandoffCoordinator` |
| **Evaluator loop** | Quality gate pattern | `CoordinationPattern.EVALUATOR_LOOP` |
| **Swarm** | Parallel explorers under budget | ORCH-5.1 caps |
| **Dynamic advisory** | `select_coordination_pattern()` | Bounded helper — production should prefer explicit `graph_spec` |

Agents **must not** call each other directly — collaboration flows through platform orchestration boundaries, `SharedTaskContext`, and artifacts.

## Declarative graph (CURRENT)

```text
ApplicationGraphSpec  →  graph_spec_to_plan()  →  NexusPlan seed  →  GraphExecutor  →  TaskResult
```

- Graph config belongs to platform/application profile — not inside Tier-2 agent code.
- Topology declares dependencies, parallelism, delegation edges.
- Nexus owns runtime interpretation; Orchestration owns **what** topology is configured/accepted.

## Parallelism and backpressure

Independent logical work may run concurrently when Nexus schedules ready child Executions — subject to:

| Control | Effect |
| ------- | ------ |
| `max_parallel_nodes` | Concurrent nodes in one topological batch (**CURRENT** graph executor) |
| `max_inflight_nodes` | Global inflight cap; `GRAPH_BACKPRESSURE` |
| Tenant / host concurrency | Cross-task fairness — host runtime |
| `max_delegation_depth` | Nested delegation limit |

**Boundary:** [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) owns infrastructure replicas — not graph scheduling caps.

## Merge

Fan-out requires fan-in. **TARGET:** merge over heterogeneous **child Execution results**. **CURRENT:** `FinalResponseComposer` + `merge_strategy` on agent node outputs.

| Strategy | Behavior (CURRENT) |
| -------- | ------------------ |
| `concat` | Concatenate summaries |
| `last_wins` | Last successful node |
| `structured_json` | Per-agent status JSON |
| `citation_preserving` | Structured merge with citations |

## Resilience (high level)

Orchestration **configures** posture; REL/UER own lifecycle semantics:

- Graph retry policy — orchestration-level decisions via Nexus; identity via UER
- Partial results — `allow_partial_result`
- Checkpoint / long-running — durable Run checkpoint includes orchestration state component
- HITL — pause points; Nexus routes continue/replan after decision

## Delegation, handoff, and specialization

| Concept | TARGET | CURRENT |
| ------- | ------ | ------- |
| **Specialization** | Capability-based routing into topology positions | Registry + classifier + `AgentRouter` |
| **Delegation** | Bounded child Execution from topology | `DelegationSpec`, `DELEGATES_TO` |
| **Handoff** | Route to child Execution / topology position | `HandoffCoordinator` node insertion |
| **Agent-to-agent** | Anti-pattern | — |

## Dynamic strategy selection

`select_coordination_pattern()` — bounded advisory (AUDIT-IDEAL-9.3 **Done**). Production hosts should prefer explicit `graph_spec`.

## Queue / async boundary (summary)

Long-running work uses platform queueing (`intergrax/queueing`) — not ad-hoc agent threads. ORCH-6 ships lab async dispatch; product exposure varies by host.

## Interaction postures (simplified)

| Posture | Orchestration touchpoint |
| ------- | ------------------------ |
| **Reactive** | Sync `Task` through wired profile |
| **Daemon / background** | Profile + scheduler flags |
| **Scheduled** | Queue/scheduler + `long_running` |
| **Long-running** | Checkpoint-friendly profile |
| **HITL** | Plan/policy pause; Nexus resumes orchestration path |

Full matrix: [satellite §55](satellites/ORCHESTRATION_production_gates.md#55-interaction-posture--orchestration-matrix).

## Responsibility boundaries

### Orchestration owns

- Accepted topology / `OrchestrationDefinition` semantics (target)
- `OrchestrationProfile` configuration and strategy catalog
- Declarative `ApplicationGraphSpec` → acceptance contract (target) / plan seed (current)
- Planner/classifier **kind** selection and fail-fast wiring
- Graph-level parallelism, merge strategy, resilience **policy** fields
- Coordination pattern catalog (satellite §56)

### Orchestration does not own

- Nexus runtime scheduling — [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md)
- Execution lifecycle, `ExecutionId` — [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md)
- Planner implementation — [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md)
- Infrastructure autoscaling — [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md)
- Authority, budget ledger, observability persistence

### Applications (Tier-3) configure

- `ApplicationEnvironmentProfile.orchestration` and `graph_spec`
- Host presets — via profile, not bespoke orchestration runtimes

## Orchestration-specific invariants (ORCH-INV-*)

ORCH-INV specialize frozen UEA for Orchestration. Full set: [UEA §24](UNIFIED_EXECUTION_ARCHITECTURE.md#24-architectural-invariants-uea-inv).

| ID | Invariant |
| -- | --------- |
| **ORCH-INV-001** | Orchestration topology is definition state, not runtime execution identity |
| **ORCH-INV-002** | Accepted topology must be structurally valid before meaningful orchestration execution |
| **ORCH-INV-003** | Planner output is proposal until accepted as executable topology |
| **ORCH-INV-004** | Orchestration configuration cannot expand effective authority or budget |
| **ORCH-INV-005** | One topology node may instantiate multiple Executions |
| **ORCH-INV-006** | Orchestration may coordinate heterogeneous executor strategies, not only Agents |
| **ORCH-INV-007** | Topology, Execution Tree, and event causality are distinct structures |
| **ORCH-INV-008** | No product/application may create a private orchestration runtime outside the platform boundary |

## Harness-proven vs not automatically production-qualified

### Harness / platform implemented (CURRENT path)

Planner/classifier wiring; graph-spec seeding; parallel caps; strategy catalog; CFG simulation; swarm guard; queue adapter scaffold.

### Not automatically production-qualified

- `OrchestrationDefinition` / child Execution target implemented
- Every product `graph_spec` at operational scale (FLOW-8 **Deferred**)
- Every coordination pattern qualified
- Customer operational windows

## Current maturity

Architecture maturity: **A4** *(UEA-aligned target)*
Implementation maturity: **I3–I4** *(CURRENT harness I4)*
Production readiness: **P2**  
Evidence maturity: **E3**

- **A4** — ORCH-INV closure; topology vs Execution Tree on hub; UEA alignment.
- **I3–I4** — ORCH-1–6 **Done** on **CURRENT** agent-centric harness. Execution-centric topology contract and heterogeneous per-node strategies are **target / planned**.
- **P2** — Harness/lab profiles; no universal production handoff.
- **E3** — Unit/gate + bounded integration; **no** public Orchestration proof route.

Harness ORCH **Done** ≠ Execution-centric orchestration implemented.

### Protocol v2 orchestration target invariants (2026-08-18)

Accepted Protocol v2 audit layer [`ORCHESTRATION`](../../audit_results/2026-08-18/ORCHESTRATION.md) (**FAIL**, 5 ACCEPTED findings). Prior Phase ORCH **Done** rows remain **historical delivery facts** on the then-current architecture.

1. **Canonical graph-node executable identity** — **CURRENT** remediation target; **frozen target** decouples topology nodes from agent identity ([`AUDIT-20260818-ORCHESTRATION-01`](../../audit_results/2026-08-18/ORCHESTRATION.md)).
2. **Typed fail-fast orchestration configuration** — ([`AUDIT-20260818-ORCHESTRATION-02`](../../audit_results/2026-08-18/ORCHESTRATION.md)).
3. **Exact delegation-edge provenance** — ([`AUDIT-20260818-ORCHESTRATION-03`](../../audit_results/2026-08-18/ORCHESTRATION.md)).
4. **Single canonical `OrchestrationProfile` ownership** — ([`AUDIT-20260818-ORCHESTRATION-04`](../../audit_results/2026-08-18/ORCHESTRATION.md)).
5. **Static graph cycle rejection** — ([`AUDIT-20260818-ORCHESTRATION-05`](../../audit_results/2026-08-18/ORCHESTRATION.md)).

Remediation: **ORCH-CONTRACT-INTEGRITY**, **ORCH-DELEGATION-INTEGRITY** in [plan](../maintainers/plans/ORCHESTRATION.md). **Not implemented** by audit persistence.

## Evidence / proof

| Evidence class | What exists | What it does not prove |
| -------------- | ----------- | ---------------------- |
| Architecture | This hub, UEA, production-gates satellite | Production operation |
| Unit / gate | Wiring, graph spec → plan, parallel cap | Every host topology |
| Integration | CFG simulation, Nexus graph integration | Universal qualification |
| Public product proof | **None** for Orchestration domain | — |

## Relationship to Intergrax

| Neighbor | Relationship |
| -------- | ------------- |
| [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) | Semantic authority |
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | Interprets accepted topology — schedules child Executions (target) |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | **How** each Execution behaves — not “inside each graph node” as target framing |
| [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | Planner/classifier implementation |
| [`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md) | Tier-3 bootstrap wires profile → Nexus |
| [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) | Retry layers, HITL |
| [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) | Infra scaling vs graph backpressure |
| [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) | Policy on orchestration hot paths |

## Extensibility

| Surface | Role |
| ------- | ---- |
| `OrchestrationProfile` | Host collaboration defaults |
| `ApplicationGraphSpec` | Declarative topology (**CURRENT**) |
| `planner_kind` / `classifier_kind` | Platform planner/classifier selection |
| `CoordinationPattern` + merge strategies | Named patterns |
| Queue/async helpers | `run_async`, `QueuedNexusExecutionAdapter` |

Do not expose private graph executors or agent-to-agent RPC as public orchestration APIs.

## Go deeper

| Depth | Route |
| ----- | ----- |
| **Engineering canon** | [Below](#engineering-canon) — §9–§26 (**CURRENT** loop fundamentals) |
| **Production gates** | [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md) |
| **Implementation plan** | [`maintainers/plans/ORCHESTRATION.md`](../maintainers/plans/ORCHESTRATION.md) |
| **Nexus flow** | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) |
| **UEA / UER** | [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) |

## Implementation readiness

### 1. TARGET STATE

Neutral accepted topology contract (`OrchestrationDefinition`); planner proposal ≠ accepted topology; `NodeId` ≠ `ExecutionId`; heterogeneous execution requirements per topology slot; one Execution Tree for runtime identity.

### 2. CURRENT STATE

`ApplicationGraphSpec` + `OrchestrationProfile` on harness path; graph nodes map to agent execution units; `UnifiedTaskRunner` → `NexusLoop` for many tasks.

### 3. GAPS

Accepted topology contract separate from `NexusPlan` seed; decouple nodes from agent identity; child Execution scheduling semantics; provenance (configured/planned/accepted).

### 4. DEPENDENCIES

- UEA (authority)
- UER Execution Boundary
- Nexus child scheduling ([`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md))
- Code mapping: **UE-DOC-0.9**

### 5. MIGRATION ORDER (high level)

1. Define neutral accepted topology contract
2. Separate planner proposal from accepted topology
3. Decouple topology nodes from Agent identity
4. Allow heterogeneous execution requirements per node
5. Preserve `ApplicationGraphSpec`/`Profile` through migration where compatibility requires
6. Remove graph node == agent execution assumptions

### 6. DO NOT VIOLATE

- ORCH-INV-* and UEA-INV-*
- Topology as Execution identity
- Orchestration expanding authority/budget
- Private orchestration runtimes in Tier-3 products

### 7. ACCEPTANCE CONDITIONS

- Accepted topology validated before Nexus execution
- Provenance preserved (configured / planned / accepted)
- Topology scheduling distinct from Execution Tree
- TARGET/CURRENT labeled where implementation lags

---
## Maintainer and Cursor context

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/ORCHESTRATION.md`](../maintainers/plans/ORCHESTRATION.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 3, 9 · multi-agent patterns: audit layer 10 (cross-ref satellite §50)  
**Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md)  
**Reasoning / planning canon:** [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) (audit layer 7)  
**Last updated:** 2026-08-26 — UE-DOC-0.5 alignment with frozen UEA (ORCH-INV, OrchestrationDefinition)
**Elastic capacity:** [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md#production-boundary) (infra capacity signals and scaling — **not** graph scheduling, agent topology or orchestration brain)

### Cursor read scope (token budget)

**Do not read this entire file in one session** (ORCHESTRATION canon).

- **Implement / audit default:** public front + engineering canon §9–§26 below.
- **Strategy / CFG / production depth:** [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md) §47+ only.
- **Plan hub:** [`plan/ORCHESTRATION.md`](../maintainers/plans/ORCHESTRATION.md) (scoped §6 only).
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

### Document roles (read order)

| Document | Role |
|----------|------|
| **This file (`ORCHESTRATION.md`)** | Public front + engineering canon §9–§26 (loop/graph fundamentals) |
| [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md) | Production gates, strategy catalog §50+, master CFG §56+, audit §59 |
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | **Runtime narrative** — sequence diagrams, UC-*, edge cases, code paths |
| [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | Classification, planning implementation |
| AGENT_CREATION_GUIDE Appendix I | Author control plane (`OrchestrationProfile`, wiring) |

**Rule:** strategy **names and configuration** — satellite §50–§56; step-by-step runtime truth — **NEXUS_EXECUTION_FLOW**.

### Architecture satellites (read on demand)

| Satellite | Contents |
|-----------|----------|
| [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md) | §47–§59 — production gates, strategy catalog, CFG canon, execution audit |

> **Cursor context budget:** read hub public front + **at most one** satellite per session.

### Table of contents (engineering canon & satellite routes)

| § | Location | Topic |
|---|----------|--------|
| [§9–§26](#engineering-canon) | **Hub** | Nexus loop, responsibilities, lifecycle, graph fundamentals |
| [§47+](satellites/ORCHESTRATION_production_gates.md) | Satellite | Production gates, intake, queueing |
| [§50](satellites/ORCHESTRATION_production_gates.md#50-orchestration-strategies-catalog) | Satellite | Coordination pattern catalog |
| [§51](satellites/ORCHESTRATION_production_gates.md#51-parallelism-merge-and-backpressure) | Satellite | Parallelism, merge, backpressure |
| [§52](satellites/ORCHESTRATION_production_gates.md#52-resilience-in-orchestration) | Satellite | Retry, checkpoint, failover, partial |
| [§53](satellites/ORCHESTRATION_production_gates.md#53-specialization-and-agent-collaboration) | Satellite | Capability routing, delegation, handoff |
| [§54](satellites/ORCHESTRATION_production_gates.md#54-maturity-and-gap-register) | Satellite | Legacy L0–L4 scorecard (historical) |
| [§55](satellites/ORCHESTRATION_production_gates.md#55-interaction-posture--orchestration-matrix) | Satellite | Posture × pattern matrix |
| [§56](satellites/ORCHESTRATION_production_gates.md#56-platform-interaction--multi-agent-configuration-canon) | Satellite | **Master configuration canon** |
| [§57](satellites/ORCHESTRATION_production_gates.md#57-synchronous-and-asynchronous-execution-postures) | Satellite | Sync vs async dispatch |
| [§58](satellites/ORCHESTRATION_production_gates.md#58-platform-runtime-capabilities-index) | Satellite | Cross-cutting runtime index |
| [§59](satellites/ORCHESTRATION_production_gates.md#59-platform-execution-audit---gaps-technical-debt-discrepancies) | Satellite | Execution-surface audit |

**Authoring rule:** Tier-3 host design starts at **satellite §56**; runtime step-by-step truth remains in **NEXUS_EXECUTION_FLOW**.

### Documentation topology note

Hub §9–§26 preserve loop and graph **fundamentals**. Strategy catalog, CFG matrices, and production gates (§47–§59) live in the **production-gates satellite** — not duplicated here. Prior hub TOC anchors for §50+ pointed at missing in-file sections; routes now target the satellite.

---

## Engineering canon

# 9.1 Global Nexus Loop

**CURRENT IMPLEMENTATION:** On the harness path, `NexusLoop` is the de facto orchestrated task control loop.

**TARGET ARCHITECTURE:** Nexus operates when a parent Execution uses **orchestration strategy** — not mandatory for direct inference or ordinary agentic Executions.

The Nexus loop controls orchestration **control-flow** for accepted topology (not universal platform entry).

Responsibilities:

- receive user task
- classify task (see [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) §9)
- determine complexity
- create or update plan (see [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) §10)
- select agents
- prepare context
- schedule/request child Executions (**TARGET**); route to agents via agentic children (**CURRENT:** `GraphExecutor` → `AgentRouter`)
- evaluate results
- decide next step
- handle retries
- coordinate parallel work
- coordinate sequential work
- request human approval when required
- finalize output

**Detailed runtime narrative** (sequence diagrams, decision matrix, `FLOW-GAP.*` plan rows): [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §4–§18.

Pseudo-flow:

```text
while task.status not in [completed, failed, cancelled]:

    current_state = load_task_state(task_id)

    reasoning_result = reason_about_current_state(current_state)

    next_action = determine_next_action(reasoning_result)

    if next_action.type == "execute_agent":
        result = execute_agent(next_action.agent, next_action.input)
        store_result(result)

    if next_action.type == "execute_parallel_agents":
        results = execute_agents_in_parallel(next_action.agents)
        store_results(results)

    if next_action.type == "ask_human":
        pause_and_request_human_input()

    if next_action.type == "retry":
        execute_retry_policy()

    validation_result = validate_current_state()

    update_task_state(validation_result)
```

---

# 9.2 Local Agent Loop

Agents MAY have local loops — but loops MUST be **runtime-controlled** (§42.32, §42.33).

Local loops are allowed when an agent requires multiple internal steps.

The agent loop MUST be bounded by:

- the input contract
- the output contract
- max steps
- max time
- max cost
- allowed tools
- validation rules

Pseudo-flow:

```text
while local_goal_not_completed and limits_not_exceeded:

    local_state = inspect_local_state()

    local_next_step = decide_local_next_step(local_state)

    local_result = execute_local_step(local_next_step)

    validate_local_result(local_result)

    update_local_state(local_result)

return agent_output_artifact
```

---

# 9.3 Why Both Loops Are Required

If only Nexus has a loop:

- Nexus becomes too large
- Nexus micromanages every domain
- domain-specific logic leaks into the runtime
- implementation becomes rigid

If only agents have loops:

- global orchestration becomes chaotic
- agents become mini-platforms
- state becomes fragmented
- retries become inconsistent
- final output becomes unpredictable

Correct decision:

> Nexus has the global loop. Agents may have bounded local loops.

---


---

# 10. Nexus Responsibilities

Nexus is responsible for the following areas.

## 10.1 Task Intake

Nexus receives tasks from:

- chat interface
- Slack
- Teams
- API
- CLI
- internal scheduler
- webhook
- event trigger

Task intake normalizes input into a standard Task object.

---

## 10.2 Task Classification

Nexus classifies the task.

Possible classifications (**CURRENT** classifier outputs — not synonymous with Execution strategy):

- simple question
- single-capability routed work (**not** “must use orchestration”)
- multi-capability / multi-node topology work (**not** “multi-agent == orchestration”)
- long-running workflow
- monitoring task
- scheduled task
- human-approval-required task
- unsafe task
- unsupported task

---

## 10.3 Planning

Nexus creates a plan when needed.

A plan may include:

- steps
- dependencies
- agent assignments
- required tools
- expected artifacts
- validation criteria
- human approval points
- risk level

---

## 10.4 Agent Selection

Nexus selects agents based on:

- task intent
- agent registry
- declared capabilities
- required tools
- previous performance
- cost
- availability
- risk level

---

## 10.5 Execution Graph

Nexus manages the execution graph.

The execution graph defines:

- nodes
- dependencies
- parallel branches
- sequential branches
- waiting states
- retry states
- failed states
- completed states

---

## 10.6 State Management

Nexus owns global task state.

Global state includes:

- task id
- run id
- user input
- normalized task
- current plan
- execution graph
- agent outputs
- tool outputs
- validation results
- human messages
- final result
- status

---

## 10.7 Context Management

Nexus decides what context is passed to each agent.

Agents MUST receive only the context needed for their bounded task.

Nexus prevents uncontrolled context growth.

---

## 10.8 Tool And Adapter Access Policy

Nexus defines which tools and adapters an agent may use.

Agents should not automatically receive access to every integration.

Tool access should be explicit.

---

## 10.9 Validation

Nexus validates whether the global task is complete.

Validation can include:

- schema validation
- rule validation
- secondary agent validation
- tests
- consistency checks
- human approval

---

## 10.10 Final Response

Nexus composes the final response to the user.

Agents produce artifacts.

Nexus decides how artifacts are presented.

---


---

# 23. Task Lifecycle

Every task should move through explicit states.

Recommended lifecycle:

```text
created
    -> classified
    -> planned
    -> waiting_for_resources
    -> running
    -> waiting_for_human
    -> validating
    -> completed
```

Failure states:

```text
failed
cancelled
expired
partially_completed
needs_more_information
```

Every transition should be logged.

---


---

# 24. Execution Graph

Complex tasks should be represented as execution graphs.

An execution graph contains:

- nodes
- dependencies
- execution status
- assigned agent
- input
- output
- validation result
- retry count

Example:

```text
Task: Find business partner for AI logistics project

Node 1: Analyze project description
Node 2: Define partner criteria
Node 3: Search companies
Node 4: Enrich company profiles
Node 5: Score companies
Node 6: Validate ranking
Node 7: Generate final recommendation
```

Some nodes may run sequentially.

Some nodes may run in parallel.

---


---

# 25. Sequential And Parallel Execution

Nexus decides whether execution is sequential or parallel.

Sequential execution is preferred when:

- later steps depend on previous outputs
- task risk is high
- context must be controlled
- quality is more important than speed

Parallel execution is allowed when:

- subtasks are independent
- agents work on separate data
- research can be split
- validation can run independently

Nexus must merge parallel results.

---


---

# 26. Long Running Tasks

Intergrax must support long-running tasks.

Examples:

- monitor Reddit for problem signals for 30 days
- onboard new employees for 2 weeks
- analyze monthly sales data
- audit vendors over multiple stages
- review a large document set

Long-running tasks require:

- persistent state
- resumability
- scheduled execution
- progress updates
- failure recovery
- human interruption
- partial results

---


---
