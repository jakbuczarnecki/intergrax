# Orchestration, Nexus, and Execution Graph

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/ORCHESTRATION.md`](../plan/ORCHESTRATION.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 3, 9 · multi-agent patterns: audit layer 10 (cross-ref §50)  
**Reasoning / planning canon:** [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) (audit layer 7)  
**Elastic capacity:** [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) (infra replicas — not graph scheduling)  
---

## Document roles (read order)

| Document | Role |
|----------|------|
| **This file (`ORCHESTRATION.md`)** | Orchestration **manifest** + **execution strategy catalog** (§50–§54) |
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | **Runtime narrative** — sequence diagrams, UC-*, edge cases, code paths |
| [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | Classification, planning, agent topology in plans |
| [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) | Author control plane (`OrchestrationProfile`, wiring) |

**Rule:** strategy **names and selection** live here; step-by-step runtime truth lives in **NEXUS_EXECUTION_FLOW**.

## Table of contents (strategy & execution)

| § | Topic |
|---|--------|
| [§50](#50-orchestration-strategies-catalog) | Coordination pattern catalog |
| [§51](#51-parallelism-merge-and-backpressure) | Parallelism, merge, backpressure |
| [§52](#52-resilience-in-orchestration) | Retry, checkpoint, failover, partial |
| [§53](#53-specialization-and-agent-collaboration) | Capability routing, delegation, handoff |
| [§54](#54-maturity-and-gap-register) | Maturity scorecard |

---

# 9.1 Global Nexus Loop

The Nexus loop is mandatory.

The Nexus loop controls global execution.

Responsibilities:

- receive user task
- classify task (see [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) §9)
- determine complexity
- create or update plan (see [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) §10)
- select agents
- prepare context
- execute agents
- evaluate results
- decide next step
- handle retries
- coordinate parallel work
- coordinate sequential work
- request human approval when required
- finalize output

**Detailed runtime narrative** (sequence diagrams, decision matrix, `FLOW-GAP.*` plan rows): [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §4–§18.

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

Possible classifications:

- simple question
- single-agent task
- multi-agent task
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

# 47. Checklist For Nexus Changes

Before changing Nexus, answer:

```text
1. Is this change domain-agnostic?
2. Does it belong in runtime rather than an agent?
3. Does it improve orchestration, lifecycle, validation or observability?
4. Does it preserve layer boundaries?
5. Does it make agents easier to implement?
6. Does it avoid hardcoded business logic?
7. Is the behavior traceable?
8. Can it support future agents?
9. Does it emit RuntimeEvents and respect UAEP (§42)?
10. Does it integrate with HookRegistry / middleware pipeline (§42.20)?
```

If the change is domain-specific, it probably belongs in an agent.

---

---

# 48. Task Intake and TaskEnvelope

All entrypoints MUST normalize into a common intake contract before NexusLoop.

## 48.1 TaskEnvelope (minimum)

```text
TaskEnvelope:
    task_id / run_id
    tenant_id
    user_id | service_id
    source_channel          # api | cli | slack | teams | webhook | scheduler
    raw_input
    constraints             # SLA, risk class, budget caps
    correlation_ids         # trace_id parent
```

## 48.2 Intake pipeline

```text
Surface adapter -> contract validation -> TaskEnvelope -> TaskClassifier -> Planner
```

| Module | Role |
|--------|------|
| `applications/_shared/task_intake.py` | Shared intake helpers |
| `fastapi_core/` | HTTP auth + request context |
| `runtime/nexus/orchestration/` | Classifier, planner, graph |
| `runtime/interactions/` | Slack/Teams interaction adapters |

**Audit layer:** INTEGRAX_HARNESS_AUDIT_MAP §3 (Interface and Task Intake).

---

# 49. Scheduler and Queueing

Long-running and asynchronous work uses the Tier-0 queueing plane — not ad-hoc threads in agents.

## 49.1 Components

| Module | Role |
|--------|------|
| `intergrax/queueing/` | Task index, registry, worker contracts |
| `intergrax/distributed/` | Rate limiting, distributed locks |
| Integration `message_bus` providers | Celery, RabbitMQ, Redis, Kafka |

## 49.2 Orchestration integration

- `OrchestrationProfile.long_running` enables checkpointed schedules.
- Graph batch concurrency caps prevent provider overload.
- Backpressure and semaphore limits are policy-aware (see [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §9).

**Elastic capacity (replicas, workers, provisioning):** [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) — ECP consumes `GRAPH_BACKPRESSURE` and queue signals; this section covers in-process scheduling only.

**Plan:** [`plan/ORCHESTRATION.md`](../plan/ORCHESTRATION.md) Phase ORCH.

---

# 50. Orchestration Strategies Catalog

Multi-agent and multi-step execution MUST use an **explicit coordination pattern** (IDEAL §6.4, AUDIT_MAP §10). Intergrax implements patterns through **declarative graphs**, **runtime handoff**, and **CVL evaluator loops** — not ad-hoc agent-to-agent calls.

## 50.1 Pattern catalog

Canonical enum: `CoordinationPattern` in `intergrax/runtime/architecture/multi_agent_coordination.py` (Phase V-MA **Done**).

| Pattern | When to use | Harness mapping | Runtime depth |
|---------|-------------|-----------------|---------------|
| **Hierarchical** | Top-down plan; planner delegates to specialized executors | `graph_spec` + `DELEGATES_TO` ([ADR-FLOW-001](adr/ADR-FLOW-001.md)) | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §13 |
| **Orchestrator–worker** | Central Nexus plan; workers are graph nodes with capabilities | `TaskPlanner` / `graph_spec` → sequential or batched nodes | §12 UC-4, §42.43 |
| **Supervisor–worker** | Quality/policy supervision over workers; re-plan on failure | HITL + `AgentDecision.INTERRUPT` + policy hooks | UAEP §42.8, FLOW §11 |
| **Peer-to-peer** | Independent subtasks; parallel decomposition | Topological **batches** + `MergePolicy` | §51 below |
| **Swarm** | Many lightweight explorers; aggregate under budget | Parallel batch + cost/step caps (catalog + selection matrix) | **Partial** — catalog Done; full swarm runtime ORCH-5.* |
| **Evaluator-loop** | Critique–revise before finalize | `CoordinationPattern.EVALUATOR_LOOP` + `EvaluatorLoopExecutor` | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |

```text
Pattern selection (helper): select_coordination_pattern(constraints)
    → uses risk / latency / cost / complexity levels
    → authors SHOULD override via declarative graph_spec for production hosts
```

**Narrative flows:** [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §12 (UC-1–UC-9), §27. **Reference product flow:** UAEP [§42.43](UNIFIED_EXECUTION_RUNTIME.md#4243-multi-agent-collaboration-flow-reference) (PM → UX → Legal → Validator → Human).

## 50.2 Collaboration vs specialization

| Concept | Mechanism | Owner |
|---------|-----------|-------|
| **Specialization** | Tier-2 agents declare **capabilities**; Nexus routes by capability match | `AgentRegistry`, `TaskClassifier`, `AgentRouter` |
| **Collaboration** | Multiple specialized agents in one **ExecutionGraph** via plan steps / edges | `NexusPlan`, `GraphExecutor` |
| **Isolation** | Delegation namespaces, tool policy on child nodes | `DelegationSpec`, `SubtaskContract` |

Agents MUST NOT call each other directly — all collaboration via **SharedTaskContext**, artifacts, and graph nodes (§53).

## 50.3 Anti-patterns

- Single mega-agent emulating multi-agent topology inside one UAEP loop.
- Hidden parallel branches without `depends_on` / graph_spec edges.
- Swarm without budget envelope (`max_delegation_depth`, cost profile).
- Pattern name in docs but undeclared in host `graph_spec` or plan metadata.

---

# 51. Parallelism, Merge, and Backpressure

## 51.1 Sequential vs parallel (summary)

Full rules: §25 above. Runtime implementation: [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §9.

| Mode | Prefer when | Harness mechanism |
|------|-------------|-------------------|
| **Sequential** | Output dependency, high risk, strict context control | `depends_on` chain in `NexusPlan` |
| **Parallel** | Independent subtasks, split research, parallel validation | `ExecutionGraph.batches()` + `asyncio.gather` |

## 51.2 Concurrency controls

| Control | Profile field | Effect |
|---------|---------------|--------|
| Parallel within batch | `max_parallel_nodes` | Semaphore on concurrent nodes in one topological batch |
| Global inflight cap | `max_inflight_nodes` | Semaphore across graph; emits `GRAPH_BACKPRESSURE` |
| Tenant cap | `RuntimeEngine.max_parallel_per_tenant` | Cross-task fairness (UAEP bridge) |
| Delegation depth | `max_delegation_depth` | Limits nested subagent expansion |

```text
GRAPH_BACKPRESSURE  →  ops:backpressure  →  optional input to ECP (infra scale)
                      →  does NOT by itself add replicas (see ELASTIC_CAPACITY)
```

## 51.3 Merge policies

After parallel or sequential multi-node runs, `FinalResponseComposer` applies `OrchestrationProfile.merge_strategy`:

| Strategy | Behavior |
|----------|----------|
| `concat` | Default — concatenate agent summaries |
| `last_wins` | Last successful node summary |
| `structured_json` | JSON payload with per-agent status |

**Future:** citation-preserving merge, LLM synthesis, conflict-aware HITL — IDEAL; not required for harness MVP.

## 51.4 Acceptance coverage

- Sequential multi-agent: `test_acceptance_02_sequential_multi_agent`
- Parallel multi-agent: `test_acceptance_03_parallel_multi_agent`
- Parallel cap: `test_graph_executor_parallel_cap.py`

---

# 52. Resilience in Orchestration

Orchestration resilience spans **three retry layers**, **checkpoints**, **alternate agents**, and **partial completion**. Full matrix: [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §14.

## 52.1 Three retry layers (do not conflate)

| Layer | Component | Scope | Default |
|-------|-----------|-------|---------|
| **A — Graph node** | `RetryEngine` | Same node; may switch `agent_id` | `max_retries` per factory profile |
| **B — UAEP / run** | `RuntimeEngine`, `AgentDecision.RETRY` | Inside one graph node | Per host `max_run_retries` |
| **C — Whole run** | `RetryCoordinator` | Re-execute full graph | `max_run_retries=0` (opt-in) |

**Failover (agent level):** closest harness primitive is **alternate agent** on node retry (Layer A) — not active-active duplicate nodes.

**Redundancy (infra level):** multiple Nexus host **replicas** — [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md); not duplicate graph nodes by default.

## 52.2 Long-running and recovery

§26 requirements implemented via:

| Mechanism | Module / event |
|-----------|----------------|
| Checkpoint store | `SQLiteTaskCheckpointStore`, long-running profile |
| Resume mid-UAEP | acceptance `05b_mid_step_uaep_resume` |
| Skip completed nodes | `GraphExecutor` + checkpoint bridge |
| Cancel | `CancellationCoordinator` → `CANCELLED` |
| Partial success | `PARTIALLY_COMPLETED` when policy allows |

## 52.3 Cross-cutting reliability (Tier-0 / ops)

Not orchestration scheduling — but required for resilient orchestration at scale:

| Mechanism | Domain |
|-----------|--------|
| Side-effect idempotency | W-OPS.1 / `IdempotentToolInvoker` |
| Integration circuit breaker | W-OPS.2 |
| HITL escalation | [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) |

## 52.4 Validation and evaluator resilience

| Gate | When |
|------|------|
| `NexusValidationEngine` | After each graph node |
| CVL partial / final verify | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |
| Evaluator-loop | Revise until budget exhausted |

---

# 53. Specialization and Agent Collaboration

## 53.1 Capability-based specialization

```text
Task.capability  →  TaskClassifier  →  CAPABILITY_ROUTED | MULTI_AGENT
                         ↓
              AgentRegistry.find_by_capability()
                         ↓
              PlanStep.agent_id assignment
```

**Multi-agent same capability:** sequential steps for all matching agents (`MULTI_AGENT`); order from `OrchestrationProfile.multi_agent_order` (FLOW-17).

Planning depth: [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) §9–§10.

## 53.2 Three collaboration mechanisms

| Mechanism | Declared | Runtime effect | Context |
|-----------|----------|----------------|---------|
| `DEPENDS_ON` | `graph_spec` / plan | Separate node; topological order | `ContextManager` |
| `DELEGATES_TO` | `graph_spec` | Child node + `DelegationSpec` on child | Isolated delegation namespace |
| `AgentDecision.HANDOFF` | Runtime | `HandoffCoordinator` inserts node | Handoff payload in shared context |

**Semantic detail:** [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §13.

## 53.3 Shared context rules

- Cross-agent data via `SharedTaskContext` / artifacts — **never** direct agent imports.
- Nexus passes **bounded** context per node (`ContextManager`).
- Delegation budgets: optional `budget_envelope` on `SubtaskContract` (FLOW-15).

## 53.4 Authoring

- Declarative: `AgentGraph` builder — `applications/contracts/graph_builder.py`
- Profile: `OrchestrationProfile` on `ApplicationEnvironmentProfile`
- Guide: [`AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md)

---

# 54. Maturity and Gap Register

| Area | Score (L0–L4) | Canon section | Notes |
|------|---------------|---------------|-------|
| Nexus loop / intake | L3–L4 | §9–§10, §48 | Done |
| Sequential / parallel execution | L3 | §25, §51 | Runtime in FLOW §9 |
| Coordination pattern catalog | L3 | §50 | Code + FLOW §27; ORCH canon (this update) |
| Declarative graph + delegation | L3–L4 | §50, §53 | ORCH-2, ADR-FLOW-001 |
| Merge policies | L3 | §51.3 | FLOW-7 Done |
| Backpressure | L3 | §51.2 | FLOW-13 Done |
| Retry / alternate agent | L3 | §52.1 | FLOW §14 |
| Checkpoint / long-running | L3 partial | §26, §52.2 | Scheduler optional |
| Swarm pattern runtime | L2 | §50.1 | Catalog Done; depth ORCH-5.1 |
| Active-active node redundancy | L0 | §52.1 | Not planned — use retry + ECP replicas |
| Infra elastic scale | L1 | cross-ref ECP | Phase ECP-DEPTH |
| Product multi-agent demos | L2 deferred | §42.43 | Phase K / FLOW-8 |

**Audit alignment:** AUDIT_MAP §9 (orchestration/graph) · §10 (subagents/multi-agent) — strategy rows consolidated in §50–§53.

**Plan backlog:** [Phase ORCH-STRAT](plan/ORCHESTRATION.md) (docs Done) · [Phase ORCH-5](plan/ORCHESTRATION.md) (runtime gaps).

---

## Related documents

| Document | Relationship |
|----------|--------------|
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | Runtime narrative §4–§18, §27 |
| [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | Plan topology (dimension B) |
| [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) | Replica/worker scale (dimension A infra) |
| [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) | Evaluator-loop, verify nodes |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.43 | Reference collaboration flow |
| [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §9–§10 | Audit procedure |

---

*End of Orchestration architecture canon (execution strategies §50–§54).*

---
