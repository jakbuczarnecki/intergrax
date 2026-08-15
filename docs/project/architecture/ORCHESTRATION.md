# Orchestration, Nexus, and Execution Graph

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)
**Plan (1:1):** [`plan/ORCHESTRATION.md`](../maintainers/plans/ORCHESTRATION.md)
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Audit layers:** 3, 9 · multi-agent patterns: audit layer 10 (cross-ref §50)  
**Audit instruction:** [`audit/ORCHESTRATION.md`](../maintainers/audit/ORCHESTRATION.md)
**Reasoning / planning canon:** [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) (audit layer 7)  
**Elastic capacity:** [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md#production-boundary) (infra capacity signals and scaling — **not** graph scheduling, agent topology or orchestration brain)  
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ORCHESTRATION canon).

- **Implement / audit default:** intake + NexusLoop + graph executor (§10–§26). §27+: [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/ORCHESTRATION.md`](../maintainers/plans/ORCHESTRATION.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/ORCHESTRATION.md`](../technical/guides/audit_slices/ORCHESTRATION.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Document roles (read order)

| Document | Role |
|----------|------|
| **This file (`ORCHESTRATION.md`)** | Orchestration **manifest** + strategy catalog (§50–§55) + **platform configuration canon** (§56) |
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | **Runtime narrative** — sequence diagrams, UC-*, edge cases, code paths |
| [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | Classification, planning, agent topology in plans |
| [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) | Author control plane (`OrchestrationProfile`, wiring) |

**Rule:** strategy **names and selection** live here; step-by-step runtime truth lives in **NEXUS_EXECUTION_FLOW**.
## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/ORCHESTRATION_production_gates.md`](satellites/ORCHESTRATION_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


## Table of contents (strategy & execution)

| § | Topic |
|---|--------|
| [§50](.#50-orchestration-strategies-catalog) | Coordination pattern catalog |
| [§51](.#51-parallelism-merge-and-backpressure) | Parallelism, merge, backpressure |
| [§52](.#52-resilience-in-orchestration) | Retry, checkpoint, failover, partial |
| [§53](.#53-specialization-and-agent-collaboration) | Capability routing, delegation, handoff |
| [§54](.#54-maturity-and-gap-register) | Maturity scorecard |
| [§55](.#55-interaction-posture--orchestration-matrix) | Posture × pattern quick matrix |
| [§56](.#56-platform-interaction--multi-agent-configuration-canon) | **Master configuration canon** — all cases, matrices, plan input |
| [§56.13](.#5613-orchestration-capability-tokens) | Orchestration tokens vs agent capabilities |
| [§57](.#57-synchronous-and-asynchronous-execution-postures) | Sync vs async dispatch, queue workers, agent contract |
| [§58](.#58-platform-runtime-capabilities-index) | Cross-cutting index: resilience, autonomy, MVP evolution |

**Authoring rule:** Tier-3 host design starts at **§56**; runtime step-by-step truth remains in **NEXUS_EXECUTION_FLOW**; posture/host wiring summary in **TIER3 §23**.

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
