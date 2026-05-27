# INTERGRAX_RUNTIME_ARCHITECTURE.md

Status: Canonical architecture and implementation guide  
Audience: Humans, LLMs, Cursor AI agents, implementation agents, future maintainers  
Purpose: Define the Intergrax runtime architecture, implementation rules, agent model, orchestration model, adapter model, experimentation model and forbidden patterns.

---

# 1. Purpose Of This Document

This document is the primary architecture and implementation specification for the Intergrax runtime.

This document is intentionally written to be readable by:

- humans
- LLMs
- GPT-like models
- Claude-like models
- Gemini-like models
- Cursor AI agents
- autonomous coding agents
- future implementation agents

This document MUST be treated as the canonical source of truth for implementing Intergrax.

When an implementation decision is unclear, the implementation agent MUST prefer the principles, boundaries and contracts defined in this document.

This document is NOT a marketing document.

This document is NOT a product roadmap.

This document is an architectural and implementation guide.

---

# 2. Executive Summary

Intergrax is an AI Operating System / Agent Runtime / Harness AI environment.

The current goal is NOT to build a finished SaaS product.

The current goal is to build an internal agent experimentation laboratory where new agentic capabilities can be created, tested, observed, validated, improved or discarded quickly.

The ideal workflow is:

```text
new idea
    -> define agent capability
    -> implement agent contract
    -> register agent in Nexus
    -> connect required adapters/tools
    -> run experiment
    -> observe traces, cost, quality and failures
    -> validate or reject hypothesis
```

Intergrax should make it easy to test ideas such as:

- Problem Radar Agent
- Customer Vendor / Partner Discovery Agent
- Legal Agent
- UX Agent
- PM Agent
- Research Agent
- Onboarding Agent
- Sales Analysis Agent
- Business Process Agent

The core asset is not any single agent.

The core asset is the runtime that allows agents to be created and tested quickly.

---

# 3. What Intergrax Is

Intergrax IS:

- an AI Operating System
- an Agent Runtime
- a Harness AI Environment
- an Orchestration Runtime
- an Agent Experimentation Laboratory
- a Capability Execution Platform
- a runtime for testing business and technical agent hypotheses
- a system for integrating agentic work with real organizational tools

Intergrax is designed to answer this question:

> Can we rapidly create, run and evaluate new AI agents without rebuilding infrastructure every time?

---

# 4. What Intergrax Is Not

Intergrax is NOT:

- a chatbot
- a simple LLM wrapper
- a prompt collection
- a single agent
- a group chat between agents
- a frontend-heavy SaaS product
- a workflow builder at this stage
- a marketplace at this stage
- a clone of NotebookLM
- a direct competitor to Cursor AI
- a direct competitor to Viktor
- a product-first startup at this stage

Intergrax should learn from Cursor AI, Viktor, NotebookLM and modern agent runtimes, but the current goal is to build a controlled internal experimentation environment.

---

# 5. Core Architectural Thesis

Modern AI systems are moving away from isolated chatbots and toward runtime environments for intelligent work.

The strongest systems are not only models.

The strongest systems are environments that provide:

- orchestration
- execution lifecycle
- tool access
- memory boundaries
- task state
- integrations
- observability
- retries
- validation
- sandboxing
- human-in-the-loop controls
- agent registration
- capability composition

Intergrax follows this direction.

The main thesis is:

> The future value is not in building one agent. The value is in building the runtime that allows many agents to be built, tested and orchestrated quickly.

---

# 6. High Level Architecture

Intergrax consists of three major layers.

```text
+--------------------------------------------------------------+
|                         LAYER 3                              |
|                         AGENTS                               |
|--------------------------------------------------------------|
| ProblemRadarAgent                                             |
| VendorDiscoveryAgent                                          |
| LegalAgent                                                    |
| ResearchAgent                                                 |
| UXAgent                                                       |
| PMAgent                                                       |
| OnboardingAgent                                               |
| SalesAnalysisAgent                                            |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                         LAYER 2                              |
|                      NEXUS RUNTIME                           |
|--------------------------------------------------------------|
| Global Reasoning Loop                                         |
| Task Lifecycle Manager                                        |
| Planning Engine                                               |
| Agent Router                                                  |
| Agent Orchestrator                                            |
| Execution Graph                                               |
| Context Manager                                               |
| State Manager                                                 |
| Memory Coordinator                                            |
| Tool Runtime                                                  |
| Adapter Gateway                                               |
| Validation Engine                                             |
| Retry Engine                                                  |
| Sandbox Manager                                               |
| Shadow Workspace Manager                                      |
| Observability / Trace System                                  |
| Human Approval Manager                                        |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                         LAYER 1                              |
|                    COMPONENTS / ADAPTERS                     |
|--------------------------------------------------------------|
| LLM Providers                                                 |
| PostgreSQL                                                    |
| Redis                                                         |
| Queue System                                                  |
| Vector Store                                                  |
| File Storage                                                  |
| Slack Adapter                                                 |
| Teams Adapter                                                 |
| Email Adapter                                                 |
| Browser Adapter                                               |
| Web Search Adapter                                            |
| Git Adapter                                                   |
| CRM / ERP Adapters                                            |
| Logging                                                       |
| Sandbox Executor                                              |
| UI Renderer                                                   |
+--------------------------------------------------------------+
```

---

# 7. Layer Responsibility Summary

## 7.1 Layer 1: Components / Adapters

Layer 1 contains reusable technical integrations.

Examples:

- database adapters
- cache adapters
- message queue adapters
- Slack adapter
- Teams adapter
- email adapter
- browser automation adapter
- file system adapter
- vector store adapter
- LLM provider adapter
- sandbox adapter
- logging adapter

Layer 1 MUST NOT contain orchestration logic.

Layer 1 MUST NOT contain business-specific agent logic.

Layer 1 exposes capabilities to Nexus and agents through stable interfaces.

---

## 7.2 Layer 2: Nexus Runtime

Nexus is the central runtime and orchestration layer.

Nexus is the AI operating layer.

Nexus owns:

- global task understanding
- routing
- planning
- task decomposition
- agent selection
- execution graph
- state transitions
- lifecycle management
- retry strategy
- validation strategy
- context distribution
- tool access policy
- adapter access policy
- human approval flow
- observability
- final response construction

Nexus MUST remain domain-agnostic.

Nexus MUST NOT become a Legal Agent, Vendor Agent or Problem Radar Agent.

---

## 7.3 Layer 3: Agents

Agents are bounded capability modules.

Agents own domain-specific execution.

Examples:

- ProblemRadarAgent searches and clusters user pains from sources such as Reddit, Hacker News and other public sources.
- VendorDiscoveryAgent finds, classifies and evaluates companies for a client need.
- LegalAgent analyzes legal documents according to defined legal-review rules.
- OnboardingAgent supports new employees through a structured onboarding process.

Agents MUST implement stable contracts.

Agents MAY have their own local reasoning loop.

Agents MUST NOT own global orchestration.

---

# 8. Core Design Principles

## 8.1 Runtime First

The runtime is more important than any single agent.

Agents are replaceable.

The runtime is the long-term asset.

Implementation rule:

> Do not optimize the architecture around one agent. Optimize around the ability to create many agents quickly.

---

## 8.2 Experimentation First

Intergrax is currently a laboratory.

It should optimize for:

- fast agent creation
- fast hypothesis testing
- clear observability
- low setup cost
- easy deletion of failed experiments
- simple integration with existing tools

It should NOT currently optimize for:

- enterprise UI complexity
- marketplace features
- billing
- advanced tenant management
- unnecessary abstractions
- premature distributed complexity

---

## 8.3 Nexus Owns Global Reasoning

Nexus owns the global reasoning loop.

Nexus decides:

- what the user wants
- which agents are needed
- whether the task is simple or complex
- whether execution is sequential or parallel
- when to retry
- when to ask a human
- when to stop
- how to compose the final answer

---

## 8.4 Agents Own Local Execution

Agents own local domain execution.

Agents decide:

- how to perform their bounded task
- which local tools to use
- how to improve their local result
- how to validate their local output

Agents do not decide the global workflow unless explicitly delegated by Nexus for a bounded subtask.

---

## 8.5 Integrations Are Adapters

Slack, Teams, email, databases, browser automation and other external tools are adapters.

They are not agents.

They are not Nexus.

They are infrastructure capabilities exposed to the runtime.

---

## 8.6 UI Is Optional

Intergrax is not frontend-first.

The runtime must work without a heavy UI.

Slack, Teams, chat, CLI or a lightweight internal dashboard can be valid interaction surfaces.

UI must not define the architecture.

---

## 8.7 Observability Is Mandatory

Every meaningful step must be observable.

An agent experiment without traces is not useful.

The system must show:

- what was requested
- what Nexus understood
- what plan was created
- which agents were selected
- which tools were used
- what data was processed
- what failed
- what was retried
- what the result was
- why the system stopped

---

# 9. Dual Loop Architecture

Intergrax uses a dual-loop architecture.

There are two types of loops:

1. Global Nexus Loop
2. Local Agent Loop

This is a required architectural decision.

---

# 9.1 Global Nexus Loop

The Nexus loop is mandatory.

The Nexus loop controls global execution.

Responsibilities:

- receive user task
- classify task
- determine complexity
- create or update plan
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

Agents MAY have local loops.

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

# 11. Agent Responsibilities

Agents are specialized execution modules.

An agent is responsible for:

- understanding its local task
- using allowed tools
- executing domain-specific logic
- producing structured output
- validating local output
- reporting uncertainty
- reporting failures
- returning artifacts to Nexus

An agent is NOT responsible for:

- global orchestration
- global task lifecycle
- global retries
- user communication outside the contract
- creating unrelated agents
- bypassing Nexus
- owning cross-agent memory

---

# 12. Agent Contract

Every agent MUST implement a clear contract.

The contract should be easy for humans and LLMs to understand.

Minimum required fields:

```text
AgentContract:
    id
    name
    description
    version
    capabilities
    input_schema
    output_schema
    allowed_tools
    required_adapters
    execution_mode
    max_steps
    max_duration
    max_cost
    risk_level
    validation_rules
    failure_modes
```

---

# 13. Suggested Agent Interface

This is conceptual pseudocode, not a required programming language implementation.

```text
interface Agent:

    get_contract() -> AgentContract

    can_handle(task_context) -> CapabilityMatchResult

    execute(agent_input, execution_context) -> AgentExecutionResult

    validate(agent_output, execution_context) -> ValidationResult
```

Agent implementations should be simple.

The goal is to let developers focus on domain logic, not infrastructure.

---

# 14. Agent Execution Result

Every agent should return a structured result.

Recommended structure:

```text
AgentExecutionResult:
    agent_id
    run_id
    status
    summary
    artifacts
    structured_data
    evidence
    confidence
    warnings
    errors
    used_tools
    cost
    duration
    next_recommendations
```

The result must be inspectable by Nexus and by humans.

---

# 15. Agent Registry

Nexus discovers agents through the Agent Registry.

The registry stores:

- agent id
- name
- description
- version
- capabilities
- required adapters
- allowed tools
- execution modes
- cost profile
- risk profile
- status

Nexus MUST use the registry for agent selection.

Agents MUST NOT be hardcoded into Nexus logic unless explicitly needed for a minimal prototype.

Even in prototypes, hardcoded agents should be treated as temporary.

---

# 16. Capability Model

A capability describes what an agent can do.

Examples:

```text
capability: vendor.discovery
capability: vendor.scoring
capability: legal.contract_review
capability: research.web_search
capability: problem_radar.source_monitoring
capability: problem_radar.clustering
capability: onboarding.daily_guidance
```

Nexus should route tasks to capabilities, not only to specific class names.

This allows agents to be replaced later.

---

# 17. Adapter Architecture

Adapters are reusable integrations with external systems.

Examples:

- SlackAdapter
- TeamsAdapter
- EmailAdapter
- PostgreSqlAdapter
- RedisAdapter
- BrowserAdapter
- WebSearchAdapter
- FileSystemAdapter
- VectorStoreAdapter
- LlmProviderAdapter
- SandboxAdapter

Adapters MUST be treated like infrastructure components.

Adapters MUST NOT contain business workflow logic.

Adapters MUST NOT decide which agent to run.

Adapters expose operations.

Nexus or agents call those operations through explicit permissions.

---

# 18. Slack / Teams / Communication Integration Philosophy

Intergrax should support Slack and Teams as interaction surfaces.

This follows the Viktor-like idea where an AI worker can live inside organizational communication tools.

Slack and Teams should be implemented as adapters.

They may provide:

- task intake
- notifications
- approval requests
- progress updates
- final responses
- interactive buttons
- user context
- channel context

They should NOT own the runtime.

Correct model:

```text
Slack message
    -> SlackAdapter
    -> normalized Task
    -> Nexus Runtime
    -> Agent execution
    -> Nexus final result
    -> SlackAdapter sends response
```

Incorrect model:

```text
Slack bot contains orchestration logic
Slack bot directly manages agents
Slack bot stores global task state
```

---

# 19. UI / UX Testing Requirement

Even though Intergrax is not frontend-heavy, agents must be testable and observable.

The system should support minimal UI/UX surfaces for:

- viewing task list
- viewing task status
- viewing execution trace
- viewing agent outputs
- viewing tool calls
- viewing errors
- viewing artifacts
- approving or rejecting steps
- re-running tasks
- comparing outputs

This may be implemented as:

- lightweight dashboard
- CLI
- chat interface
- Slack/Teams messages
- internal debug panel

The UI is for observability and experimentation, not product polish.

---

# 20. Shadow Workspace Model

A Shadow Workspace is an isolated temporary workspace used to perform work without directly modifying the main environment.

Inspired by Cursor-like execution environments.

Shadow Workspaces may be used for:

- code experiments
- document analysis
- temporary data transformations
- simulated business workflows
- vendor research sessions
- legal document review sessions
- onboarding simulations

A Shadow Workspace should provide:

- isolation
- temporary storage
- reproducibility
- rollback safety
- inspectable artifacts
- cleanup

---

# 21. Sandbox Model

A sandbox is a controlled execution environment.

Use sandboxes for:

- code execution
- browser automation
- file manipulation
- risky tool use
- external data extraction
- generated script execution

Sandbox execution should be:

- isolated
- observable
- permission-controlled
- interruptible
- disposable
- reproducible when possible

---

# 22. Tool Runtime

Tools are callable operations exposed to Nexus and agents.

Examples:

- search web
- read file
- write file
- query database
- send Slack message
- create document
- call LLM
- run browser action
- execute script in sandbox

Tools must have:

- name
- description
- input schema
- output schema
- risk level
- permission requirement
- timeout
- retry policy

Tools should be registered in a Tool Registry.

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

# 27. Memory Model

Memory must be explicit and bounded.

Types of memory:

1. Task Memory
2. Agent Local Memory
3. User / Organization Memory
4. Long-Term Knowledge Memory
5. Execution Trace Memory

Rules:

- Nexus owns global task memory.
- Agents may own local temporary memory.
- Long-term memory must be explicitly written.
- Agents must not silently mutate global memory.
- Sensitive memory writes should require policy checks.

---

# 28. Context Management

Context is expensive and dangerous when uncontrolled.

Nexus must control what context each agent receives.

Rules:

- pass only relevant context
- avoid dumping entire history into agents
- use summaries when needed
- separate task context from user memory
- separate evidence from interpretation
- preserve provenance

---

# 29. Validation Model

Validation is mandatory.

Validation should not rely only on LLM confidence.

Possible validation types:

- schema validation
- rule-based validation
- data completeness validation
- source citation validation
- secondary model review
- separate validator agent
- human review
- executable tests
- consistency checks

Validation should be defined before or during planning.

For high-risk tasks, Nexus should create a validation contract before execution.

---

# 30. Failure Model

Failures are expected.

The system must treat failure as normal.

Failure types:

- agent failure
- tool failure
- adapter failure
- timeout
- invalid output
- missing data
- low confidence
- unsafe action
- human rejection
- incomplete result

Failure handling options:

- retry same step
- retry with different agent
- ask human
- degrade gracefully
- return partial result
- stop execution
- mark as failed

---

# 31. Retry Policy

Retries must be controlled.

Every retry should have:

- reason
- retry count
- changed strategy if possible
- stop condition

Do not retry endlessly.

Retries should be visible in traces.

---

# 32. Human In The Loop

Human approval may be required for:

- sending external messages
- modifying external systems
- deleting data
- financial actions
- legal conclusions
- risky automation
- uncertain results

Nexus manages human approval.

Agents may request approval, but Nexus controls the approval flow.

---

# 33. Observability And Tracing

Every execution should create a trace.

Trace should include:

- task id
- run id
- user input
- normalized task
- plan
- reasoning summaries
- agent selections
- agent inputs
- agent outputs
- tool calls
- adapter calls
- errors
- retries
- validation results
- final result
- timestamps
- cost information if available

Observability exists for:

- debugging
- evaluation
- hypothesis validation
- cost control
- safety
- future improvement

---

# 34. Evaluation Model

Since Intergrax is an experimentation laboratory, every agent should be evaluated.

Evaluation criteria may include:

- task success
- output quality
- factuality
- completeness
- cost
- latency
- usefulness
- repeatability
- user satisfaction
- failure frequency
- business value

Agents should not be considered successful only because they produced text.

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

# 36. Example: Problem Radar Agent

Purpose:

Identify repeated user problems, complaints and pain signals from public sources.

Possible sources:

- Hacker News
- Reddit
- forums
- review sites
- product communities
- social platforms

Possible steps:

```text
1. Collect posts/comments
2. Filter noise
3. Extract pain statements
4. Group similar pains
5. Cluster by market/problem
6. Score frequency and intensity
7. Apply problem quality filters
8. Generate opportunity report
```

Expected output:

```text
ProblemRadarOutput:
    clusters
    representative_quotes
    source_links
    frequency_estimate
    intensity_score
    affected_user_group
    possible_product_ideas
    mom_test_risk_notes
    confidence
```

This agent should be implemented as a capability module and executed through Nexus.

---

# 37. Example: Vendor Discovery Agent

Purpose:

Find, categorize, evaluate and recommend companies for a given business need.

Possible use cases:

- find subcontractors
- find business partners
- find potential customers
- audit vendors
- compare companies

Possible steps:

```text
1. Understand customer need
2. Define search criteria
3. Search company sources
4. Enrich company profiles
5. Categorize companies
6. Score fit
7. Detect risks
8. Produce recommendation
```

Expected output:

```text
VendorDiscoveryOutput:
    companies
    categories
    fit_scores
    strengths
    risks
    evidence
    source_links
    recommendation
    confidence
```

---

# 38. Example: Organization Worker Agent

Purpose:

Act as a virtual worker inside an organization through Slack, Teams or other communication tools.

Possible tasks:

- prepare monthly sales analysis
- onboard employees
- coordinate document review
- summarize project updates
- monitor operational signals
- prepare vendor reports

Architecture:

```text
User message in Slack
    -> SlackAdapter
    -> Nexus
    -> task classification
    -> agent selection
    -> execution
    -> progress updates
    -> final response in Slack
```

The Slack integration is only an interaction surface.

The runtime remains in Nexus.

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

# 42. Anti-Patterns

The following patterns are forbidden or strongly discouraged.

## 42.1 Fat Agent Anti-Pattern

Do not create agents that contain:

- routing
- global orchestration
- global memory
- scheduler
- UI logic
- platform state

---

## 42.2 Fat Nexus Anti-Pattern

Do not put domain-specific workflows directly inside Nexus.

Nexus should orchestrate, not become the agent.

---

## 42.3 UI-Driven Architecture Anti-Pattern

Do not design the runtime around a frontend screen.

The runtime must work from API, Slack, Teams, CLI or chat.

---

## 42.4 Prompt-Only Architecture Anti-Pattern

Do not treat prompts as the architecture.

Prompts are part of agents and reasoning, but the runtime must have real execution structures.

---

## 42.5 Unobservable Execution Anti-Pattern

Do not execute important steps without traces.

If it cannot be inspected, it cannot be trusted.

---

## 42.6 Product Too Early Anti-Pattern

Do not build billing, marketplace, advanced UI or enterprise features before validating the runtime.

---

# 43. Decision Records

## 43.1 Decision: Nexus Has Global Loop

Decision:

Nexus owns the global reasoning and execution loop.

Reason:

Global coordination must be centralized to avoid chaotic autonomous agents.

---

## 43.2 Decision: Agents May Have Local Loops

Decision:

Agents may contain bounded local execution loops.

Reason:

Complex domain tasks require local multi-step execution.

Constraint:

Agent loops must be bounded by contracts, limits and validation rules.

---

## 43.3 Decision: Slack And Teams Are Adapters

Decision:

Slack and Teams are Layer 1 adapters / interaction surfaces.

Reason:

Communication tools should not own orchestration.

---

## 43.4 Decision: Intergrax Is A Laboratory First

Decision:

Intergrax is currently an internal experimentation runtime, not a full SaaS product.

Reason:

The current strategic goal is rapid hypothesis validation.

---

## 43.5 Decision: Agents Are Capabilities

Decision:

Agents are capability modules, not independent products.

Reason:

This allows rapid creation, replacement and composition.

---

# 44. Checklist For New Agent Implementation

Before implementing a new agent, answer:

```text
1. What hypothesis does this agent test?
2. What capability does it provide?
3. What input does it require?
4. What structured output does it produce?
5. What tools/adapters does it need?
6. What is the validation rule?
7. What are failure modes?
8. What is the maximum acceptable cost/time?
9. How will success be evaluated?
10. How will Nexus route tasks to this agent?
```

If these questions cannot be answered, do not implement the agent yet.

---

# 45. Checklist For New Adapter Implementation

Before implementing a new adapter, answer:

```text
1. What external system does it connect to?
2. What operations does it expose?
3. What permissions are required?
4. Is it read-only or write-capable?
5. What are risk levels?
6. What errors can happen?
7. What timeout/retry policy is needed?
8. What data should be logged?
9. What data must be protected?
10. Which agents or runtime components may use it?
```

Adapters should be generic and reusable.

---

# 46. Checklist For Nexus Changes

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
```

If the change is domain-specific, it probably belongs in an agent.

---

# 47. Naming Guidance

Recommended naming:

- NexusRuntime
- AgentContract
- AgentRegistry
- ToolRegistry
- AdapterRegistry
- ExecutionContext
- TaskContext
- TaskState
- ExecutionGraph
- ExecutionNode
- TraceEvent
- AgentExecutionResult
- ValidationResult
- ShadowWorkspace
- SandboxSession

Names should be explicit and boring.

Avoid clever names that make the architecture harder for LLMs and humans to understand.

---

# 48. LLM Readability Rules For This Project

Because Cursor AI and LLM agents will read this project, code and documentation should follow these rules:

- use explicit names
- avoid hidden magic
- avoid ambiguous abstractions
- prefer small files with clear responsibility
- document public contracts
- include examples
- include state transitions
- include schemas
- include error cases
- avoid overly clever metaprogramming
- keep architecture boundaries visible

LLMs perform better when responsibilities are explicit.

---

# 49. Future Evolution

Intergrax may later evolve into:

- enterprise AI operating system
- organization-wide agent platform
- agent marketplace
- visual workflow builder
- autonomous business process runtime
- multi-tenant SaaS
- internal company AI worker ecosystem

But these are future possibilities.

Current priority:

> Build a reliable minimal runtime for fast agent experimentation.

---

# 50. Final Canonical Statement

Intergrax is an AI Operating System and Harness Runtime for creating, orchestrating and validating agentic capabilities.

The current purpose of Intergrax is to serve as an internal laboratory for rapid experimentation with agentic business functionality.

Nexus is the global orchestration runtime.

Agents are bounded capability modules.

Adapters are reusable integrations.

The architecture must optimize for rapid hypothesis validation, observability, modularity and clean separation of responsibilities.

The system should make it possible to quickly implement a new agent, run it through Nexus, observe results, evaluate business value and decide whether the capability deserves further investment.

This is the core architectural direction of Intergrax.

