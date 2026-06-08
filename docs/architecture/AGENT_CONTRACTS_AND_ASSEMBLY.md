# Agent Contracts, Registry, and Capability Model

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

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

All `execute()` implementations MUST delegate to `AgentEngine` and the Unified Agent Execution Protocol ([§42](UNIFIED_EXECUTION_RUNTIME.md).5). Agents MUST NOT implement private runtime lifecycles.

---


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


---

# 45. Checklist For New Agent Implementation

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
11. Which AgentSteps does the agent declare ([§42](UNIFIED_EXECUTION_RUNTIME.md).6)?
12. Which AgentDecision types can the agent emit ([§42](UNIFIED_EXECUTION_RUNTIME.md).7)?
13. Does the agent conform to UAEP via AgentEngine ([§42](UNIFIED_EXECUTION_RUNTIME.md).5)?
14. Are all tool calls routed through ToolRuntime ([§42](UNIFIED_EXECUTION_RUNTIME.md).12)?
15. Are forbidden runtime patterns avoided ([§42](UNIFIED_EXECUTION_RUNTIME.md).41)?
```

If these questions cannot be answered, do not implement the agent yet.

---

