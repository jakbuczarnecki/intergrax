## Agent Factory — Architectural Principles

Intergrax is designed as an **Agent Factory**, not as a single general-purpose conversational system.

The primary product of Intergrax is **specialized, production-grade agents**, each dedicated to solving a well-defined class of problems with high precision, predictability, and domain awareness.

The framework itself (engine, runtime, tooling) exists to **enable, constrain, and safely operate** these agents — not to define their behavior.

---

## Core Principle

**Agent behavior is defined by the pipeline implementation.**

A pipeline represents a complete, self-contained behavioral program for an agent:
- how the agent reasons,
- how it decides what to ask,
- how it gathers data,
- how it uses tools,
- how it iterates or stops,
- how it produces a final answer.

The runtime engine does not impose reasoning strategies or execution models.  
It only enforces global policies and contracts.

---

## Separation of Responsibilities

### Engine (Runtime)

The engine is a generic orchestration and enforcement layer.

Responsibilities:
- invoke the configured pipeline,
- enforce budgets (tokens, time, tool calls),
- manage retries, timeouts, and aborts,
- handle tracing, diagnostics, and observability,
- enforce security, isolation, and guardrails,
- validate and finalize the returned `Answer`.

The engine treats pipelines as opaque executables that must satisfy a strict output contract.

---

### Pipeline (Agent Behavior)

A pipeline is the **single source of truth for agent behavior**.

Responsibilities:
- define the reasoning strategy (single-step, planned execution, iterative decision-making),
- construct and orchestrate prompts using the prompt registry,
- decide when to ask clarifying questions,
- decide when and how to use tools (RAG, web, external APIs, calculators),
- control iteration, stopping conditions, and success criteria,
- transform intermediate results into a final `Answer`.

A pipeline may implement any internal logic it requires, as long as it returns a valid `Answer`.

---

## Execution Model

Intergrax uses an **inversion-of-control execution model**:

1. A pipeline instance is provided via configuration.
2. The engine invokes `pipeline.run(...)`.
3. The pipeline executes its internal logic (single step, loops, phases, or custom workflows).
4. The pipeline returns an `Answer`.
5. The engine validates, logs, tracks cost, and delivers the response to the user.

This model allows pipelines to:
- be arbitrarily complex,
- evolve independently,
- be tested in isolation,
- and be versioned as product artifacts.

---

## Agent Specialization

Intergrax does not aim to create a single “universal” agent.

Each agent is:
- highly specialized,
- designed around a specific domain or task,
- implemented as a dedicated pipeline.

Examples:
- `MathTeacherPipeline`
- `FinancialCompanyPipeline`
- `SeniorHRPipeline`

Specialization is achieved through:
- pipeline logic,
- prompt packs per execution phase,
- tool scopes and permissions,
- memory and data source configuration,
- budget and guardrail policies.

---

## Prompting Strategy

Prompts are not embedded directly in pipeline code.

Instead:
- all prompts are stored in the YAML prompt registry,
- prompts are versioned, pinned, and auditable,
- pipelines assemble prompts dynamically per execution phase.

Typical phases include:
- routing,
- planning,
- step decision,
- tool usage,
- verification / critique,
- final response synthesis.

---

## Product-Oriented Design

Agents are treated as **first-class product units**:
- independently versioned,
- independently tested,
- independently deployed or rolled out.

The Agent Factory enables:
- building many agents on a shared runtime,
- strict separation between platform and product logic,
- predictable cost and behavior per agent,
- long-term maintainability and scalability.

---

## Design Goals

The Agent Factory architecture is designed to ensure:
- strong behavioral determinism under constraints,
- high observability and auditability,
- safe and enforceable use of tools and data,
- minimal coupling between runtime and agent logic,
- sustainable evolution of agent behavior over time.
