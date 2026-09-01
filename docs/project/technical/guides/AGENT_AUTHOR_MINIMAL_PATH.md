# Agent Author Minimal Path

**Status:** Normative authoring guide (Tier-2 agents)  
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Canon:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md)

---

## Purpose

This guide defines the minimal safe path for implementing a Tier-2 agent in Intergrax.  
It is intended for human developers, Cursor, Codex and other coding agents.  
It does not replace [`docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md).
It is the practical authoring shortcut derived from that architecture and from [`docs/project/technical/guides/SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md).

For scaffold → register → run → evaluate workflows, see [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md). Use **this file first** when you need the smallest correct mental model.

---

## Minimal authoring model

An agent author is responsible for:

1. Defining the agent contract.
2. Defining the agent capability/capabilities.
3. Implementing domain-specific step behavior.
4. Returning typed decisions/results.
5. Using allowed tools through the harness.
6. Relying on runtime-provided context, memory, policy, tracing and validation.

An agent author is not responsible for:

1. Creating a runtime.
2. Creating a Nexus loop.
3. Creating schedulers, queues, HTTP servers or workers.
4. Calling vendor SDKs directly.
5. Calling tool handlers or integrations directly.
6. Managing global retries, HITL or orchestration.
7. Creating private observability or memory systems.
8. Hand-assembling production prompts from unbounded history.

---

## Minimal files and artifacts

A minimal Tier-2 agent in this repository typically includes:

| Artifact | Role |
|----------|------|
| Agent implementation class/module | Domain logic; step handler (`on_next_step` or cognitive pattern base) |
| `AgentContract` (or equivalent contract builder) | Identity, capabilities, skills, risk, lifecycle, pattern metadata |
| Capability declaration | Tokens Nexus uses for routing |
| Optional prompt references | Via Prompt Registry or approved prompt mechanism - not inline production strings |
| Optional skill/tool declarations | On contract and/or skill resolver |
| Tests or smoke scenario | When the agent family already has a test pattern in `agents/<name>/tests` |
| Documentation entry | Only when the agent is part of a reusable family (e.g. `README.md`, `ARCHITECTURE.md`) |

**Patterns (examples - do not copy wholesale):**

- Contract + capabilities: `agents/local_indexer/contract.py`, `agents/local_indexer/capabilities.py`
- Agent class: `agents/local_indexer/local_indexer_agent.py`
- Authoring bases: `intergrax/agents/authoring` (`IntergraxAgent`, `CognitiveAgent`, `StepOutcome`)
- Contract type: `intergrax/contracts/agent_contract_meta.py` (`AgentContract`)

Do not invent file names beyond what the repository and [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §12–§16 already establish.

---

## Minimal execution flow

```text
1. Tier-3 application receives a task.
2. Application routes the task into UnifiedTaskRunner.run_task().
3. Nexus/Graph runtime selects the agent.
4. AgentEngine runs the agent under policy, budget, context and observability controls.
5. The agent receives runtime-provided step context.
6. The agent decides the next domain action.
7. Tool calls go through ToolRuntime.
8. LLM calls go through LLMAdapter / approved model routing.
9. Context is assembled by ContextCompiler / ContextEngine.
10. Memory is accessed through MemoryView / approved memory services.
11. The agent returns typed output/decision.
12. Runtime validates, traces and finalizes or continues.
```

The agent author implements steps **6** and **11**; the harness owns the rest.

---

## What the agent implements

A minimal agent should implement only:

- identity / metadata required by the contract,
- declared capabilities,
- input/output schema expectations,
- domain-specific decision logic,
- bounded local step behavior,
- validation hooks only if domain-specific,
- tool intent or tool request through approved runtime APIs,
- structured final result.

The agent may **request**:

- a tool call,
- more context,
- a memory lookup,
- RAG retrieval,
- human input,
- termination/finalization,

but the **runtime decides** how these requests are executed.

Primary author surfaces (canon §13): session `run(AgentRunRequest)` and step `on_next_step()` returning `StepOutcome` factories - not a private orchestration loop.

---

## What the agent must not implement

A Tier-2 agent **MUST NOT**:

- instantiate or own `NexusLoop`,
- instantiate or own `UnifiedTaskRunner`,
- create its own execution runtime,
- create long-running background loops,
- create HTTP servers or external listeners,
- manage global graph topology,
- perform cross-agent scheduling,
- implement global retry policy,
- implement ad-hoc HITL approval flow,
- call integration adapters directly,
- call vendor LLM SDKs directly,
- call vector stores directly,
- bypass ToolRuntime,
- bypass PolicyEngine,
- bypass RuntimeEventBus / observability spine,
- bypass ContextCompiler for production LLM context,
- persist private long-term memory outside approved memory services,
- store secrets in traces, prompts, memory or events,
- return only unstructured text when typed output is required.

---

## Allowed extension points

Safe extension points for Tier-2 authors:

- contract metadata,
- capability definitions,
- domain prompts via Prompt Registry / approved prompt mechanism,
- `on_next_step` or equivalent approved step handler,
- domain validators,
- domain-specific rubrics for CVL,
- skill/tool declarations,
- agent-local configuration allowed by the contract,
- tests and smoke scenarios.

If an extension requires runtime lifecycle, scheduling, infrastructure, global orchestration, provider SDK usage or storage ownership, it is probably **not** an agent extension and must be moved to the appropriate tier (Tier-0 harness or Tier-3 application).

---

## Review checklist for Cursor

Before committing a new or modified agent, verify:

- [ ] Does the change stay inside Tier-2 responsibilities?
- [ ] Does the agent rely on Nexus/AgentEngine instead of creating its own loop?
- [ ] Are all side effects routed through ToolRuntime?
- [ ] Are all LLM calls routed through LLMAdapter or approved abstractions?
- [ ] Is context runtime-provided rather than built from unbounded history?
- [ ] Is LLM context assembled via ContextCompiler / ContextEngine or an approved equivalent ([`CONTEXT_ENGINEERING.md`](../../architecture/CONTEXT_ENGINEERING.md) §12)?
- [ ] Are outputs typed where required?
- [ ] Are retries/HITL/policy handled by runtime rather than ad-hoc agent code?
- [ ] Are observability events emitted through the harness spine?
- [ ] Are secrets excluded from prompts, traces, memory and events?
- [ ] Is the agent contract/capability declaration updated?
- [ ] Are tests/smoke checks updated if this agent family has a test pattern?

Extended pre-implementation questions: [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §45.

---

## Related documents

| Document | Why |
|----------|-----|
| [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) | Full contract, ACP, step loop, production gates |
| [`NEXUS_EXECUTION_FLOW.md`](../../architecture/NEXUS_EXECUTION_FLOW.md) | Nexus routing and graph execution |
| [`UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) | AgentEngine, policy, harness kernel |
| [`TOOLS.md`](../../architecture/TOOLS.md) | ToolRuntime and tool invocation |
| [`CONTEXT_ENGINEERING.md`](../../architecture/CONTEXT_ENGINEERING.md) | ContextCompiler / ContextEngine; Context Path Unification (§12) |
| [`MEMORY.md`](../../architecture/MEMORY.md) | MemoryView and approved memory services |
| [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) | Cross-layer MUST/MUST NOT |
| [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md) | Maturity vocabulary (A/I/P/E) |
