# Intergrax — Agent Creation Guide

**The single canonical guide for creating, registering, running, and evaluating agents.**

This is the **only** step-by-step workflow document. Do not duplicate this process elsewhere.
Architecture canon: [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  
Implementation status: [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md)

**Audience:** human developers, GPT, Claude, Gemini, Cursor agents.

**Success metric:** from idea to first Nexus run in **under one hour**, with **zero changes** to `intergrax/runtime/`.

---

## Table of contents

1. [Mental model](#1-mental-model)
2. [End-to-end workflow](#2-end-to-end-workflow)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — Hypothesis and capability](#step-1--hypothesis-and-capability)
5. [Step 2 — Scaffold the agent](#step-2--scaffold-the-agent)
6. [Step 3 — Implement domain logic](#step-3--implement-domain-logic)
7. [Step 4 — Register the agent](#step-4--register-the-agent)
8. [Step 5 — Run the agent](#step-5--run-the-agent)
9. [Step 6 — Inspect traces and runtime events](#step-6--inspect-traces-and-runtime-events)
10. [Step 7 — Record experiment and decision](#step-7--record-experiment-and-decision)
11. [Step 8 — Test and gate](#step-8--test-and-gate)
12. [Step 9 — Sign-off before business agents](#step-9--sign-off-before-business-agents)
13. [Appendix A — Human-in-the-loop](#appendix-a--human-in-the-loop)
14. [Appendix B — Shadow workspace and sandbox](#appendix-b--shadow-workspace-and-sandbox)
15. [Appendix C — Multi-agent graphs](#appendix-c--multi-agent-graphs)
16. [Appendix D — Advanced execution paths](#appendix-d--advanced-execution-paths)
17. [Anti-patterns](#anti-patterns)
18. [Instructions for LLM coding agents](#instructions-for-llm-coding-agents)

---

## 1. Mental model

```text
Tier-0  intergrax/           Platform (LLM, storage, queues, logging, …)
Tier-1  intergrax/runtime/   Nexus — Agent Operating System
Tier-2  agents/              Reusable agent capabilities
Tier-3  applications/        Execution environments (wiring only)
```

```text
Nexus (Tier-1)          = Agent Operating System  (orchestration, lifecycle, trace, memory, HITL)
Agents (Tier-2)         = domain logic            (decisions, prompts, tools, workflows)
Applications (Tier-3)   = configuration           (which agents, routes, integrations)
```

When you create an agent you work **only** on Tier-2:

| You implement | Nexus owns (do not touch for one agent) |
|---------------|----------------------------------------|
| decisions, rules, workflows | orchestration |
| prompts, tools, outputs | lifecycle, tracing, memory |
| `AgentContract`, UAEP steps | checkpointing, retries, HITL, graphs |

**Registration rule:** a new agent integrates through `AgentRegistry.register()` — never by editing `NexusLoop`, `GraphExecutor`, or task lifecycle code.

---

## 2. End-to-end workflow

```text
idea
  → hypothesis
  → capability id
  → scaffold                    (python -m intergrax.scaffold new-agent …)
  → implement domain logic      (steps/, prompts/, contract.py)
  → register                    (pick context: test / script / lab / product app)
  → run                         (pytest / NexusLoop / lab HTTP)
  → inspect                     (debug API / CLI)
  → evaluate
  → decision: keep | improve | pause | delete
```

---

## 3. Prerequisites

- Repository root with `uv` / Python 3.12
- `agents/` and `applications/` on import path (configured in `pyproject.toml` → `pythonpath`)
- No external network required for smoke tests and gate

Verify platform health:

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

---

## Step 1 — Hypothesis and capability

Write one sentence:

> When **&lt;trigger&gt;**, agent **&lt;name&gt;** should **&lt;outcome&gt;** using **&lt;tools/data&gt;**.

Define a **capability id** — stable routing key used by Nexus classifier and `TaskContext`:

```text
documents.automation
vendor.discovery.basic
signoff.probe
```

Convention: lowercase, dot-separated namespace (`<domain>.<action>`).

---

## Step 2 — Scaffold the agent

```bash
python -m intergrax.scaffold new-agent document_automation \
    --capability documents.automation
```

Repeat `--capability` for multiple capabilities. Default if omitted: `<slug>.basic`.

**Generated layout:**

```text
agents/document_automation/
    document_automation_agent.py   # Agent class (UAEP entry point)
    contract.py                    # AgentContract builder
    capabilities.py                # capability id list
    steps/pipeline.py              # domain execution (start here)
    schemas/                       # Pydantic I/O models
    prompts/system.md              # prompt assets
    tests/test_document_automation_agent.py   # smoke test (includes registration)
    notebooks/01_document_automation_experiment.ipynb
    README.md
```

The scaffold is **UAEP-first**: every agent implements `get_steps` / `run_step` / `decide_after_step`.

**Important:** scaffold creates files only. It does **not** register the agent globally or in any application.

---

## Step 3 — Implement domain logic

| File | Responsibility |
|------|----------------|
| `capabilities.py` | Public capability ids |
| `contract.py` | `AgentContract` — id, description, tools, risk, max_steps |
| `steps/` | Business logic — prompts, rules, tool calls |
| `prompts/` | System/user prompt templates |
| `schemas/` | Request/response models |

**Reuse Tier-0** (LLM adapters, storage, queues, notifications). Do not duplicate platform infrastructure inside the agent folder.

**Tool access under UAEP:**

```python
from intergrax.contracts.tool_request import ToolRequest

response = await ctx.invoke_tool(
    ToolRequest(
        tool_name="sandbox.exec",
        agent_id=ctx.agent_id,
        step_id=step.step_id,
        input={"operation": "write_file", "payload": {"path": "out.txt", "content": "…"}},
    )
)
```

Never import Nexus runtime steps from agent code.

---

## Step 4 — Register the agent

Registration means adding your agent instance to an `AgentRegistry` that is passed to `NexusLoop` (directly or via a Tier-3 application).

```python
from intergrax.runtime.registry.agent_registry import AgentRegistry
from document_automation.document_automation_agent import DocumentAutomationAgent

registry = AgentRegistry()
registry.register(DocumentAutomationAgent())
```

Optional contract override:

```python
contract = DocumentAutomationAgent().get_contract().model_copy(update={"version": "0.2.0"})
registry.register(DocumentAutomationAgent(), contract=contract)
```

### Choose a registration context

| Context | When to use | Where to register |
|---------|-------------|-------------------|
| **A — Smoke test** | Fastest first run; CI for the agent | Already in `agents/<slug>/tests/` (generated) |
| **B — Script / notebook** | Interactive experiments | Your script: `registry.register(...)` |
| **C — Lab application** | HTTP experimentation via `/v1/lab/run` | `applications/lab_application/host/wiring.py` |
| **D — Product application** | Dedicated product host (Legal, Research, …) | `applications/<product>/host/wiring.py` |

There is **no auto-discovery**. Every context requires an explicit `registry.register()` call.

### A — Smoke test (recommended first run)

Generated by scaffold — no extra wiring:

```bash
uv run pytest agents/document_automation/tests -q
```

The test creates its own `AgentRegistry`, registers the agent, and runs `NexusLoop.handle_task()`.

### B — Script or notebook

```python
import asyncio
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from document_automation.document_automation_agent import DocumentAutomationAgent

async def main() -> None:
    registry = AgentRegistry()
    registry.register(DocumentAutomationAgent())
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="lab",
            user_id="dev",
            message="hello",
            context=TaskContext(capability="documents.automation"),
        )
    )
    print(result.state, result.answer)

asyncio.run(main())
```

Notebook template: `agents/<slug>/notebooks/01_<slug>_experiment.ipynb`.

### C — Lab application (HTTP)

**Step C.1 — Add agent to lab registry**

Edit `applications/lab_application/host/wiring.py`:

```python
def build_lab_registry(*, settings: LabApplicationSettings | None = None) -> AgentRegistry:
    settings = settings or LabApplicationSettings.from_env()
    registry = AgentRegistry()

    # … existing Echo / mock registrations …

    from document_automation.document_automation_agent import DocumentAutomationAgent
    registry.register(DocumentAutomationAgent())

    return registry
```

**Step C.2 — Start lab host**

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
```

**Step C.3 — Verify registration**

```bash
curl http://127.0.0.1:8090/v1/lab/agents
```

**Step C.4 — Run**

```bash
curl -X POST http://127.0.0.1:8090/v1/lab/run \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "lab",
    "user_id": "dev",
    "message": "process this document",
    "capability": "documents.automation"
  }'
```

Lab app also exposes `/debug/*` for trace, events, checkpoints, experiments, and HITL.

### D — Product application (Tier-3)

Follow the Legal / Research pattern:

1. Keep agent logic in `agents/<slug>/`
2. In `applications/<product>/host/wiring.py` — import and `registry.register(...)`
3. In `applications/<product>/host/factory.py` — build registry → `NexusLoop` → routes

Example references:

- `applications/legal_application/host/wiring.py`
- `applications/research_application/host/wiring.py`

Applications contain **wiring only** — never agent business logic.

---

## Step 5 — Run the agent

Every run uses the same contract:

```python
Task(
    tenant_id="…",
    user_id="…",
    message="user input or instruction",
    context=TaskContext(capability="<capability from capabilities.py>"),
)
```

| Method | Command / entry point |
|--------|----------------------|
| Smoke test | `uv run pytest agents/<slug>/tests -q` |
| Python | `NexusLoop(registry).handle_task(task)` |
| Lab HTTP | `POST /v1/lab/run` |
| Debug-only API | `uv run uvicorn intergrax.debug.app:create_debug_app --factory --port 8099` |
| Legal host | `uv run uvicorn legal_application.host.main:app --port 8000` |
| Research host | `uv run uvicorn research_application.host.main:app --port 8010` |

**Capability routing:** Nexus reads `task.context.capability` and selects the agent whose `AgentContract.capabilities` includes that id (via `can_handle` / registry lookup).

---

## Step 6 — Inspect traces and runtime events

After a run, inspect via **Lab Application** or **Debug API** (both mount `/debug/*`).

### HTTP endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /debug/tasks?tenant=<t>` | List recent runs |
| `GET /debug/tasks/{task_id}?tenant=<t>` | Run metadata and trace |
| `GET /debug/tasks/{task_id}/trace?tenant=<t>` | Full trace payload |
| `GET /debug/tasks/{task_id}/events?tenant=<t>` | Runtime events |
| `GET /debug/tasks/{task_id}/checkpoints?tenant=<t>` | Checkpoint history |
| `GET /debug/tasks/{task_id}/progress?tenant=<t>` | Partial results (long-running) |
| `POST /debug/human-response` | HITL approve / reject / escalate |
| `POST /debug/interactions/intake?execute=true` | Interaction → task → Nexus |

### CLI

```bash
python -m intergrax.debug tasks list --tenant lab --limit 20
python -m intergrax.debug tasks trace TASK_ID --tenant lab --format json --runtime
```

Environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `INTERGRAX_TRACE_DB` | `build/intergrax_trace.db` | Trace persistence |
| `INTERGRAX_EXPERIMENTS_DB` | `build/intergrax_experiments.db` | Experiment registry |
| `INTERGRAX_TASK_CHECKPOINTS_DB` | `build/intergrax_checkpoints.db` | Checkpoints |
| `INTERGRAX_RUNTIME_EVENTS_DB` | `build/intergrax_runtime_events.db` | Runtime events |

---

## Step 7 — Record experiment and decision

Before running, register the hypothesis. After inspecting results, record the verdict.

### HTTP

```bash
curl -X POST http://127.0.0.1:8090/debug/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "hypothesis": "Document automation returns prefixed stub answer",
    "capability": "documents.automation",
    "agent_id": "document_automation",
    "expected_output": "document_automation: …",
    "validation_criteria": "non-empty answer with agent prefix"
  }'

curl -X POST http://127.0.0.1:8090/debug/experiments/EXPERIMENT_ID/runs/TASK_ID
curl -X POST http://127.0.0.1:8090/debug/experiments/EXPERIMENT_ID/decision \
  -H "Content-Type: application/json" \
  -d '{"decision": "keep"}'
```

### CLI

```bash
python -m intergrax.debug experiments register \
  --hypothesis "…" \
  --capability documents.automation \
  --agent-id document_automation

python -m intergrax.debug experiments link-run EXPERIMENT_ID TASK_ID
python -m intergrax.debug experiments decide EXPERIMENT_ID --decision keep
```

**Decisions:** `keep` · `improve` · `pause` · `delete`

---

## Step 8 — Test and gate

```bash
# Agent smoke test
uv run pytest agents/document_automation/tests -q

# Platform acceptance (10 Agent OS scenarios)
uv run pytest tests/acceptance/agent_os -m agent_os -q

# Full regression gate
uv run pytest tests/ -m gate -q
```

### Pre-merge checklist

- [ ] Capability id defined in `capabilities.py`
- [ ] `AgentContract` complete (description, tools, risk, max_steps)
- [ ] UAEP steps implemented
- [ ] Registered in chosen context (test / lab / product wiring)
- [ ] **Zero** changes to `intergrax/runtime/`
- [ ] Smoke test passes
- [ ] Trace inspectable via debug API
- [ ] No duplicated Tier-0 infrastructure
- [ ] Agent `README.md` present (generated by scaffold)

---

## Step 9 — Sign-off before business agents

Before starting Problem Radar, Vendor Discovery, or other business agents, complete one **live exercise**:

1. Scaffold a **new** agent (not Echo / not an existing mock)
2. Implement minimal domain change in `steps/`
3. Register and run (smoke test or lab app)
4. Confirm **no runtime files** were modified
5. Record result in implementation plan **Appendix A** sign-off template

```text
Date:
Agent exercise: <slug>
Time to first run:
Runtime files modified: none
Acceptance suite: pass
Gate suite: pass
Decision: GO Phase K / HOLD
```

---

## Appendix A — Human-in-the-loop

Agents return `AgentDecision.REQUEST_HUMAN` during UAEP. Nexus pauses in `WAITING_FOR_HUMAN`.

Resume via task metadata:

```python
Task(
    tenant_id="t1",
    user_id="u1",
    message="sensitive action",
    context=TaskContext(capability="…"),
    task_id=original_task_id,
    metadata={"human_response": "approve"},  # or "reject" / "escalate"
)
```

Or HTTP: `POST /debug/human-response`.

---

## Appendix B — Shadow workspace and sandbox

**Shadow workspace** — isolated file artifacts:

```python
Task(..., metadata={"shadow_workspace": True})
```

Inside `run_step`: `workspace = ctx.metadata.get("shadow_workspace")`.

**Sandbox** — permission-controlled tool execution:

```python
Task(..., metadata={"sandbox": True})
```

Add `"sandbox.exec"` to `AgentContract.allowed_tools`. Use `ctx.invoke_tool(ToolRequest(tool_name="sandbox.exec", …))`.

Result metadata includes `shadow_workspace_id` or `sandbox_session_id`.

---

## Appendix C — Multi-agent graphs

Register multiple agents, then run through graph orchestration (Nexus planner or explicit graph):

```python
from intergrax.runtime.registry.bootstrap import build_research_registry

registry = build_research_registry()  # Research + Summary
loop = NexusLoop(registry)
result = await loop.handle_task(
    Task(
        tenant_id="t1",
        user_id="u1",
        message="AI logistics partners in Poland",
        context=TaskContext(capability="research.pipeline", intent="research_summarize"),
    )
)
```

Agents share context via `SharedTaskContext` and `MemoryView` — owned by Nexus, not agent code.

---

## Appendix D — Advanced execution paths

These are platform features consumed by applications — not agent-creation steps.

| Feature | Entry point |
|---------|-------------|
| Unified run API | `POST /runs` via FastAPI Core + `UnifiedTaskRunner` |
| Worker queue | `QueuedNexusExecutionAdapter` + `create_nexus_celery_worker_app` |
| Long-running scheduler | `LongRunningScheduler` + checkpoint store |
| Partial results | `GET /debug/tasks/{id}/progress` |

See [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) for phase tracking.

---

## Anti-patterns

| Do not | Do instead |
|--------|------------|
| Put agent logic in `applications/` | Logic in `agents/`, wiring in application |
| Modify `NexusLoop` for one agent | `registry.register()` + contract/metadata |
| Expect lab app to auto-load new agents | Add explicit `register()` in `lab_application/host/wiring.py` |
| Duplicate LLM/trace/queue stacks | Extend Tier-0 platform |
| Tie agent to one product | Reusable capability in `agents/` |
| Document this workflow in multiple files | Update **this guide only** |

---

## Instructions for LLM coding agents

When asked to create a new Intergrax agent:

1. Read this guide end-to-end.
2. Run `python -m intergrax.scaffold new-agent <slug> --capability <id>`.
3. Edit only `agents/<slug>/` — primarily `steps/`, `prompts/`, `schemas/`, `contract.py`.
4. Register in the appropriate context (§ Step 4). For HTTP lab runs, edit `lab_application/host/wiring.py`.
5. Verify: `uv run pytest agents/<slug>/tests -q` then `uv run pytest tests/ -m gate -q`.
6. Do **not** modify `intergrax/runtime/` unless a reusable Tier-0 gap is proven and approved.
7. Do **not** create duplicate workflow documentation — update this file if the process changes.
