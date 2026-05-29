# Intergrax — Agent Creation Guide

**Canonical process for creating new Tier-2 agents (Phase L.2).**

Audience: humans, GPT, Claude, Gemini, Cursor agents.

**Related docs:**

- [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) — canon
- [INTERGRAX_AGENT_OS_READINESS_PLAN.md](INTERGRAX_AGENT_OS_READINESS_PLAN.md) — readiness status
- [experiment_guide.md](experiment_guide.md) — debug API and experiments

---

## Mental model

```text
Nexus (Tier-1)     = Agent Operating System
Agents (Tier-2)    = reusable capabilities
Applications (Tier-3) = execution environments
```

When creating an agent you think about **domain logic only**:

- decisions, rules, workflows, prompts, tools, outputs

You do **not** modify:

- orchestration, lifecycle, tracing, memory, checkpointing, retries, HITL, state machines

---

## Workflow

```text
new idea
    → hypothesis
    → capability definition
    → scaffold agent
    → implement contract + steps
    → register
    → run in lab
    → trace inspection
    → evaluation
    → decision (keep / improve / pause / delete)
```

Target: **first run in under 1 hour**.

---

## Step 1 — Hypothesis and capability

Write one sentence:

> When `<trigger>`, agent `<name>` should `<outcome>` using `<tools/data>`.

Define capability id(s):

```text
documents.automation
vendor.discovery.basic
```

Capability ids are stable routing keys — not class names.

---

## Step 2 — Scaffold

```bash
python -m intergrax.scaffold new-agent document_automation \
    --capability documents.automation
```

Generated layout:

```text
agents/document_automation/
    document_automation_agent.py   # Agent class (UAEP)
    contract.py                    # AgentContract builder
    capabilities.py                # capability ids
    steps/pipeline.py              # domain execution
    schemas/                       # Pydantic I/O models
    prompts/                       # prompt assets
    tests/                         # smoke test
    notebooks/                     # interactive experiment
    README.md
```

The scaffold is **UAEP-first**: `get_steps` / `run_step` / `decide_after_step`.

---

## Step 3 — Implement domain logic

| File | Responsibility |
|------|----------------|
| `capabilities.py` | Public capability ids |
| `contract.py` | `AgentContract`, risk, tools, max_steps |
| `steps/` | Business steps — prompts, rules, tool calls |
| `prompts/` | System/user prompt templates |
| `schemas/` | Request/response models |

Reuse Tier-0 platform (LLM adapters, storage, queues, notifications). **Do not** duplicate infrastructure inside the agent.

Tool access under UAEP:

```python
response = await ctx.invoke_tool(ToolRequest(...))
```

Never import Nexus runtime steps from agent code.

---

## Step 4 — Register

```python
from intergrax.runtime.registry import AgentRegistry
from document_automation.document_automation_agent import DocumentAutomationAgent

registry = AgentRegistry()
registry.register(DocumentAutomationAgent())
```

Registration is the only runtime integration point for a new agent.

---

## Step 5 — Run

### Option A — NexusLoop (Python)

```python
import asyncio
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task import Task, TaskContext

async def main():
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="lab",
            user_id="dev",
            message="process this document",
            context=TaskContext(capability="documents.automation"),
        )
    )
    print(result.state, result.answer)

asyncio.run(main())
```

### Option B — Lab Application (HTTP)

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
```

```bash
curl -X POST http://127.0.0.1:8090/v1/lab/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"documents.automation"}'
```

---

## Step 6 — Inspect and evaluate

| Action | Endpoint / command |
|--------|-------------------|
| Trace | `GET /debug/tasks/{task_id}` |
| Runtime events | `GET /debug/tasks/{task_id}/events` |
| Checkpoints | `GET /debug/tasks/{task_id}/checkpoints` |
| Partial progress | `GET /debug/tasks/{task_id}/progress` |
| HITL resume | `POST /debug/human-response` |
| Experiment registry | `POST /debug/experiments` |

Register experiment decision: `keep` · `improve` · `pause` · `delete`.

---

## Step 7 — Test

Agent smoke test (generated):

```bash
uv run pytest agents/document_automation/tests -q
```

Agent OS acceptance (platform):

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

Full gate:

```bash
uv run pytest tests/ -m gate -q
```

---

## Checklist before merging a new agent

- [ ] Capability id defined and documented
- [ ] `AgentContract` complete (tools, risk, max_steps)
- [ ] UAEP steps implemented
- [ ] Registered without Nexus code changes
- [ ] Smoke test passes
- [ ] Trace inspectable via debug API
- [ ] No duplicated Tier-0 infrastructure
- [ ] README in agent folder

---

## Anti-patterns

| Do not | Do instead |
|--------|------------|
| Put agent logic in `applications/` | Keep logic in `agents/`, wire in application |
| Modify `NexusLoop` for one agent | Register + configure via contract/metadata |
| Create parallel LLM/trace stacks | Use Tier-0 adapters and runtime events |
| Tie agent to one product | Design reusable capabilities |

---

## For LLM agents (Cursor / Claude / GPT)

When asked to create a new Intergrax agent:

1. Read this guide and `intergrax_runtime_architecture.md` §5.1, §42.
2. Run scaffold CLI — do not hand-create folder structure.
3. Implement only `steps/`, `prompts/`, `schemas/`, `contract.py`.
4. Do not touch `intergrax/runtime/` unless a **reusable** Tier-0 gap is proven.
5. Verify with `pytest agents/<name>/tests` and lab application run.
6. Update experiment registry via debug API if running a formal experiment.
