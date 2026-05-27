# Intergrax Experiment Guide

Minimal workflow for validating agent capabilities through Nexus (CANON §2, §35, §41).

**Canonical spec:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md)  
**Documentation map:** [README.md](README.md)

## Prerequisites

- Repository root on `PYTHONPATH` (pytest / uv run handle this via `conftest.py`)
- `agents/` and `applications/` on path for `legal`, `echo`, `legal_application`

## 1. Register an agent

```python
from intergrax.runtime.registry import AgentRegistry, build_harness_registry
from echo.echo_agent import EchoAgent

registry = build_harness_registry(include_echo=True)
# or: registry = AgentRegistry(); registry.register(EchoAgent())
```

Or scaffold a new agent:

```bash
python -m intergrax.scaffold new-agent research --capabilities research.web_search
```

## 2. Run via NexusLoop

```python
import asyncio
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task import Task, TaskContext

async def main():
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="hello experiment",
            context=TaskContext(capability="echo.basic"),
        )
    )
    print(result.answer, result.state)

asyncio.run(main())
```

**Execution paths:** `EchoAgent` uses **UAEP** (`get_steps` / `run_step` via `AgentEngine`). `LegalAgent` still uses the **legacy pipeline** path until migrated (P4.5).

## 3. Human-in-the-loop (governance)

Agents may return `AgentDecision.REQUEST_HUMAN` during UAEP execution. Nexus pauses the task in `WAITING_FOR_HUMAN` and stores `HumanRequest` in task metadata.

Resume by re-submitting with approval in metadata:

```python
task = Task(
    tenant_id="t1",
    user_id="u1",
    message="sensitive action",
    context=TaskContext(capability="hitl.basic"),
    metadata={"human_approved": True},
)
result = await loop.handle_task(task)
```

Inspect pause events: `HUMAN_APPROVAL_REQUESTED` / `HUMAN_APPROVAL_RECEIVED` on `loop.event_bus.history`.

## 4. Run Legal Agent via application host

```bash
uv run uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
```

Optional global loop for HTTP:

```bash
set LEGAL_USE_NEXUS_LOOP=true
```

## 5. Research pipeline (multi-agent)

```python
from intergrax.runtime.registry import build_research_registry
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task import Task, TaskContext

registry = build_research_registry()
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

HTTP host:

```bash
uv run uvicorn research_application.host.main:app --host 0.0.0.0 --port 8010
```

## 6. Decision

After observing traces, cost, and quality: **keep**, **improve**, **pause**, or **delete** the experiment (§35).
