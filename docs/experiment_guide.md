# Intergrax Experiment Guide

Minimal workflow for validating agent capabilities through Nexus (§2, §35, §41).

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

## 3. Run Legal Agent via application host

```bash
uv run uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
# backward compatible:
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

## 4. Decision

After observing traces, cost, and quality: **keep**, **improve**, **pause**, or **delete** the experiment (§35).
