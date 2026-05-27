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

**Execution paths:** Echo, Research, Summary, and Legal agents run through **UAEP** (`get_steps` / `run_step` / `decide_after_step` via `AgentEngine`). Pipeline-backed agents use `intergrax/agents/uaep_pipeline.py` as a thin wrapper over `RuntimeEngine`.

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

**Tool access (§42.12):** agents invoke capabilities via `ctx.invoke_tool(ToolRequest(...))` under UAEP, or through the Legal bridge which uses `RuntimeToolGateway` internally — never import Nexus runtime steps from agent code.

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

## 6. Inspect traces (CLI or API)

After a run is finalized to SQLite trace store:

**CLI:**

```bash
python -m intergrax.debug tasks list --tenant t1 --limit 20
python -m intergrax.debug tasks trace RUN_ID --tenant t1 --format json --runtime
```

**HTTP (Phase D.2):**

```bash
uv run uvicorn intergrax.debug.app:create_debug_app --factory --port 8099
curl "http://127.0.0.1:8099/debug/tasks?tenant=t1"
curl "http://127.0.0.1:8099/debug/tasks/RUN_ID/trace?tenant=t1&include_runtime=true"
```

Set `INTERGRAX_TRACE_DB` or pass `--db` (CLI) / `db_path=` (router factory) to point at your trace database.

## 7. Register and decide (experiment registry)

Before running, register the hypothesis (§35 steps 1–4). After inspecting traces (§6), record the verdict.

**CLI:**

```bash
python -m intergrax.debug experiments register \
  --hypothesis "Echo returns deterministic prefixed answer" \
  --capability echo.basic \
  --agent-id echo \
  --expected-output "echo: hello" \
  --validation-criteria "non-empty answer with echo prefix"

# After Nexus run — link trace run_id and decide
python -m intergrax.debug experiments link-run EXPERIMENT_ID RUN_ID
python -m intergrax.debug experiments decide EXPERIMENT_ID --decision keep
python -m intergrax.debug experiments list --decision pending
```

Decisions: `keep` · `improve` · `pause` · `delete` (delete removes the record).

**HTTP:**

```bash
curl -X POST http://127.0.0.1:8099/debug/experiments \
  -H "Content-Type: application/json" \
  -d '{"hypothesis":"...", "capability":"echo.basic"}'

curl -X POST http://127.0.0.1:8099/debug/experiments/EXPERIMENT_ID/runs/RUN_ID
curl -X POST http://127.0.0.1:8099/debug/experiments/EXPERIMENT_ID/decision \
  -H "Content-Type: application/json" \
  -d '{"decision":"keep"}'
```

Registry database: `INTERGRAX_EXPERIMENTS_DB` (default `build/intergrax_experiments.db`).

## 8. Decision

After observing traces, cost, and quality: **keep**, **improve**, **pause**, or **delete** the experiment (§35). Use the registry (§7) to persist the verdict and linked `run_id`s.
