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
17. [Appendix E — Integrations and Tier-0 wiring](#appendix-e--integrations-and-tier-0-wiring)
18. [Appendix F — Tier-3 application environment](#appendix-f--tier-3-application-environment)
19. [Anti-patterns](#anti-patterns)
20. [Instructions for LLM coding agents](#instructions-for-llm-coding-agents)

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

**Reuse Tier-0** (LLM adapters, tools, RAG helpers). Do not duplicate platform infrastructure inside the agent folder. For **which database, cache, or Slack backend** the host uses, see [Appendix E — Integrations](#appendix-e--integrations-and-tier-0-wiring) — agents declare **tools and capabilities**, applications wire **integration slugs**.

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

Follow the Legal / Research pattern using the **Tier-3 composition engine**:

1. Keep agent logic in `agents/<slug>/`
2. Define roster in `applications/<product>/manifest.py` — `AgentBinding.mount(AgentClass, factory=...)`
3. Implement factories in `host/agent_factories.py` or `host/agent_builders.py`
4. In `host/wiring.py` — `build_application_registry(manifest, ctx, builders=...)`
5. In `host/factory.py` — registry → `NexusLoop` → routes

**Usage guides (define / invoke / run):**

- Composition engine API: [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)
- Application folder layout: [`applications/USAGE.md`](../applications/USAGE.md)

Example references:

- `applications/legal_application/manifest.py` + `host/agent_factories.py`
- `applications/lab_application/manifest.py` + `host/agent_builders.py`
- `applications/research_application/host/wiring.py` (legacy explicit register — migrate to manifest when touched)

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
- [ ] No integration slug imports under `agents/` (see Appendix E)
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

## Appendix E — Integrations and Tier-0 wiring

**Canon:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) §7.1  
**Catalog:** `intergrax/integrations/` (Phase M)

### Separation of concerns

```text
Tier-2  agents/           WHAT the agent needs     → capabilities, allowed_tools, ToolRequest
Tier-3  applications/     WHICH vendor/backend     → IntegrationProfile, factory wiring
Tier-0  integrations/     HOW to talk to backend   → providers/<slug>/, contracts
```

| Layer | Declares | Example |
|-------|----------|---------|
| **Agent** (`AgentContract`) | Routing + tool policy | `capabilities=["research.web_search"]`, `allowed_tools=["websearch.query"]` |
| **Application** (`factory.py`) | Provider selection | `relational_store=IntegrationSlug.SQLITE`, `notification_channel=IntegrationSlug.LOG` |
| **Integration provider** | Adapter implementation | `create_sqlite_integration()`, `create_slack_integration()` |

Agents **never** import `intergrax.integrations.providers.*` or choose integration slugs. That belongs in Tier-3 composition roots (`factory.py`, `integration_wiring.py`).

### What agents declare

In `contract.py`, describe **behavior**, not infrastructure:

```python
AgentContract(
    id="research",
    capabilities=["research.web_search", "research.pipeline"],
    allowed_tools=["websearch.query", "sandbox.exec"],  # enforced by ToolAccessPolicy
    risk_level=AgentRiskLevel.MEDIUM,
    max_steps=20,
)
```

| Field | Purpose |
|-------|---------|
| `capabilities` | Nexus routing — which tasks this agent handles |
| `allowed_tools` | Tool gateway allow-list — which `ToolRequest.tool_name` values are permitted |
| `required_adapters` | Optional documentation hint for operators (not auto-wired today) |

When the agent needs external data or side effects, call tools via UAEP — do not open Redis, Postgres, or Slack clients inside `agents/`:

```python
response = await ctx.invoke_tool(
    ToolRequest(tool_name="websearch.query", agent_id=ctx.agent_id, step_id=step.step_id, input={...})
)
```

The application ensures the tool runtime is backed by the correct Tier-0 provider (e.g. Google CSE vs Bing via host config, not agent code).

### What applications wire

Tier-3 factories compose integrations once and pass concrete adapters into `NexusLoop`, schedulers, and debug services.

**Laboratory default** — `IntegrationProfile.lab()` (no external vendors):

| Category | Default slug |
|----------|--------------|
| `relational_store` | `sqlite` |
| `notification_channel` | `log` |
| `interaction_surface` | `lab_json` |

Reference implementation: `applications/lab_application/host/integration_wiring.py` → `wire_lab_integrations()`.

```python
from intergrax.integrations import IntegrationCategory, IntegrationProfile, register_default_integrations
from lab_application.host.integration_wiring import wire_lab_integrations

register_default_integrations()
integrations = wire_lab_integrations(settings=settings, db_path=trace_db_path)

nexus_loop = NexusLoop(
    registry,
    trace_store=integrations.trace_store,
    notification_adapter=integrations.notification_adapter,
    interaction_adapter=integrations.interaction_adapter,
    checkpoint_store=integrations.checkpoint_store,
    runtime_event_store=integrations.runtime_event_store,
)
```

**Custom product profile** — pick slugs per category:

```python
from intergrax.integrations import (
    IntegrationCategory,
    IntegrationProfile,
    IntegrationSlug,
    register_default_integrations,
)

register_default_integrations()
profile = IntegrationProfile(
    relational_store=IntegrationSlug.POSTGRESQL,
    key_value_cache=IntegrationSlug.REDIS,
    notification_channel=IntegrationSlug.SLACK,
    interaction_surface=IntegrationSlug.SLACK,
    options={
        IntegrationSlug.SQLITE: {"data_dir": "build/my_app"},
        IntegrationSlug.REDIS: {"url": "redis://localhost:6379/0"},
    },
)

notifier = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
db_bundle = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

**Cloud-hosted profile** — platform defaults for object storage, message bus, etc.:

```python
profile = IntegrationProfile.with_cloud_platform(IntegrationSlug.AWS)
cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)  # inherits cloud default slug
```

### Environment overrides

Applications may override profile fields without code changes:

```text
INTERGRAX_INTEGRATION_RELATIONAL_STORE=sqlite
INTERGRAX_INTEGRATION_NOTIFICATION_CHANNEL=log
INTERGRAX_INTEGRATION_KEY_VALUE_CACHE=redis
```

Pattern: `INTERGRAX_INTEGRATION_<CATEGORY>` where `<CATEGORY>` is the uppercase enum name (e.g. `MESSAGE_BUS=kafka`).

Use `build_profile_from_env(defaults=IntegrationProfile.lab())` to merge env overrides onto lab defaults.

Provider-specific secrets and paths use each slug's own env prefix (e.g. `INTERGRAX_SQLITE_*`, `INTERGRAX_SLACK_*`) — see `intergrax/integrations/providers/<slug>/`.

### P0 catalog (available today)

| Slug | Category | Typical Tier-3 use |
|------|----------|-------------------|
| `sqlite` | relational_store | Trace, checkpoints, runtime events, HITL (lab) |
| `postgresql` | relational_store | Production SQL facade (`RelationalStore` via `create_postgresql_relational_store()` only) |
| `mysql` | relational_store | Production SQL facade (`RelationalStore` via `create_mysql_relational_store()` only) |
| `jira` | issue_tracker | Jira Cloud REST (`create_jira_issue_tracker()` only) |
| `confluence` | wiki_knowledge | Confluence REST (`create_confluence_wiki_knowledge()` only) |
| `prometheus` | observability_backend | PromQL queries (`create_prometheus_observability_backend()` only) |
| `ms365_graph` | collaboration_suite | Microsoft Graph mail/calendar (`create_ms365_graph_collaboration_suite()` only) |
| `cassandra` | document_store | Cassandra CQL store (`create_cassandra_document_store()` only) |
| `aws` | cloud_platform | AWS facade (`create_aws_cloud_platform()` only) |
| `azure` | cloud_platform | Azure facade (`create_azure_cloud_platform()` only) |
| `gcp` | cloud_platform | GCP facade (`create_gcp_cloud_platform()` only) |
| `elasticsearch` | observability_backend | Log search / aggregations (`create_elasticsearch_observability_backend()` only) |
| `databricks` | relational_store | SQL Warehouse / Unity Catalog (`create_databricks_relational_store()` only) |
| `mongodb` | document_store | Flexible JSON store (`create_mongodb_document_store()` only) |
| `pinecone` | vector_store | RAG index bridge (`create_pinecone_vector_store()` only) |
| `qdrant` | vector_store | RAG index bridge (`create_qdrant_vector_store()` only) |
| `chroma` | vector_store | RAG index bridge (`create_chroma_vector_store()` only) |
| `s3` | object_storage | Blob storage (`create_s3_object_storage()` only) |
| `redis` | key_value_cache | Idempotency, rate limits, distributed locks |
| `kafka`, `rabbitmq`, `celery` | message_bus | Worker queues, async Nexus execution |
| `google_cse`, `bing` | search_provider | Research / web tools |
| `slack`, `teams`, `webhook`, `log` | notification_channel | Long-running progress, HITL alerts |
| `lab_json`, `slack`, `teams` | interaction_surface | Inbound webhooks / lab JSON intake |

### Planned integrations (M.6 P2 / P3 — not yet in default bootstrap)

| Slug | Category | Notes |
|------|----------|-------|
| `azure_blob`, `gcs` | object_storage | Follow S3 bridge pattern (B.34+) |
| `notion`, `sharepoint` | wiki_knowledge | REST wiki sources (B.35) |
| `github`, `linear` | issue_tracker | Dev workflow ingestion (B.36) |
| `email_smtp` | notification_channel | SMTP outbound (B.37) |
| `otel` | observability_backend | OTLP export (B.38) |
| `playwright` | browser_automation | Dynamic web (B.39) |

Full prioritized backlog: [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) — **M.6 P2 tracker**, **M.6 P3 backlog**, **B.31–B.39**.

LLM adapters (`intergrax/llm_adapters/`) are **not** part of the Integration Library — configure them separately.

### Decision checklist

When building a new agent or application:

1. **Agent author:** list capabilities and `allowed_tools` only; implement domain logic in `steps/`.
2. **Application author:** choose `IntegrationProfile`, call `register_default_integrations()`, resolve categories in `factory.py`.
3. **Platform author:** add missing backends under `integrations/providers/<slug>/` — never inside `agents/`.
4. **Verify:** agent smoke tests use minimal wiring (in-memory / defaults); application acceptance tests exercise the chosen profile.

Further detail: [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase M, migration map M.5.

Each provider under `intergrax/integrations/providers/<slug>/` includes an English **`USAGE.md`** with factory + `IntegrationProfile` wiring and a minimal contract API example.

---

## Appendix F — Tier-3 application environment

When an agent needs a **dedicated host** (env, Docker, stable HTTP API) — not only the shared lab — use the Tier-3 stack under `applications/<app>/`.

| Topic | Document |
|-------|----------|
| **Composition engine** — `ApplicationManifest`, `AgentBinding.mount()`, `build_application_registry()` | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| **Application layout** — manifest, host, serving, `.env.example`, docker | [`applications/USAGE.md`](../applications/USAGE.md) |
| Architecture rules | `docs/intergrax_runtime_architecture.md` §7.4.8–§7.4.10 |

Minimal pattern:

```python
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.wiring import build_application_registry

manifest = ApplicationManifest.lab(app_id="my_lab", name="My Lab", agents=[AgentBinding.mount(EchoAgent)])
ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
registry = build_application_registry(manifest, ctx, builders=MY_BUILDERS)
```

Use `python -m intergrax.scaffold new-application <name> --profile lab --agents <slug>` to generate this layout, or copy `lab_application` / `legal_application` as reference.

---

## Anti-patterns

| Do not | Do instead |
|--------|------------|
| Put agent logic in `applications/` | Logic in `agents/`, wiring in application |
| Modify `NexusLoop` for one agent | `registry.register()` + contract/metadata |
| Expect lab app to auto-load new agents | Add `AgentBinding.mount(...)` in `lab_application/manifest.py` + builder |
| Use string `import_path` / `factory_path` in Python manifests | `AgentBinding.mount(AgentClass, factory=callable)` — see `intergrax/applications/USAGE.md` |
| Duplicate LLM/trace/queue stacks | Extend Tier-0 platform |
| Import `integrations/providers/` from `agents/` | Declare `allowed_tools`; wire slugs in Tier-3 `factory.py` |
| Hardcode Slack/Postgres/Redis in agent steps | `ToolRequest` + application `IntegrationProfile` |
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
7. Do **not** import `intergrax.integrations.providers.*` from agent code — wire integrations in Tier-3 only (Appendix E).
8. Do **not** create duplicate workflow documentation — update this file if the process changes.
