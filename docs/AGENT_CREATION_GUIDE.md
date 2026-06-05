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
19. [Appendix G — Memory & RAG naming](#appendix-g--memory--rag-naming-phase-q)
20. [Appendix H — Governance, policy & observability](#appendix-h--governance-policy--observability-control-plane)
21. [Appendix I — Orchestration control plane](#appendix-i--orchestration-control-plane)
22. [Appendix J — Tools & skills control plane](#appendix-j--tools--skills-control-plane)
23. [Appendix K — Integration & RAG control plane](#appendix-k--integration--rag-control-plane)
24. [Appendix L — Context engineering control plane](#appendix-l--context-engineering-control-plane)
25. [Appendix M — Prompt registry control plane](#appendix-m--prompt-registry-control-plane)
26. [Appendix N — Agent assembly control plane](#appendix-n--agent-assembly-control-plane)
27. [Appendix O — Registry architecture control plane](#appendix-o--registry-architecture-control-plane)
28. [Appendix P — Capability graph control plane](#appendix-p--capability-graph-control-plane)
29. [Appendix Q — Observability control plane closeout](#appendix-q--observability-control-plane-closeout)
30. [Appendix R — Reliability control plane closeout](#appendix-r--reliability-control-plane-closeout)
31. [Appendix S — Security control plane closeout](#appendix-s--security-control-plane-closeout)
32. [Appendix T — Cost governance control plane closeout](#appendix-t--cost-governance-control-plane-closeout)
33. [Anti-patterns](#anti-patterns)
34. [Instructions for LLM coding agents](#instructions-for-llm-coding-agents)

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
  → register                    (pick context: test / script / lab / product / scaffold app)
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

For agents that run inside **lab** or **product** hosts with injected integrations (echo, legal, research pattern):

```bash
python -m intergrax.scaffold new-agent my_probe \
    --capability my_probe.basic \
    --reference
```

`--reference` emits `HarnessReferenceAgent` + `LabHarnessContext` wiring — Tier-3 `host/agent_builders.py` injects the harness; the agent package must **not** import `applications.*`.

Repeat `--capability` for multiple capabilities. Default if omitted: `<slug>.basic`.

**DX shortcuts (no Nexus edits):**

```bash
uv run intergrax doctor
uv run intergrax run applications.poc_template_application.host.main:app --reload
uv run python -m intergrax.scaffold new-stack my_lab --profile lab --capability my_lab.basic
```

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
| **C — Lab application** | HTTP experimentation via `/v1/lab/run` | `applications/lab_application/manifest.py` + `host/wiring.py` |
| **D — Product application** | Existing product host (Legal, Research, …) | `applications/<product>/manifest.py` + `host/wiring.py` |
| **E — Dedicated application (scaffold)** | New deployable host (env, Docker, HTTP API) | `python -m intergrax.scaffold new-application` → § [Step 4E](#e--dedicated-application-scaffold) |

### Production lifecycle governance note (Phase V)

When an agent is intended for production eligibility, registration and runtime success are not sufficient.
The agent must satisfy lifecycle governance gates tracked in implementation plan **Phase V (V-ALG.\*)**:

- certification evidence (quality/policy/security),
- promotion path evidence (dev -> staging -> production),
- explicit owner/on-call metadata,
- deprecation/retirement policy metadata.

Use this guide for creation workflow, and use Phase V governance streams for production lifecycle readiness.

There is **no auto-discovery**. Every context requires an explicit roster entry (`AgentBinding.mount` or `registry.register()`).

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

**Step C.1 — Add agent to lab roster**

Edit `applications/lab_application/manifest.py` (and `host/agent_builders.py` if the agent needs a custom factory). The lab host assembles the registry via `build_application_registry()` in `host/wiring.py` — do not call `registry.register()` by hand unless you are in a one-off script.

Example binding:

```python
from document_automation.document_automation_agent import DocumentAutomationAgent
# inside build_lab_manifest() agents=[ ... ]
AgentBinding.mount(DocumentAutomationAgent, capabilities=["documents.automation"]),
```

Add a zero-arg builder in `host/agent_builders.py` when needed:

```python
DocumentAutomationAgent: lambda ctx, binding: DocumentAutomationAgent(),
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

Use this path when extending an **existing** product host (Legal, Research, …). For a **new** host, prefer **Step 4E** (scaffold).

Manual / reference pattern:

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
- `applications/research_application/` — product-style host

Applications contain **wiring only** — never agent business logic.

### E — Dedicated application (scaffold)

**When to use:** you need a **separate** Tier-3 host with its own `.env`, HTTP API, optional Docker image, and stable package name — not only the shared lab. Typical after agent smoke tests (Step 4A) pass.

**Canon:** `docs/intergrax_runtime_architecture.md` §7.4.8–§7.4.10  
**Phase N status:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) (Phase N table)  
**Reference tree:** `applications/poc_template_application/` (committed scaffold example)

#### E.0 — 15-minute minimal path (Phase DX-3.6)

From repository root — fastest harness loop (no Docker/MCP until you promote):

```bash
# 1. Agent + lab host (minimal factory)
python -m intergrax.scaffold new-stack my_feature --profile lab --minimal \
  --capability my_feature.basic

# 2. Agent smoke test
uv run pytest agents/my_feature/tests -q

# 3. Run HTTP host (prints route + sample curl)
uv run intergrax run my_feature_application.host.main:app

# 4. Invoke (replace port/prefix from CLI output)
curl -s -X POST http://127.0.0.1:8091/v1/my_feature/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"my_feature.basic"}'
```

**Progressive disclosure (Phase DX-0.4):**

| Stage | Command / artifact | What you get |
|-------|-------------------|--------------|
| **Minimal** | `new-stack --minimal` | Lab host via `build_harness_host_runtime` + `create_lab_fastapi_from_runtime`; agent under `agents/<slug>/` |
| **Standard** | `new-stack` (no `--minimal`) | Docker, MCP, `BUILD_AND_DEPLOY.md`, full factory + debug scheduler |
| **Promote** | `python -m intergrax.scaffold expand <app_slug>` | Adds standard files to an existing minimal lab application |

See also [`applications/USAGE.md`](../applications/USAGE.md) § Progressive disclosure.

#### E.1 — Choose scaffold profile

| Profile | CLI | Host style | Default port |
|---------|-----|------------|--------------|
| `lab` | `--profile lab` | Debug API + `POST <prefix>/run` + `/debug/*` | 8091 |
| `product` | `--profile product` | FastAPI Core (`/health`, `/v1/*`) + auth env stubs | 8000 |

**Full stack (agent + application):**

```bash
python -m intergrax.scaffold new-stack my_feature \
  --profile lab \
  --capability my_feature.basic
```

Creates `agents/my_feature/` and `applications/my_feature_application/` in one step.

```bash
# From repository root — lab host for experimentation
python -m intergrax.scaffold new-application my_lab \
  --profile lab \
  --agents echo,my_agent \
  --port 8091 \
  --prefix /v1/my_lab

# Product-style host (Legal-like factory layout)
python -m intergrax.scaffold new-application my_product \
  --profile product \
  --agents echo \
  --port 8000
```

`--agents` accepts built-in slugs (`echo`, `research`, `signoff_probe`) or your scaffolded agent slug under `agents/<slug>/`.

The CLI prints package name, uvicorn command, pytest path, MCP mount, and Docker script paths.

#### E.2 — Generated layout

Creates `applications/<name>_application/` (package suffix is automatic):

```text
applications/my_lab_application/
  manifest.py                 # ApplicationManifest.lab | .product
  README.md                   # Three-command quickstart
  BUILD_AND_DEPLOY.md         # Runbook: local, verify, Docker, prod checklist
  .env.example                # MY_LAB_* (copy to .env, gitignored)
  host/                       # factory, settings, wiring, integration_wiring
  serving/                    # fastapi_router (+ schemas.py for product)
  mcp/server.py               # FastMCP on same uvicorn process (/mcp)
  docker/
    Dockerfile, .dockerignore, docker-compose.yml
    build-docker.sh, build-docker.bat
  my_lab_application_tests/   # host smoke tests
```

Import as `my_lab_application.host.main:app` (folder `applications/` is on `pythonpath`).

#### E.3 — Register your agent in the new host

Scaffold pre-registers agents from `--agents`. To add or change the roster after creation:

1. **Manifest** — `applications/<pkg>/manifest.py`:

   ```python
   AgentBinding.mount(MyAgent, capabilities=["my_domain.action"], default=True),  # product: one default
   ```

2. **Builders** (zero-arg agents) — `host/agent_builders.py`:

   ```python
   MyAgent: lambda ctx, binding: MyAgent(),
   ```

3. **Factories** (settings-driven agents, product profile) — `host/agent_factories.py` + typed factory callable.

4. **Wiring** — usually unchanged; calls `build_application_registry(manifest, ctx, builders=...)`.

Re-run host smoke tests after edits.

#### E.4 — Configure environment

```bash
cp applications/my_lab_application/.env.example applications/my_lab_application/.env
```

Variables use the application prefix (`MY_LAB_`, `MY_PRODUCT_`, …). Do not put app-only secrets in the repository-root `.env` only.

Product profile: optional dev API key via `*_BACKEND_BOOTSTRAP_API_KEY` (+ tenant/user); production requires keys or explicit `*_BACKEND_ALLOW_UNAUTHENTICATED=true` (see generated `host/settings.py`).

**Lab / scaffold harness defaults (Phase Q-N.10, DX-6.1):** Tier-2 agents and lab hosts use `intergrax.agents.defaults.harness_production_mode()` (returns `False`) on `RuntimeConfig` so governance and shadow policies stay relaxed during local iteration. Product profiles set `production_mode=True` explicitly in `host/factory.py`. Tier-3 may re-export via `intergrax.applications._shared.runtime_defaults`.

#### E.5 — Three-command quickstart

From **repository root**:

```bash
# 1. Verify
uv run pytest applications/my_lab_application/my_lab_application_tests -q

# 2. Run locally
uv run uvicorn my_lab_application.host.main:app --host 127.0.0.1 --port 8091

# 3. Container image (monorepo root context)
applications/my_lab_application/docker/build-docker.sh
# Windows: applications\my_lab_application\docker\build-docker.bat
```

Product profile: also `curl http://127.0.0.1:8000/health` before `POST <route_prefix>/run`.

Operational detail: `applications/<pkg>/BUILD_AND_DEPLOY.md`.

#### E.6 — HTTP verification

**Lab profile:**

```bash
curl -s http://127.0.0.1:8091/v1/my_lab/agents
curl -s -X POST http://127.0.0.1:8091/v1/my_lab/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'
```

**Product profile:**

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/my_product/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'
```

**MCP (both profiles):** FastMCP on `/mcp` by default — tools `list_agents`, `run_agent` (same Nexus loop as HTTP). Toggle with `<PREFIX>_INCLUDE_MCP` / `<PREFIX>_MCP_MOUNT_PATH`.

#### E.7 — Integrations and deploy

- **Integrations:** edit `host/integration_wiring.py` — lab scaffold uses `IntegrationProfile.lab()`; product uses `wire_nexus_observability()`. Agents still declare tools only; see [Appendix E](#appendix-e--integrations-and-tier-0-wiring).
- **Docker:** scripts `cd` to monorepo root and build with `applications/<pkg>/docker/Dockerfile`. Override tag: `IMAGE_TAG=my-registry/my_lab:1.0.0` (sh) or `build-docker.bat my-registry/my_lab:1.0.0` (bat).
- **Gate:** after application smoke tests, run `uv run pytest -m gate -q`.

#### E.8 — Further reading

| Topic | Document |
|-------|----------|
| Composition engine API | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| Application folder conventions | [`applications/USAGE.md`](../applications/USAGE.md) |
| Tier-3 summary (manifest snippet) | [Appendix F](#appendix-f--tier-3-application-environment) |

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

**Full orchestration map:** [Appendix I](#appendix-i--orchestration-control-plane) (control plane, contracts, hooks, customization).

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

**Declarative topology (Tier-3):** `AgentGraph` fluent builder → `ApplicationGraphSpec` on `ApplicationEnvironmentProfile.graph_spec` (roster validation, DX round-trip). Runtime bridge via `GraphSpecSeedingPlanner` when the task has no pre-built plan id — see Appendix I §I.4.

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
Tier-3  applications/     WHICH vendor/backend     → IntegrationProfile, ToolProfile, factory wiring
Tier-0  integrations/     HOW to talk to backend   → providers/<slug>/, contracts
Tier-0  tools/              WHAT the LLM invokes     → providers/<domain>/, ToolContract (see TOOLS.md)
```

| Layer | Declares | Example |
|-------|----------|---------|
| **Agent** (`AgentContract`) | Routing + tool policy | `capabilities=["research.web_search"]`, `allowed_tools=["websearch.query", "rag.retrieve"]` |
| **Application** (`factory.py`) | Provider + tool selection | `IntegrationProfile`, `ToolProfile`, `ToolWiringContext` |
| **Integration provider** | Adapter implementation | `create_jira_issue_tracker()`, `create_google_cse_search_provider()` |
| **Tool provider** | LLM-facing operation | `jira.search_tasks`, `websearch.query` — composes integrations |

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

The application ensures the tool runtime is backed by the correct Tier-0 provider (e.g. Google CSE vs Bing via host config, not agent code). See [TOOLS.md](TOOLS.md) for catalog tool_ids and Phase O wiring (`ToolProfile`, `ToolWiringContext`).

#### Tool catalog wiring (Phase O.8 — unified model)

Applications enable catalog tools via `ToolProfile` and inject dependencies via `ToolWiringContext`. Reference implementations:

| Application | `host/tool_wiring.py` |
|-------------|----------------------|
| Lab | `wire_lab_tools()` — RAG, websearch, sandbox |
| Legal | `wire_legal_tools()` — env-driven RAG/websearch |
| Research | `wire_research_tools()` — websearch by default |
| POC template | `wire_poc_template_tools()` — lab-like defaults |

```python
from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.bootstrap import register_default_tools

register_default_tools()
tool_wiring = build_application_tool_wiring(
    ToolProfile(enabled=["rag.retrieve", "websearch.query", "jira.search_tasks"]),
    integration_profile=integration_profile,
)

# Pass into RuntimeConfig when building agent runtime:
RuntimeConfig(
    llm_adapter=...,
    tool_profile=tool_wiring.profile,
    tool_wiring_context=tool_wiring.wiring_context,
    enable_rag=True,
    enable_websearch=True,
)
```

**Unified tool model (Phase O.5):** agents and legal tool-decision SHOULD prefer explicit `tool_ids` (`rag.retrieve`, `websearch.query`) over legacy `use_rag` / `use_websearch` booleans. Booleans still work — they map to catalog tool_ids and emit a deprecation trace.

```python
# Canonical plan surface (Tier-1 / legal bridge)
ToolRequest(
    tool_name="nexus.capability_plan",
    input={
        "tool_ids": ["rag.retrieve", "websearch.query"],
        "use_tools": False,
    },
)
```

MCP hosts may expose catalog schemas via `list_catalog_tools` / `describe_catalog_tool` (see `intergrax/applications/_shared/mcp_catalog_tools.py`).

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
    register_default_integrations,
)

register_default_integrations()
profile = IntegrationProfile(
    relational_store="postgresql",
    key_value_cache="redis",
    notification_channel="slack",
    interaction_surface="slack",
    options={
        "sqlite": {"data_dir": "build/my_app"},
        "redis": {"url": "redis://localhost:6379/0"},
    },
)

notifier = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
db_bundle = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

**Cloud-hosted profile** — platform defaults for object storage, message bus, etc.:

```python
profile = IntegrationProfile.with_cloud_platform("aws")
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
| `google_cse`, `bing`, `brave`, `serpapi` | search_provider | Research / web tools |
| `slack`, `teams`, `webhook`, `log`, `email_smtp` | notification_channel | Long-running progress, HITL alerts, SMTP mail |
| `lab_json`, `slack`, `teams` | interaction_surface | Inbound webhooks / lab JSON intake |
| `playwright` | browser_automation | JS-heavy pages via headless browser |

### Extended integrations (M.6 P2/P3 — registered in default bootstrap, beta)

| Slug | Category | Notes |
|------|----------|-------|
| `azure_blob`, `gcs`, `s3` | object_storage | Blob put/get/delete/presigned URL |
| `dynamodb`, `mongodb`, `cassandra` | document_store | Partition-scoped document CRUD |
| `sqs`, `service_bus`, `pubsub` | message_bus | Cloud-native queues (also via platform facades) |
| `memcached`, `elasticache`, `redis` | key_value_cache | Cache tiers |
| `oracle`, `mssql`, `azure_sql`, `cloud_sql` | relational_store | Enterprise SQL backends |
| `notion`, `sharepoint`, `confluence` | wiki_knowledge | Internal docs / runbooks |
| `github`, `linear`, `azure_devops`, `jira` | issue_tracker | ALM / dev workflow sources |
| `google_workspace`, `ms365_graph` | collaboration_suite | Mail / calendar / directory |
| `otel`, `prometheus`, `elasticsearch` | observability_backend | Metrics and log search |

Full catalog (99 providers, each with English `USAGE.md`): [`INTEGRATIONS.md`](INTEGRATIONS.md). Per-slug examples: `intergrax/integrations/providers/<category>/<slug>/USAGE.md`.

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

**Primary workflow:** [Step 4E — Dedicated application (scaffold)](#e--dedicated-application-scaffold) (CLI, three-command quickstart, Docker scripts).

| Topic | Document |
|-------|----------|
| **Scaffold CLI** — `new-application`, lab vs product profile | [Step 4E](#e--dedicated-application-scaffold) |
| **Composition engine** — `ApplicationManifest`, `AgentBinding.mount()`, `build_application_registry()` | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| **Application layout** — manifest, host, serving, `.env.example`, docker | [`applications/USAGE.md`](../applications/USAGE.md) |
| **Deploy runbook** — per-app `BUILD_AND_DEPLOY.md` | Generated by scaffold; see `applications/poc_template_application/` |
| Architecture rules | `docs/intergrax_runtime_architecture.md` §7.4.8–§7.4.10 |
| Implementation plan | [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase N |

Minimal pattern (hand-written manifest):

```python
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.wiring import build_application_registry

manifest = ApplicationManifest.lab(app_id="my_lab", name="My Lab", agents=[AgentBinding.mount(EchoAgent)])
ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
registry = build_application_registry(manifest, ctx, builders=MY_BUILDERS)
```

Prefer `python -m intergrax.scaffold new-application <name> --profile lab|product --agents <slug>` over copying folders by hand.

---

## Appendix G — Memory & RAG naming (Phase Q)

### Four memory stores (canon §27 mapping)

Canon §27 defines five memory **types**; runtime implements **four operational stores** plus trace and RAG:

| Canon type | Runtime store | Module |
|------------|---------------|--------|
| Task memory | Task KV (`TaskMemory` + `MemoryView`) | `runtime/task_memory/` |
| Agent local memory | Same task KV namespaces (UAEP) | `PolicyScopedMemoryView` |
| User / org memory | `UserProfileManager` + `OrganizationProfileManager` | `intergrax/memory/`, `runtime/organization/` |
| Long-term knowledge | RAG vectorstore (not agent-mutable memory) | `rag/` |
| Execution trace | `RunTraceWriter` / `RuntimeEvent` (immutable) | `runtime/nexus/tracing/` |

Short-term session history uses `SessionManager` + `SessionStorage` (SQLite when `relational_store=sqlite` on `IntegrationProfile`).

**MemoryKind tags** (`USER_FACT`, `PREFERENCE`, `SESSION_SUMMARY`, `ORG_FACT`, `POLICY`) classify LTM **entries** — not a full episodic/semantic/procedural taxonomy (IDEAL vision only).

### Session vs checkpoint vs task KV (LangGraph thread analogy)

| Concept | Intergrax | Persists when |
|---------|-----------|---------------|
| Thread / session | `SessionManager` + `session_id` | `INTERGRAX_SESSION_DB` / sqlite bundle |
| Checkpointer | `SQLiteTaskCheckpointStore` (long-running UAEP) | `INTERGRAX_TASK_CHECKPOINTS_DB` |
| Scoped KV / store | `TaskMemory` + `MemoryView` namespaces | `INTERGRAX_TASK_MEMORY_DB` |

Use **session** for turn-by-turn chat; **task KV** for per-run agent scratch state; **checkpoints** for resumable UAEP loops — do not mix them.

### Persistence backends (Appendix G matrix)

| Layer | In-memory | SQLite (lab default) | Notes |
|-------|-----------|----------------------|-------|
| Task KV | tests | `INTERGRAX_TASK_MEMORY_DB` | `wire_task_memory_from_profile` |
| Session | fallback | sqlite bundle | `memory_wiring.resolve_memory_platform_wiring` |
| User LTM | tests | `intergrax_user_profile.db` in bundle | `SQLiteUserProfileStore`; optional Mongo `DocumentStoreUserProfileStore` |
| Org profile | tests | sqlite bundle | `SQLiteOrganizationProfileStore` |
| Redis | — | — | **Integration cache only** — not session/LTM |

### Context compression strategy (§28.1)

| Mechanism | Location | Strategy |
|-----------|----------|----------|
| Context budget | `ContextBudgetPolicy` on `ContextProfile` | char + token-estimate trim |
| Summary tiers | `TaskContextAssemblyOptions` | FULL / SUMMARY_ONLY / STRUCTURED_ONLY / MINIMAL |
| History layer | `engine_history_layer.py` | `SUMMARIZE_OLDEST`, truncate fallback |
| LTM limits | `RuntimeConfig` | `max_longterm_entries_per_query`, `max_longterm_tokens` |

Configure via `ApplicationEnvironmentProfile.context_profile` — mapped by `materialize_runtime_config` (Phase MEM).

### Org memory scope

Organization memory in Intergrax is **profile + instructions** (`OrganizationProfileManager`) — not a full shared episodic or team knowledge product. Use RAG / document stores for org-wide knowledge bases; use org profile for tone, constraints, and system instructions.

### Task memory wiring vs Nexus LTM steps

`wire_task_memory_from_profile` enables the **task KV database** when `MemoryProfile.enable_task_memory` (or user/org/LTM flags) is set. It does **not** auto-register Nexus runtime steps for user/org LTM — those flow through `SessionManager` profile managers when `enable_user_memory` / `enable_org_memory` are true on the environment profile.

### MemoryView namespaces + delegation

- Default namespaces: agent-specific keys under `PolicyScopedMemoryView`
- Delegation: `task_id/delegation/{node_id}/` (see `delegation_memory.py`)
- Shared handoff: `shared_task_context` metadata bridge

### Recovery semantics

| Layer | Key | Survives restart (sqlite lab) |
|-------|-----|------------------------------|
| Task KV | `tenant_id` + `task_id` + namespace | Yes |
| Session | `session_id` | Yes |
| User LTM | `tenant_id` + `user_id` | Yes (sqlite bundle) |
| Checkpoint | `task_id` + UAEP cursor | Yes |

### Four memory stores (legacy table)

| Store | Module | When to use |
|-------|--------|-------------|
| Session history | `SessionManager` / `HistoryStep` | Turn-by-turn chat in one session |
| User LTM | `UserProfileManager` + vector index | Stable user facts across sessions |
| Task KV | `TaskMemory` (`INTERGRAX_TASK_MEMORY_DB`) | Per-task scratch state, UAEP steps |
| Shared graph context | `shared_task_context` metadata | Multi-agent handoff on one Nexus task |

Enable SQLite task memory in Tier-3 via `wire_task_memory_from_profile` and `memory_wiring` (Phase MEM). Lab, `poc_template`, `legal_application`, and `research_application` reference hosts call `ApplicationEnvironmentProfile.with_harness_memory()` (or `lab_defaults`) so `MemoryProfile` drives `RuntimeConfig`.

### Three “context builders”

| Name | Location | Role |
|------|----------|------|
| Nexus `ContextBuilder` | `runtime/nexus/context/context_builder.py` | Assembles RAG/history/web for one runtime turn |
| `ContextManager` | `runtime/nexus/context/context_manager.py` | Nexus task-level context orchestration |
| `DefaultContextBuilder` | `rag/context/` (ingest/index) | Document chunking for index pipelines |

Use `tool_ids` including `rag.retrieve` instead of legacy plan boolean `use_rag` (shim emits deprecation event).

### Context engineering (Harness AI)

Nexus owns **what the LLM sees** per step (`ContextManager`, `TaskContextAssemblyOptions`, `MemoryView`). See architecture §28.1. `ContextBudgetPolicy` provides central trim + `CONTEXT_ASSEMBLED` / `CONTEXT_TRIMMED` events (R-Context **Done**).

### Integration → Tool → Skill → Agent

| Layer | Declare in agent | Wire in Tier-3 |
|-------|------------------|----------------|
| Integration | — | `IntegrationProfile` |
| Tool | `allowed_tools` | `ToolProfile` |
| Skill | `skill_ids` on contract | `SkillProfile` |

Do **not** register markdown instruction packs as `ToolContract`. Import external skills via `CursorSkillImporter` — see [SKILLS.md](SKILLS.md). Canon: §7.1.8.

---

## Appendix H — Governance, policy & observability (control plane)

**Audience:** Tier-3 application authors, platform engineers, operators.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §5 (Policy), §21 (Observability); [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.3, §3.9.

Intergrax is **policy-first** and **event-first**. Governance and observability are **modular, composable layers** — not a single monolithic dashboard. Authors configure them through typed profiles, bundles, hooks, and integration slugs; Nexus enforces them on every run.

### H.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Policy-first | No tool/LLM path without `ToolRuntime`, `PolicyEngine`, or `ApplicationSecurityProfile` middleware |
| Trace-everything | `RuntimeEvent` + Nexus trace DB; canon §42.1, §42.1.5 |
| Composable-by-default | `RuntimePolicyBundle`, `SkillResolver`, `HookRegistry`, plugin entry points |
| Tier separation | Agents declare capabilities; **Tier-3** composes policy and observability |

### H.2 Control plane map (where to customize)

```text
ApplicationEnvironmentProfile (Tier-3 umbrella)
  ├── security_profile          → V-SEC toggles (prompt/tool/retrieval/tenant) → application_security_wiring.py
  ├── policy_rules              → YAML declarative rules (lab: harness_lab.yaml) → policy_wiring.py
  ├── identity_profile          → API key, tenant_required, service identities
  ├── context_profile           → budget, RAG/web flags → RuntimeConfig + CONTEXT_* events
  ├── memory_profile            → STM/LTM/task flags → memory_wiring.py (Appendix G)
  ├── observability_profile     → trace SQLite, OTEL, metrics plugins
  ├── reliability_profile       → idempotency, circuit breaker, checkpoints, scheduler
  ├── execution_mode            → strict | balanced | exploratory → runtime_policies
  └── integration_profile       → observability_backend slug (prometheus, otel, elasticsearch, …)

RuntimePolicyBundle (composed once per host)
  ├── tool_access / tool_scope  → allowed_tools intersection with AgentContract
  ├── budget / plan_loop        → token/cost ceilings, plan iteration limits
  ├── hitl                        → human approval requirements
  └── domain_fragments          → skill policy_fragment_id + app-specific keys

Skill layer
  └── SkillManifest.policy_fragment_id → merges into domain_fragments (never bypasses ToolRuntime)

Hook layer (Tier-1)
  └── HookPoint (BEFORE_TOOL_CALL, BEFORE_MEMORY_WRITE, …) → middleware + trace

Plugin extension (Tier-0)
  ├── intergrax.policy_rules     → custom PolicyEngine rule handlers (DX-5.8)
  ├── intergrax.integrations/tools/skills/memory_stores → catalog plugins (P-Ext, MEM)
  └── RuntimePlugin              → Nexus metrics/persistence middleware (not catalog EP)
```

**Rule:** compose policy **once** at Tier-3 startup (`build_runtime_policy_bundle`, `wire_application_environment`). Agents MUST NOT construct parallel policy objects.

### H.3 Security profile (per application)

`ApplicationEnvironmentProfile.security_profile` (`ApplicationSecurityProfile`) maps to Phase V-SEC / V-REM wiring:

| Field | Effect when enabled |
|-------|---------------------|
| `prompt_defense_enabled` | Prompt injection defense on LLM path |
| `tool_injection_defense_enabled` | `ToolInjectionDefenseMiddleware` on `BEFORE_TOOL_CALL` |
| `retrieval_poisoning_defense_enabled` | Trust-score / quarantine on RAG retrieval |
| `tenant_security_verify_enabled` | Tenant boundary checks at task intake |

Wiring: `intergrax/applications/_shared/application_security_wiring.py`. Gate tests under `tests/unit/runtime/architecture/` and integration paths.

### H.4 Policy bundle — operator read order

Canonical operator checklist: architecture [§42.11.5](intergrax_runtime_architecture.md#42115-how-to-read-policy-for-a-run-operator).

| Step | Inspect | Location |
|------|---------|----------|
| 1 | Composed bundle | `ApplicationBuildContext.policy_bundle` |
| 2 | Agent + skills | `AgentContract.skill_ids` → `SKILL_RESOLVED` event |
| 3 | Tool enforcement | `ToolRuntime` + `resolve_allowed_tools_from_config` |
| 4 | Domain overlays | `domain_fragments` / `policy_rules` YAML |
| 5 | Human gates | `PolicyDecision.REQUIRE_HUMAN` → HITL queue |

Lab example: `applications/lab_application/policy/rules/harness_lab.yaml` referenced from `build_lab_environment_profile()`.

### H.5 Observability — what is mandatory vs optional

| Signal | Mechanism | Mandatory in harness? |
|--------|-----------|------------------------|
| Lifecycle events | `RuntimeEventBus` → SQLite / trace store | **Yes** — gate + §42.1 rules |
| Event catalog + ops filters | `EVENT_OPS_FILTER_HINTS` (§42.1.5) | **Yes** — `test_all_runtime_event_types_have_ops_filter_hint` |
| LLM/RAG metrics | `TASK_COMPLETED` payload + plugins | **Yes** when env flags set |
| External observability backend | `IntegrationProfile.observability_backend` | **Optional** — prometheus, otel, elasticsearch |
| Lab debug APIs | `GET /debug/tasks/{id}/trace`, `/events`, `/metrics` | **Lab default** |
| Unified product dashboard | — | **Not shipped** — integrate via observability_backend or scrape debug APIs |

**Inspect a run (lab):**

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
# POST /v1/lab/run  →  GET /debug/tasks/{id}/trace?include_runtime=true
# GET /debug/tasks/{id}/events  →  GET /debug/tasks/{id}/metrics
```

Operator SLO catalog, runbooks, release cycles: [`HARNESS_ENVIRONMENT.md`](HARNESS_ENVIRONMENT.md).

### H.6 Policy rule plugins

Entry point group: `intergrax.policy_rules` (mirror P-Ext pattern).

- Loader: `intergrax/runtime/policy/rules/plugin_loader.py`
- Declarative YAML: `load_policy_rules_from_path` + `PolicyRulesProfile.rules_path`
- Author guide: [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) §10

### H.7 What agents (Tier-2) must not do

| Do not | Do instead |
|--------|------------|
| Import vendor SDKs or integration slugs | Declare `allowed_tools`; wire `IntegrationProfile` in Tier-3 |
| Build custom `PolicyEngine` per agent | Use `RuntimePolicyBundle` + `skill_ids` |
| Bypass `ToolRuntime` | All tool calls through gateway |
| Assume observability is automatic in tests | Assert `RuntimeEvent` types in integration tests |

### H.8 Verification (audit evidence)

| Concern | Command / artifact |
|---------|-------------------|
| Policy + memory bridge | `pytest tests/unit/applications/test_reference_hosts_memory_bridge.py -m gate` |
| W-OPS memory platform | `python scripts/phase_w_ops_evidence.py` |
| Event catalog completeness | `pytest tests/unit/runtime/events/ -m gate -k ops_filter` |
| Harness getattr hygiene | `python scripts/check_harness_no_getattr.py` |
| Operational L3 | `release_cycles.json` ≥2 + `phase_w_ops_evidence.py --enforce` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer checklists: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix I — Orchestration control plane

**Audience:** Tier-3 application authors, platform engineers, operators.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §7 (Reasoning/planning), §8 (Agent OS), §9 (Orchestration/graph), §10 (Subagents); canon [§42.3](intergrax_runtime_architecture.md#423-hook-system)–[§42.15](intergrax_runtime_architecture.md#4215-agent-handoff-contracts), [§42.43](intergrax_runtime_architecture.md#4243-multi-agent-collaboration-flow-reference).

Intergrax orchestration is **centralized in Tier-1 (Nexus)** — agents own **local** UAEP steps only. Planning, scheduling, graph execution, handoff, retry, HITL, and trace are **composable runtime responsibilities** with typed contracts and hook extension points.

### I.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Single execution stack | `NexusLoop` → `GraphExecutor` → `AgentEngine` → UAEP — no parallel OS per agent |
| Policy-first orchestration | Graph steps still pass `ToolRuntime`, `PolicyEngine`, security middleware |
| Composable-by-default | Inject planners, classifiers, retry policy, middleware via Nexus/bootstrap — not agent code |
| Graph-native delegation | Subagents = `ExecutionGraph` nodes + `DelegationSpec` — not nested harness instances |
| Trace-everything | `RuntimeEvent` per phase; graph callbacks; `ops:handoff` / `ops:planning` filter hints |

### I.2 Orchestration control plane map

```text
Task intake
  └── NexusIntakeRunner          resume · long-running restore · early HITL
        └── NexusPlanningRunner  classify → plan → pre-graph HITL
              └── plan_to_execution_graph(NexusPlan) → ExecutionGraph
                    └── NexusGraphRunner
                          └── GraphExecutor (batches · parallel within batch)
                                ├── AgentRouter (capability / contract routing)
                                ├── ContextManager (SharedTaskContext · assembly options)
                                ├── HandoffCoordinator (§42.15 — graph mutation)
                                ├── RetryEngine + RetryCoordinator
                                └── AgentEngine → UAEPExecutor | RuntimeEngine pipeline

ApplicationEnvironmentProfile (Tier-3)
  ├── orchestration_profile     planner_kind · classifier_kind · retry_policy_name · max_parallel_nodes · max_delegation_depth
  ├── graph_spec                ApplicationGraphSpec → GraphSpecSeedingPlanner (`graph_spec_to_plan.py`)
  ├── execution_mode            strict | balanced | exploratory → Nexus production_mode + policies
  ├── reliability_profile       checkpoint store · scheduler · idempotency
  └── context_profile           assembly budget · RAG flags → per-node AgentContextBundle

Hook layer (Tier-1) — full lifecycle
  └── HookPoint BEFORE/AFTER: intake · classification · planning · agent_selection ·
      context_build · step · tool · validation · decision · interrupt · human · retry ·
      handoff · finalization · trace_persist · memory_write
      → MiddlewarePipeline (priority-ordered handlers; ALLOW | BLOCK | MODIFY | ESCALATE)

Coordination patterns (Phase V-MA)
  └── multi_agent_coordination.py — catalog + select_coordination_pattern(constraints)
      → RuntimeArchitectureGovernanceBridge metadata on runs
```

**Rule:** register agents via `AgentRegistry` — **never** edit `NexusLoop` / `GraphExecutor` for one agent. Extend via hooks, injected collaborators, or Tier-3 profile wiring.

### I.3 Core contracts (typed, inspectable)

| Contract | Module | Role |
|----------|--------|------|
| `NexusPlan` / `PlanStep` | `planning/task_planner.py` | Structured plan before execution |
| `ExecutionGraph` / `ExecutionNode` | `execution/execution_graph.py` | Typed nodes, `depends_on`, `DelegationSpec` |
| `DelegationSpec` | `contracts/delegation.py` | Child agent, isolated memory namespace, context assembly |
| `AgentHandoff` | `contracts/agent_handoff.py` | Nexus-mediated transfer (never direct agent calls) |
| `TaskContextAssemblyOptions` | `contracts/context_assembly.py` | Bounded child context (FULL / SUMMARY_ONLY / …) |
| `AgentExecutionResult` | `contracts/agent_execution_result.py` | Status, decision, artifacts for merge |
| `AgentDecision` | `contracts/agent_decision.py` | COMPLETE · RETRY · INTERRUPT · MODIFY_PLAN · HANDOFF |
| `ValidationResult` | `contracts/validation.py` | Step/node/task validation gates |
| `ApplicationGraphSpec` | `applications/contracts/graph_spec.py` | Declarative multi-agent topology on manifest roster |

Canon reference flow (PM → UX → Legal → Validator → Human): [§42.43](intergrax_runtime_architecture.md#4243-multi-agent-collaboration-flow-reference).

### I.4 Planning strategies (explicit, customizable)

| Strategy | Entry | When used |
|----------|-------|-----------|
| No planner (single agent) | `TaskClassification.SINGLE_AGENT_DEFAULT` | Default lab path |
| Deterministic multi-step | `TaskPlanner._multi_agent_plan`, `_research_pipeline_plan` | Known capability pipelines |
| LLM step planner | `step_planner/` + `RuntimeConfig.step_planner_cfg` | Agent-local tool loops |
| LLM engine planner | `engine_planner_orchestrator.py` | Nexus-level plan from LLM |
| Graph from plan | `plan_to_execution_graph()` | Every Nexus run after planning |

`OrchestrationProfile.planner_kind` / `classifier_kind` resolve via `orchestration_wiring.py` → `build_nexus_loop_from_environment` (ORCH-1 **Done**). Also wired: `retry_policy_name`, `long_running_enabled`, `max_parallel_nodes` (ORCH-3). Kinds: `default` | `engine` (requires `llm_adapter` at factory). Unknown kinds fail fast at bootstrap.

### I.5 Graph execution and merge

| Mechanism | Behavior |
|-----------|----------|
| Topological batches | `ExecutionGraph.batches()` — parallel `asyncio.gather` within batch |
| Sequential failure | First failed node stops graph (unless retry recovers) |
| Retry | `RetryEngine` at node level; `RetryCoordinator` at run level; hooks `BEFORE_RETRY` / `AFTER_RETRY` |
| Handoff | `AgentDecision` / `resolve_handoff_from_execution` → `HandoffCoordinator` inserts node |
| Delegation | `ExecutionNode.delegation: DelegationSpec` → isolated `MemoryView` namespace |
| Merge | `FinalResponseComposer.compose_summary(executions)` — deterministic summary merge |
| Checkpoint skip | `apply_runtime_checkpoint_to_graph` — resume long runs |
| Cancel | `CancellationCoordinator` — marks pending nodes cancelled |

**Concurrency:** `OrchestrationProfile.max_parallel_nodes` caps parallel nodes per graph batch (`GraphExecutor` semaphore). Tenant-level cap remains on `RuntimeEngine` (`max_parallel_per_tenant`).

### I.6 Subagent / delegation semantics (R-Delegate — Done)

| Harness subagent | Intergrax |
|------------------|-----------|
| Spawn child with own context | `ExecutionGraph` node + `DelegationSpec` |
| Isolated memory | `task_id/delegation/{node_id}/` via `delegation_memory.py` |
| Parent tool policy | `inherit_tool_policy=True` → intersect with child `AgentContract.allowed_tools` |
| Trace | `parent_run_id`, `parent_node_id` on delegation metadata; `ops:handoff` events |

**Forbidden:** Tier-2 agent importing and calling another agent. **Required:** Nexus schedules child after plan edge or handoff.

### I.7 Customization surfaces (full control without forking Nexus)

| Surface | How to customize |
|---------|------------------|
| **Hooks** | Register `MiddlewarePipeline` handlers on any `HookPoint` at Tier-3 bootstrap |
| **Runtime plugins** | `RuntimePlugin.register(bus, hooks, policy)` — metrics/persistence middleware (not catalog EP) |
| **Planner / classifier kinds** | `OrchestrationProfile.planner_kind` / `classifier_kind` via `orchestration_wiring.py` → `build_nexus_loop_from_environment` |
| **Planner (direct inject)** | Pass custom planner/classifier implementing `NexusTaskPlannerProtocol` / `NexusTaskClassifierProtocol` to `NexusLoop(...)` |
| **Graph executor** | Inject `GraphExecutor` with custom `AgentEngine`, `HandoffCoordinator`, `ContextManager` |
| **Coordination pattern** | `select_coordination_pattern(PlanningConstraints)` — metadata + planning guidance (V-MA) |
| **Application graph** | `AgentGraph().add(...).edge(...).delegates_to(...).build()` → `graph_spec` on env profile |
| **Execution mode** | `ExecutionMode.STRICT` → `production_mode=True` on Nexus + stricter agent routability |
| **Agent contract** | `capabilities`, `max_steps`, `allowed_tools` — routing and tool scope per agent |

Agent-local tool orchestration: `RuntimeConfig.tool_planner` (`ToolPlannerProtocol`) + `CatalogToolPlanner` — separate from Nexus graph, still through `ToolRuntime`.

### I.8 Observability for orchestration runs

| Signal | Event / trace |
|--------|----------------|
| Plan created | `PLAN_CREATED` · `ops:planning` |
| Plan failed | `PLAN_FAILED` (engine planner parse/LLM) |
| Node start/complete | Graph trace callbacks → Nexus trace DB |
| Handoff | `HANDOFF_INITIATED` / `HANDOFF_COMPLETED` · `ops:handoff` |
| Retry | `RETRY_SCHEDULED` · `ops:retry` |
| HITL pause | `HUMAN_APPROVAL_REQUESTED` · `ops:hitl` |

**Inspect (lab):** `GET /debug/tasks/{id}/trace?include_runtime=true` · event stream §42.1.5 filter hints.

### I.9 What agents (Tier-2) must not do

| Do not | Do instead |
|--------|------------|
| Call another agent directly | Emit `AgentDecision` / handoff; let Nexus route |
| Build private execution graphs | Declare `capabilities`; let `TaskPlanner` + registry route |
| Implement retry loops over adapters | Return `AgentDecision.RETRY`; runtime `RetryEngine` executes |
| Own global task lifecycle | UAEP steps only; Nexus owns `TaskLifecycle` |
| Spawn nested Nexus / harness | Use `DelegationSpec` on graph node |

### I.10 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Graph executor (handoff, retry) | `pytest tests/integration/runtime/test_graph_executor_handoff_retry.py -m gate` |
| Graph coverage | `pytest tests/unit/runtime/execution/ -m gate` |
| Delegation memory | `pytest tests/unit/runtime/task_memory/ -m gate -k delegation` |
| Multi-agent patterns (V-MA) | `pytest tests/unit/runtime/architecture/test_multi_agent_coordination.py -m gate` |
| Nexus decomposition | `pytest tests/unit/runtime/nexus/ -m gate` |
| No agent branches in NexusLoop | `python scripts/check_harness_no_getattr.py` + code review |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layers §7–§10: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix J — Tools & skills control plane

**Audience:** Tier-3 application authors, extension authors, platform engineers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §11 (Tool layer), §12 (Skill layer); canon [§7.1.6](intergrax_runtime_architecture.md#716-tool-catalog)–[§7.1.8](intergrax_runtime_architecture.md#718-skill-catalog).

Intergrax separates **Integration → Tool → Skill → Agent** (Tier-0 → Tier-2). Tools are atomic, policy-governed operations; skills are composable capability packs (tool_ids + prompt instructions + policy fragments). Agents declare `skill_ids` on `AgentContract` — never copy tool lists or vendor SDK calls into agent steps.

### J.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Atomic tools | One `ToolContract` = one operation; no workflow-sized tools |
| Policy-first invocation | Every call through `ToolRuntime` + `ToolScopePolicy` + security middleware |
| Composable skills | `SkillManifest` merges tool allow-lists and prompt fragments — not agents |
| Tier separation | Agents never import `integrations/providers/`; Tier-3 wires slugs via profiles |
| Typed wiring | `ToolProfile`, `SkillProfile`, `SkillResolverProtocol` — no `getattr`/`setattr` on harness paths |

### J.2 Tools & skills control plane map

```text
ApplicationEnvironmentProfile (Tier-3)
  ├── tool_profile              enabled tool_ids / bundles · sandbox flags
  ├── skill_profile             enabled skill bundles
  └── integration_profile       backend slugs → ToolWiringContext

wire_application_environment()
  ├── build_application_tool_wiring()   → ToolRegistry + ToolWiringContext
  ├── build_application_skill_wiring()    → SkillRegistry
  ├── EnvironmentSkillToolConsistencyCheck (roster vs env profiles)
  └── ApplicationBuildContext             tool/skill profiles + registries

materialize_runtime_config() / build_runtime_context_from_environment()
  ├── catalog_runtime_bridge.py           tool_profile · skill_profile · tool_wiring_context
  └── memory_runtime_bridge.py            context/memory toggles (MEM)

Agent execution (Tier-1)
  ├── AgentRegistry + SkillResolver         contract.skills → allowed_tools merge
  ├── CatalogToolPlanner + ToolRuntime    policy-checked invocation
  └── RuntimeEvent bus                      tool/skill telemetry

build_harness_host_runtime()
  └── build_nexus_loop_from_environment() + resolve_llm_adapter() (engine planner)
```

**Rule:** register tools/skills in Tier-0 catalogs and enable them on `ApplicationEnvironmentProfile` — **never** create agent-local tool registries.

### J.3 Core contracts (typed, inspectable)

| Contract | Module | Role |
|----------|--------|------|
| `ToolContract` | `tools/core/contracts.py` | Atomic tool schema, risk, timeout |
| `ToolProfile` | `tools/registry/profile.py` | Enabled tools/bundles for a host |
| `ToolWiringContext` | `tools/registry/wiring.py` | Integration slug → provider wiring |
| `ToolPlannerProtocol` | `runtime/nexus/tools/tool_planner_protocol.py` | Agent-local tool loop planning |
| `SkillManifest` | `skills/core/contracts.py` | skill_id, tool_ids, prompts, policy fragment |
| `SkillProfile` | `skills/registry/profile.py` | Enabled skill bundles for a host |
| `SkillResolverProtocol` | `skills/resolver.py` | Resolve skill_ids → `ResolvedSkillPack` |
| `RuntimeConfig.tool_profile` / `skill_profile` | `runtime/nexus/config.py` | Runtime catalog snapshot (TS-1) |

### J.4 Customization surfaces (full control without forking runtime)

| Surface | How to customize |
|---------|------------------|
| **Tool profile** | `ApplicationEnvironmentProfile.tool_profile` — enable tool_ids or bundles |
| **Skill profile** | `ApplicationEnvironmentProfile.skill_profile` — enable skill bundles |
| **Integration backends** | `IntegrationProfile` + `ToolWiringContext.from_integration_profile()` |
| **Tool scope policy** | `RuntimePolicyBundle.tool_access` → `RuntimeConfig.tool_scope_policy` |
| **Sandbox / shadow** | `tool_profile_with_sandbox()` + `wire_sandbox_sessions()` at bootstrap |
| **Plugin catalogs** | `ToolPlugin` / `SkillPlugin` entry points (Phase P-Ext **Done**) |
| **Agent contract** | `skills: list[SkillManifest]` + `extra_tools` — merged at registry bind time |
| **Conformance** | `EnvironmentSkillToolConsistencyCheck` — roster tools/skills ⊆ environment |

Agent-local tool orchestration: `RuntimeConfig.tool_planner` (`CatalogToolPlanner`) + `tools_mode` — still through `ToolRuntime`, separate from Nexus graph planning (Appendix I).

### J.5 Runtime bridge (TS-1 — Done)

| Bridge | Module | Maps |
|--------|--------|------|
| Catalog → RuntimeConfig | `catalog_runtime_bridge.py` | `tool_profile`, `skill_profile`, `tool_wiring_context` |
| Environment → RuntimeConfig | `memory_runtime_bridge.py` | memory/context toggles |
| Host → Nexus LLM | `harness_host_runtime.py` + `llm_resolver.py` | `resolve_llm_adapter(env)` for `planner_kind=engine` (TS-2) |

Wired `ApplicationBuildContext` profiles **override** raw environment defaults (sandbox-adjusted tools).

### J.6 What agents (Tier-2) must not do

| Do not | Do instead |
|--------|------------|
| Import vendor SDKs in `agents/` | Declare `allowed_tools`; wire integration in Tier-3 |
| Register tools inside agent package | Add `ToolPlugin` or catalog bundle; enable in `tool_profile` |
| Model workflows as one giant tool | Create **Skill** pack + UAEP steps |
| Copy prompt + tool lists per agent | Reuse `skill_ids` from [Skill Library](SKILLS.md) |
| Bypass `ToolRuntime` | Return `ToolRequest`; runtime invokes with policy + trace |
| Use `use_rag` / `use_websearch` booleans | Pass explicit `tool_ids` (`rag.retrieve`, `websearch.query`) |

### J.7 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Catalog runtime bridge | `pytest tests/unit/applications/test_catalog_runtime_bridge.py -m gate` |
| Harness host LLM wiring | `pytest tests/unit/applications/test_harness_host_runtime_llm.py -m gate` |
| Skill resolver | `pytest tests/unit/skills/test_skill_resolver.py -m gate` |
| Tool runtime / policy | `pytest tests/unit/runtime/nexus/tools/ -m gate` |
| Environment conformance | `pytest tests/unit/applications/ -m gate -k conformance` |
| Plugin catalogs | `python scripts/check_plugin_catalog.py` |
| Legacy boolean flags | `python scripts/check_legacy_tool_plan_booleans.py` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layers §11–§12: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix K — Integration & RAG control plane

**Audience:** Tier-3 application authors, extension authors, platform engineers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §13 (Integration), §14 (RAG); canon [§7.1](intergrax_runtime_architecture.md#71-integration-library)–[§7.1.5](intergrax_runtime_architecture.md#715-integration-profile); memory/RAG naming: [Appendix G](#appendix-g--memory--rag-naming-phase-q).

Integrations are **backend/provider adapters** (Tier-0). RAG is a **full retrieval layer** composed from integration vector stores + embedding/rerank managers — not agent-local vector queries. Agents stay vendor-agnostic; Tier-3 selects providers via `IntegrationProfile`.

### K.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Category contracts | Every integration slot maps to `IntegrationCategory` + stable contract |
| Vendor isolation | SDK imports only in `integrations/providers/` boundary modules |
| Profile-first wiring | Tier-3 resolves providers via `IntegrationProfile.resolve(category)` |
| Single retrieval path | `RetrievalService` + `rag.retrieve` tool — no agent `vectorstore.query` |
| Health at bootstrap | `probe_integration_profile_health` on environment wire |
| Typed bridges | `integration_runtime_bridge`, `rag_runtime_bridge` — no dynamic attribute access |

### K.2 Integration & RAG control plane map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── integration_profile          category slots (relational_store, vector_store, …)

wire_application_environment()
  ├── bootstrap_application_integration_catalog()
  ├── probe_integration_profile_health()     → ApplicationEnvironmentWiring.integration_health
  ├── resolve_rag_stack_for_environment()    when context_profile.enable_rag
  ├── ToolWiringContext.from_integration_profile()
  └── build_application_tool_wiring()          RAG managers injected into tool context

materialize_runtime_config() / build_runtime_context_from_environment()
  ├── integration_runtime_bridge.py          integration_profile on RuntimeConfig
  ├── rag_runtime_bridge.py                  vectorstore / retrieval_service / RagProfile
  ├── memory_runtime_bridge.py               context toggles (MEM)
  └── catalog_runtime_bridge.py              tool/skill profiles (TS)

Agent execution (Tier-1)
  ├── RuntimeConfig.integration_profile      memory backends, notifications, vector store
  ├── Nexus ContextBuilder + RetrievalService canonical RAG path
  └── Catalog tool rag.retrieve              policy-checked retrieval
```

**Rule:** declare integrations on `IntegrationProfile` in Tier-3 — **never** import `integrations/providers/` from `agents/`.

### K.3 Core contracts (typed, inspectable)

| Contract | Module | Role |
|----------|--------|------|
| `IntegrationProfile` | `integrations/registry/profile.py` | Typed provider selection per category |
| `IntegrationCategory` | `integrations/contracts/base.py` | Category enum + profile field map |
| `IntegrationHealthProbe` | `integrations/contracts/health_probe.py` | Optional provider health() |
| `HealthStatus` | `integrations/contracts/base.py` | Bootstrap probe result |
| `RagStack` | `rag/bootstrap/rag_stack_bootstrap.py` | Composed RAG managers + `RetrievalService` |
| `RagProfile` | `rag/profiles/rag_profile.py` | Retrieval modes, top-k, rerank toggles |
| `RetrievalService` | `rag/retrieval/retrieval_service.py` | Canonical retrieval orchestration |
| `RuntimeConfig.integration_profile` | `runtime/nexus/config.py` | Runtime integration snapshot (INT-1) |

### K.4 Customization surfaces

| Surface | How to customize |
|---------|------------------|
| **Integration profile** | `ApplicationEnvironmentProfile.integration_profile` or manifest default |
| **Presets** | `IntegrationProfile.lab_harness_preset()`, `legal_stack()`, `research_stack()` |
| **Per-slug options** | `IntegrationProfile.options` dict (e.g. sqlite `data_dir`) |
| **RAG enable** | `ContextProfile.enable_rag` on environment → stack bootstrap |
| **RAG tuning** | `RagProfile` env vars or explicit profile passed to `create_default_rag_stack` |
| **Vector store** | `vector_store` slot on profile (falls back to in-memory when unset) |
| **Plugin catalogs** | Integration entry points (Phase P-Ext **Done**) |
| **Health probes** | Implement `IntegrationHealthProbe` on provider; run via `probe_integration_profile_health` |

### K.5 Runtime bridges (INT + RAG — Done)

| Bridge | Module | Maps |
|--------|--------|------|
| Integration → RuntimeConfig | `integration_runtime_bridge.py` | `integration_profile` |
| Integration health | `integration_health_wiring.py` | bootstrap `HealthStatus` tuple |
| RAG → RuntimeConfig | `rag_runtime_bridge.py` | managers + `RetrievalService` + `RagProfile` |
| Memory backends | `memory_wiring.py` | sqlite/mongo session + LTM from `integration_profile` |

Wired `ApplicationBuildContext.integration_profile` **overrides** raw environment defaults.

### K.6 What agents (Tier-2) must not do

| Do not | Do instead |
|--------|------------|
| Import `redis`, `boto3`, `psycopg` in agents | Declare tools; wire integration in Tier-3 profile |
| Call `vectorstore.query` directly | Use `rag.retrieve` tool or Nexus `ContextBuilder` |
| Store integration config in agent dir | `IntegrationProfile` on environment/manifest |
| Treat LLM provider as Integration slug | Use `LLMProfile` / `resolve_llm_adapter` (LLM Adapter layer) |
| Use legacy `use_rag` plan booleans | Explicit `tool_ids` (`rag.retrieve`) — gateway uses `tool_invocation_plan_from_capability_payload` (Phase LEG **Done**) |

### K.7 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Integration runtime bridge | `pytest tests/unit/applications/test_integration_runtime_bridge.py -m gate` |
| Integration health wiring | `pytest tests/unit/applications/test_integration_health_wiring.py -m gate` |
| RAG runtime bridge | `pytest tests/unit/applications/test_rag_runtime_bridge.py -m gate` |
| Harness lab health | `pytest tests/unit/integrations/test_harness_lab_health.py -m gate` |
| RAG tool catalog | `pytest tests/unit/tools/providers/rag/ -m gate` |
| Vendor import gates | `python scripts/check_agents_vendor_imports.py` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layers §13–§14: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix L — Context engineering control plane

**Audience:** Tier-3 application authors, platform engineers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §16; canon [§28](intergrax_runtime_architecture.md#281-context-engineering); memory/RAG naming: [Appendix G](#appendix-g--memory--rag-naming-phase-q).

Context engineering is a **first-class Nexus concern** — budgeted assembly, provenance, trimming telemetry, and deterministic pipelines. Agents do not hand-build prompts; `ContextManager` + `ContextBuilder` assemble bounded context from task, memory, RAG, tools, and graph outputs.

### L.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Central assembly | `ContextManager` (graph nodes) + `ContextBuilder` (runtime turns) |
| Budget-first | `ContextBudgetPolicy` on `ContextProfile` → trim + `CONTEXT_TRIMMED` events |
| Provenance | `ContextProvenance` on `AgentContextBundle` — source lineage per fragment |
| Environment-driven | `ContextProfile` on `ApplicationEnvironmentProfile` — not agent code |
| Typed bridges | `context_runtime_bridge`, `context_wiring` — explicit Tier-3 → Tier-1 mapping |

### L.2 Context control plane map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── context_profile            budget_policy · assembly_options · decision · RAG flags

materialize_runtime_config()
  ├── context_runtime_bridge.py   → RuntimeConfig.context_budget_policy
  │                                 task_context_assembly_options · run_budget
  └── memory_runtime_bridge.py    memory toggles (MEM)

wire_application_environment() / build_harness_host_runtime()
  └── context_wiring.py           resolve_context_manager_from_environment()
        └── build_nexus_loop_from_environment() → NexusLoop.context_manager

Task intake (Tier-3 hosts)
  └── merge_task_context_options_from_environment()   overlay assembly_options on TaskExecutionOptions

Nexus graph execution (Tier-1)
  └── ContextManager.build_agent_context()   provenance · summary tiers · budget trim
        └── record_context_assembly() on RuntimeEventBus
```

### L.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `ContextProfile` | `environment_profile.py` | Tier-3 context defaults |
| `ContextBudgetPolicy` | `context/context_budget.py` | Char/token limits + trim |
| `TaskContextAssemblyOptions` | `contracts/context_assembly.py` | Per-task assembly rules |
| `ContextManager` | `context/context_manager.py` | Graph-node context bundles |
| `AgentContextBundle` | `context/context_manager.py` | Bounded message + provenance |
| `ContextProvenance` | `context/context_models.py` | Source lineage metadata |

### L.4 Customization surfaces

| Surface | How to customize |
|---------|------------------|
| **Context profile** | `ApplicationEnvironmentProfile.context_profile` |
| **Budget** | `ContextProfile.budget_policy` → `RunBudget` via bridge |
| **Assembly tier** | `ContextProfile.assembly_options.summary_tier` (FULL / SUMMARY_ONLY / …) |
| **Memory in context** | `ContextDecisionProfile.max_memory_entries_in_context` |
| **RAG/web flags** | `ContextProfile.enable_rag` / `enable_websearch` |
| **Graph context** | `ContextManager` injected via `build_nexus_loop_from_environment` |
| **Task intake** | `merge_task_context_options_from_environment()` at host boundary |

### L.5 Runtime bridges (CTX — Done)

| Bridge | Module | Maps |
|--------|--------|------|
| Context → RuntimeConfig | `context_runtime_bridge.py` | budget, assembly, decision, run_budget |
| Context → Nexus | `context_wiring.py` | `ContextManager` + task option merge |
| Memory → RuntimeConfig | `memory_runtime_bridge.py` | memory toggles only |

### L.6 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Context runtime bridge | `pytest tests/unit/applications/test_context_runtime_bridge.py -m gate` |
| Context wiring | `pytest tests/unit/applications/test_context_wiring.py -m gate` |
| ContextManager v2 | `pytest tests/unit/runtime/nexus/context/ -m gate` |
| Memory + context round-trip | `pytest tests/unit/applications/test_memory_profile_runtime_bridge.py -m gate` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §16: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix M — Prompt registry control plane

**Audience:** Tier-3 application authors, platform engineers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §17; governance schema: V-REM-PE.1/PE.2 (**Done**).

Prompts are **versioned YAML assets** — not inline strings in agent code. Tier-3 hosts declare catalog location via `PromptProfile`; Nexus prompt builders resolve through `YamlPromptRegistry`.

### M.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Asset-first | Prompts live under versioned YAML catalogs (`prompt_id/1.yaml`, `stable.yaml`) |
| Governance metadata | `PromptMeta` carries `owner_team`, `risk_tier`, version fields — validated on load |
| Environment-driven | `PromptProfile` on `ApplicationEnvironmentProfile` — not agent imports |
| Typed bridges | `prompt_runtime_bridge`, `prompt_wiring` — explicit Tier-3 → Tier-1 mapping |
| Injectable registry | `RuntimeContext.build(prompt_registry=…)` — builders share one registry instance |

### M.2 Prompt control plane map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── prompt_profile              catalog_path · load_on_startup

materialize_runtime_config()
  └── prompt_runtime_bridge.py    → RuntimeConfig.prompt_catalog_path

wire_application_environment() / build_runtime_context_from_environment()
  └── prompt_wiring.py            resolve_prompt_registry()
        ├── ApplicationBuildContext.prompt_registry
        └── RuntimeContext.build(prompt_registry=…)

Nexus prompt builders (Tier-1)
  └── DefaultRagPromptBuilder · DefaultUserLongTermMemoryPromptBuilder · …
        └── prompt_registry.resolve() / resolve_localized()
```

### M.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `PromptProfile` | `environment_profile.py` | Tier-3 catalog selection |
| `PromptRegistryProtocol` | `prompts/registry/prompt_registry_protocol.py` | Typed resolve surface |
| `YamlPromptRegistry` | `prompts/registry/yaml_registry.py` | Production YAML catalog loader |
| `PromptMeta` | `prompts/schema/prompt_schema.py` | Owner/risk/version governance |

### M.4 Customization surfaces

| Surface | How to customize |
|---------|------------------|
| **Catalog path** | `ApplicationEnvironmentProfile.prompt_profile.catalog_path` (default: `prompts/`) |
| **Eager load** | `PromptProfile.load_on_startup` — passed to `YamlPromptRegistry.create_default(load=…)` |
| **Runtime fallback** | `RuntimeConfig.prompt_catalog_path` when `prompt_registry` not injected explicitly |
| **Pin / version** | `PromptPinConfig` on `registry.resolve(prompt_id, pin=…)` |

### M.5 Runtime bridges (PE — Done)

| Bridge | Module | Maps |
|--------|--------|------|
| Prompt → RuntimeConfig | `prompt_runtime_bridge.py` | `catalog_path` → `prompt_catalog_path` |
| Prompt → registry | `prompt_wiring.py` | `PromptProfile` → `YamlPromptRegistry` |
| Environment wire | `environment_wiring.py` | `ApplicationBuildContext.prompt_registry` |

### M.6 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Prompt runtime bridge | `pytest tests/unit/applications/test_prompt_runtime_bridge.py -m gate` |
| Prompt wiring | `pytest tests/unit/applications/test_prompt_wiring.py -m gate` |
| Nexus registry injection | `pytest tests/unit/runtime/nexus/runtime_steps/test_tools_step_prompt_registry.py -m gate` |
| PromptMeta governance | `pytest tests/unit/prompts/test_prompt_governance_meta.py -m gate` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §17: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix N — Agent assembly control plane

**Audience:** Tier-2 agent authors, platform engineers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §18; ideal model §17 in [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md).

Agents are **composable capability units** — not monolithic orchestrators. Assembly happens through declarative `AgentContract` metadata, skill packs, and registry-time resolution into `allowed_tools`.

### N.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Contract-first | `AgentContract` carries id, capabilities, skills, lifecycle — no runtime edits |
| Skill composition | Authors declare `skills` (`SkillManifest`) + optional `extra_tools` (`ToolContract`) |
| Registry resolution | `AgentRegistry.register` merges skills → `allowed_tools`; authors keep `allowed_tools=[]` |
| Bounded local loop | UAEP steps on agent; Nexus owns global orchestration |
| Lifecycle governance | `AgentLifecycleState` + `evaluate_agent_routing` gate production selection |
| Register-time validation | `agent_assembly_resolver` fails fast on incomplete contracts |

### N.2 Agent assembly map

```text
agents/<slug>/contract.py (Tier-2)
  └── AgentContract
        ├── capabilities[]          routing / discovery ids
        ├── skills[]                SkillManifest packs from catalog
        ├── extra_tools[]           optional ToolContract references
        └── lifecycle_state         development → staging → production → deprecated → retired

AgentRegistry.register (Tier-1)
  └── agent_assembly_resolver       metadata + lifecycle validation
  └── resolve_contract_tools        SkillResolver → allowed_tools
  └── evaluate_agent_routing        production / deprecated / retired gating
```

### N.3 Lifecycle state mapping

Audit vocabulary maps to the canonical enum in `agent_lifecycle_state.py`:

| Audit term | `AgentLifecycleState` | Routing notes |
|------------|----------------------|---------------|
| draft | `development` | Lab / non-production routing |
| experimental | `staging` | Allowed in production_mode selection |
| certified | `production` | Default for GA agents; set `production_eligible` + owner metadata when shipping |
| deprecated | `deprecated` | Not routable |
| retired | `retired` | Not routable; may remain registered for introspection |

### N.4 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `AgentContract` | `contracts/agent_contract_meta.py` | Declarative agent metadata |
| `AgentLifecycleState` | `contracts/agent_lifecycle_state.py` | Lifecycle enum |
| `SkillManifest` | `skills/core/contracts.py` | Skill pack declaration |
| `SkillResolverProtocol` | `skills/resolver.py` | Typed skill → tool resolution |
| `AgentAssemblyValidationResult` | `runtime/registry/agent_assembly_resolver.py` | Register-time validation outcome |
| `AgentRoutingDecision` | `runtime/registry/agent_routing_policy.py` | Nexus selection gating |

### N.5 Authoring rules

| Rule | Enforcement |
|------|-------------|
| Declare `capabilities` | `validate_contract_metadata` at register |
| Use `skills` / `extra_tools` for tools | Do not set author-time `allowed_tools` |
| `production_eligible=True` | Requires `owner_team`, `owner_contact`, `runbook_ref` |
| Reuse across applications | Agent logic in `agents/`; wiring in Tier-3 `manifest.py` |
| No private tool registry | Tools from catalog + `ToolProfile` on application host |

### N.6 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Assembly resolver | `pytest tests/unit/runtime/registry/test_agent_assembly_resolver.py -m gate` |
| Skill → allowed_tools | `pytest tests/unit/runtime/registry/test_agent_registry_skills.py -m gate` |
| Author-time allowed_tools ban | `python scripts/check_agent_skill_resolution.py` |
| Lifecycle routing | `pytest tests/unit/runtime/architecture/test_agent_routing_policy.py -m gate` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §18: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix O — Registry architecture control plane

**Audience:** Tier-3 application authors, platform engineers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §19; capability graph: canon §53.2 · Phase V-CG **Done**.

Registries are **runtime primitives** — not optional documentation. Tier-3 hosts materialize catalog registries through `wire_application_environment`; Nexus and `AgentRegistry` resolve artifacts from typed snapshots.

### O.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Registry-first | Tools, skills, prompts, policies resolve through registries — not ad-hoc imports |
| Environment-driven | `ToolProfile` / `SkillProfile` / `PromptProfile` on `ApplicationEnvironmentProfile` |
| Typed snapshot | `HarnessRegistrySnapshot` captures wired handles for conformance audits |
| Register-time validation | `registry_assembly_resolver` fails fast when profiles require missing registries |
| Capability graph | `CapabilityGraph` links integration → tool → skill → agent → application (V-CG) |
| Plugin catalogs | `bootstrap_catalogs()` + entry points (Phase P-Ext **Done**) |

### O.2 Registry control plane map

```text
ApplicationEnvironmentProfile (Tier-3)
  ├── integration_profile         Integration catalog selection
  ├── tool_profile                ToolRegistry materialization
  ├── skill_profile               SkillRegistry materialization
  ├── prompt_profile              YamlPromptRegistry materialization
  └── policy_rules_profile        RuntimePolicyBundle composition

wire_application_environment()
  ├── tool_wiring.py              build_application_tool_wiring → ToolRegistry
  ├── skill_wiring.py             build_application_skill_wiring → SkillRegistry
  ├── prompt_wiring.py            resolve_prompt_registry → YamlPromptRegistry
  ├── policy_wiring.py            wire_policy_bundle → RuntimePolicyBundle
  └── registry_wiring.py          resolve_registry_snapshot → HarnessRegistrySnapshot
        └── registry_assembly_resolver.py   profile ↔ registry conformance

AgentRegistry.register (Tier-1)
  └── SkillResolver + resolve_contract_tools → allowed_tools
```

### O.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `ToolRegistry` | `tools/registry/runtime.py` | Runtime tool catalog |
| `SkillRegistry` | `skills/registry/runtime.py` | Runtime skill catalog |
| `PromptRegistryProtocol` | `prompts/registry/prompt_registry_protocol.py` | Typed prompt resolve |
| `AgentRegistry` | `runtime/registry/agent_registry.py` | Agent discovery + contracts |
| `RuntimePolicyBundle` | `runtime/policy/policy_bundle.py` | Policy composition |
| `HarnessRegistrySnapshot` | `applications/_shared/registry_snapshot.py` | Wired registry handles |
| `RegistrySnapshotProtocol` | `applications/_shared/registry_snapshot_protocol.py` | Snapshot audit surface |
| `CapabilityGraph` | `runtime/architecture/capability_graph.py` | Dependency / impact graph |

### O.4 Artifact coverage (audit §19)

| Artifact | Registry / profile | Resolution path |
|----------|-------------------|-----------------|
| Agent | `AgentRegistry` | `build_application_registry` |
| Tool | `ToolRegistry` | `build_application_tool_wiring` |
| Skill | `SkillRegistry` | `build_application_skill_wiring` |
| Policy | `RuntimePolicyBundle` | `wire_policy_bundle` |
| Prompt | `YamlPromptRegistry` | `resolve_prompt_registry` |
| Integration | `IntegrationProfile` | `bootstrap_application_integration_catalog` |
| Evaluation | `OnlineEvaluationRegistry` | Phase W-OPS / V-EVAL |

### O.5 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Registry snapshot wiring | `pytest tests/unit/applications/test_registry_wiring.py -m gate` |
| Agent skill resolution | `python scripts/check_agent_skill_resolution.py` |
| Host registry resolution | `python scripts/check_harness_registry_resolution.py` |
| Capability graph guard | `uv run python scripts/phase_v_capability_graph_guard.py --enforce` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §19: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix P — Capability graph control plane

**Audience:** Tier-3 application authors, platform engineers, release/ops.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §20; Phase V-CG **Done**; environment closeout Phase CG.

Dependencies between integrations, tools, skills, policies, agents, and applications must be **explicit, typed, and analyzable** for blast-radius and compatibility checks.

### P.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Typed graph | `CapabilityNodeType` + `CapabilityEdgeType` on `CapabilityGraph` |
| Catalog baseline | `build_catalog_capability_graph()` from registries + reference manifests |
| Environment slice | `resolve_environment_capability_graph()` — host-scoped subgraph |
| Blast radius | `build_capability_impact_report()` — downstream node sets |
| Compatibility | `evaluate_capability_graph_compatibility()` — release guard |
| Wire-time validation | `capability_graph_assembly_resolver` at `wire_application_environment` |

### P.2 Capability graph map

```text
Catalog baseline (Tier-0/1)
  └── build_catalog_capability_graph()
        ├── integration:*  → tool:*  (depends_on)
        ├── skill:*        → tool:*  (depends_on)
        ├── agent:*        → skill:* / tool:*  (depends_on)
        └── application:*  → agent:*  (depends_on)

wire_application_environment() (Tier-3)
  └── capability_graph_wiring.py
        ├── resolve_environment_capability_graph()
        └── capability_graph_assembly_resolver.py

Release / CI
  └── phase_v_capability_graph_guard.py --enforce
  └── check_harness_capability_graph_wiring.py
```

### P.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `CapabilityGraph` | `runtime/architecture/capability_graph.py` | Typed nodes + edges |
| `CapabilityGraphViewProtocol` | `applications/_shared/capability_graph_protocol.py` | Environment graph audit surface |
| `EnvironmentCapabilityGraphView` | `applications/_shared/capability_graph_wiring.py` | Host-scoped subgraph |
| `build_capability_lineage_report` | `capability_graph_lineage.py` | Upstream/downstream lineage |
| `build_capability_impact_report` | `capability_graph_lineage.py` | Blast-radius analysis |
| `evaluate_capability_graph_compatibility` | `capability_graph_compatibility.py` | Baseline diff guard |

### P.4 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Environment graph wiring | `pytest tests/unit/applications/test_capability_graph_wiring.py -m gate` |
| Host graph materialization | `python scripts/check_harness_capability_graph_wiring.py` |
| Catalog compatibility guard | `uv run python scripts/phase_v_capability_graph_guard.py --enforce` |
| Lineage / impact reports | `build/architecture_hardening/capability_*_report.json` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §20: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix Q — Observability control plane closeout

**Audience:** Tier-3 application authors, platform engineers, release/ops.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §21; Phase OBS **Done**; complements [Appendix H §H.5](#h5-observability--what-is-mandatory-vs-optional) (mandatory vs optional signals).

Tier-3 hosts must materialize Nexus trace stores, runtime event journals, and integration observability backends from typed `ObservabilityProfile` — not ad-hoc `wire_nexus_observability()` calls in host factories.

### Q.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Profile-driven | `ObservabilityProfile` on `ApplicationEnvironmentProfile` |
| Wire-time validation | `observability_assembly_resolver` at `build_harness_host_runtime` |
| Typed bridge | `ObservabilityWiringOptions` maps profile → `wire_nexus_observability` flags |
| Integration coupling | `otel_enabled` requires `IntegrationProfile.observability_backend` |
| Single host path | `wire_application_observability` + `assert_observability_assembly_valid` |

### Q.2 Observability wiring map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── observability_profile
        ├── trace_sqlite_enabled  → SQLite trace + runtime event journal
        ├── otel_enabled          → IntegrationProfile.observability_backend (otel, prometheus, …)
        ├── metrics_plugins_enabled → platform_wiring LLM/RAG plugins (lab default)
        └── debug_surface_override  → Tier-3 debug API posture

wire_application_observability() (Tier-3)
  └── observability_runtime_bridge.py
        ├── resolve_observability_wiring_options()
        └── apply_observability_profiles_from_environment() → RuntimeConfig

build_harness_host_runtime() (Tier-3)
  └── observability_wiring.py + observability_assembly_resolver.py
        └── NexusObservabilityStores → NexusLoop(trace_store, runtime_events_db_path)

Release / CI
  └── check_harness_observability_wiring.py
  └── test_harness_observability_wiring.py
```

### Q.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `ObservabilityProfile` | `contracts/environment_profile.py` | Author-facing observability flags |
| `ObservabilityWiringOptions` | `observability_runtime_bridge.py` | Profile → wiring flags |
| `ApplicationObservabilityWiring` | `observability_wiring.py` | Resolved stores + options |
| `ObservabilityAssemblyValidationResult` | `observability_assembly_resolver.py` | Wire-time conformance |
| `NexusObservabilityStores` | `runtime/nexus/observability_wiring.py` | Trace + event journal handles |

### Q.4 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Profile → wiring bridge | `pytest tests/unit/applications/test_harness_observability_wiring.py -m gate` |
| Host observability materialization | `python scripts/check_harness_observability_wiring.py` |
| Lab OTLP / debug APIs | [`HARNESS_ENVIRONMENT.md`](HARNESS_ENVIRONMENT.md#otlp--observability-s-ops2) |
| Mandatory vs optional signals | [Appendix H §H.5](#h5-observability--what-is-mandatory-vs-optional) |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §21: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix R — Reliability control plane closeout

**Audience:** Tier-3 application authors, platform engineers, release/ops.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §22; Phase REL **Done**; complements H-APP `ReliabilityProfile` (§H.2).

Tier-3 hosts must materialize idempotency stores, circuit breaker thresholds, and long-running coherence from typed `ReliabilityProfile` — not ad-hoc store construction in host factories.

### R.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Profile-driven | `ReliabilityProfile` on `ApplicationEnvironmentProfile` |
| Wire-time validation | `reliability_assembly_resolver` at `build_harness_host_runtime` |
| Typed bridge | `ReliabilityWiringOptions` maps profile → stores and breaker config |
| Orchestration coupling | `long_running_scheduler_enabled` requires `orchestration_profile.long_running_enabled` |
| Integration resilience | `circuit_breaker_failure_threshold` drives `IntegrationCircuitBreakerConfig` on health probes |

### R.2 Reliability wiring map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── reliability_profile
        ├── idempotency_enabled        → SQLite or in-memory IdempotencyStore
        ├── circuit_breaker_failure_threshold → IntegrationCircuitBreakerConfig
        ├── checkpoint_interval_steps  → long-running checkpoint cadence (Nexus)
        └── long_running_scheduler_enabled → requires orchestration long_running + checkpoint store

wire_application_reliability() (Tier-3)
  └── reliability_runtime_bridge.py
        ├── resolve_reliability_wiring_options()
        └── apply_reliability_profiles_from_environment() → RuntimeConfig.idempotency_store

wire_application_environment() (Tier-3)
  └── probe_integration_profile_health(circuit_breaker_config=…)

build_harness_host_runtime() (Tier-3)
  └── reliability_wiring.py + reliability_assembly_resolver.py

Release / CI
  └── check_harness_reliability_wiring.py
  └── test_harness_reliability_wiring.py
```

### R.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `ReliabilityProfile` | `contracts/environment_profile.py` | Author-facing reliability flags |
| `ReliabilityWiringOptions` | `reliability_runtime_bridge.py` | Profile → wiring flags |
| `ApplicationReliabilityWiring` | `reliability_wiring.py` | Resolved store + breaker config |
| `ReliabilityAssemblyValidationResult` | `reliability_assembly_resolver.py` | Wire-time conformance |
| `IdempotencyStore` | `contracts/idempotency_store.py` | Tool side-effect deduplication port |

### R.4 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Profile → wiring bridge | `pytest tests/unit/applications/test_harness_reliability_wiring.py -m gate` |
| Host reliability materialization | `python scripts/check_harness_reliability_wiring.py` |
| Long-running via environment | `pytest tests/unit/applications/test_reliability_profile.py -m gate` |
| Integration circuit breaker | `pytest tests/unit/integrations/test_integration_circuit_breaker.py -m gate` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §22: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix S — Security control plane closeout

**Audience:** Tier-3 application authors, platform engineers, security reviewers.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §23; Phase SEC **Done**; complements [Appendix H §H.3](#h3-security-profile-per-application) (V-SEC toggles).

Tier-3 hosts must materialize V-SEC middleware and `RuntimeConfig.security_profile` from typed `ApplicationSecurityProfile` — not ad-hoc middleware registration in host factories.

### S.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Profile-driven | `ApplicationSecurityProfile` on `ApplicationEnvironmentProfile` |
| Wire-time validation | `security_assembly_resolver` at `build_harness_host_runtime` |
| Typed bridge | `SecurityWiringOptions` maps profile → middleware set |
| Identity coupling | `identity_profile.tenant_required` requires `tenant_security_verify_enabled` |
| Single host path | `wire_application_security` + `apply_application_security_wiring` |

### S.2 Security wiring map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── security_profile
        ├── prompt_defense_enabled           → PromptDefenseMiddleware
        ├── tool_injection_defense_enabled   → ToolInjectionDefenseMiddleware
        ├── retrieval_poisoning_defense_enabled → RagStep trust-score path (RuntimeConfig)
        └── tenant_security_verify_enabled   → TenantSecurityMiddleware

wire_application_security() (Tier-3)
  └── security_runtime_bridge.py
        ├── resolve_security_wiring_options()
        └── apply_security_profiles_from_environment() → RuntimeConfig.security_profile

build_nexus_loop_from_environment() (Tier-3)
  └── security_wiring.py + application_security_wiring.py
        └── apply_application_security_wiring() → NexusLoop middleware

Release / CI
  └── check_harness_security_wiring.py
  └── test_harness_security_wiring.py
```

### S.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `ApplicationSecurityProfile` | `contracts/environment_profile.py` | Author-facing V-SEC toggles |
| `SecurityWiringOptions` | `security_runtime_bridge.py` | Profile → wiring flags |
| `ApplicationSecurityWiring` | `security_wiring.py` | Resolved profile + middleware names |
| `SecurityAssemblyValidationResult` | `security_assembly_resolver.py` | Wire-time conformance |
| V-SEC middleware | `application_security_wiring.py` | Prompt/tool/tenant defenses |

### S.4 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Profile → wiring bridge | `pytest tests/unit/applications/test_harness_security_wiring.py -m gate` |
| Host security materialization | `python scripts/check_harness_security_wiring.py` |
| Middleware behavior | `pytest tests/unit/applications/test_application_security_wiring.py -m gate` |
| Integration security path | `pytest tests/integration/runtime/test_nexus_loop_security_wiring.py -m gate` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §23: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Appendix T — Cost governance control plane closeout

**Audience:** Tier-3 application authors, platform engineers, FinOps.  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; complements [Appendix H §H.2](#h2-control-plane-map-where-to-customize) (`RuntimePolicyBundle.budget`).

Tier-3 hosts must materialize `BudgetPolicy`, `RunBudget`, and quota domain fragments from typed `CostProfile` — not ad-hoc budget objects in host factories.

### T.1 Design principles (Harness audit)

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| Profile-driven | `CostProfile` on `ApplicationEnvironmentProfile` |
| Wire-time validation | `cost_assembly_resolver` at `build_harness_host_runtime` |
| Typed bridge | `CostWiringOptions` maps profile → Nexus budget config |
| Policy bundle merge | `wire_policy_bundle` attaches `BudgetPolicy` + `cost_governance` fragment |
| Context fallback | Explicit cost limits or `ContextProfile.budget_policy` when enforcement enabled |

### T.2 Cost wiring map

```text
ApplicationEnvironmentProfile (Tier-3)
  └── cost_profile
        ├── budget_enforcement_enabled  → BudgetPolicy
        ├── max_* limits                → RunBudget
        └── quota_degrade_threshold_ratio → domain_fragments.cost_governance

wire_application_cost() (Tier-3)
  └── cost_runtime_bridge.py
        ├── resolve_cost_wiring_options()
        └── apply_cost_profiles_from_environment() → RuntimeConfig

wire_policy_bundle() (Tier-3)
  └── policy_wiring.py + cost_wiring.py

build_harness_host_runtime() (Tier-3)
  └── cost_assembly_resolver.py

Release / CI
  └── check_harness_cost_wiring.py
  └── test_harness_cost_wiring.py
```

### T.3 Core contracts

| Contract | Module | Role |
|----------|--------|------|
| `CostProfile` | `contracts/environment_profile.py` | Author-facing budget/quota flags |
| `CostWiringOptions` | `cost_runtime_bridge.py` | Profile → wiring flags |
| `ApplicationCostWiring` | `cost_wiring.py` | Resolved budget policy + run budget |
| `CostAssemblyValidationResult` | `cost_assembly_resolver.py` | Wire-time conformance |
| `BudgetPolicy` / `RunBudget` | `runtime/nexus/budget/budget_models.py` | Nexus enforcement |

### T.4 Verification (audit evidence)

| Concern | Command / test |
|---------|----------------|
| Profile → wiring bridge | `pytest tests/unit/applications/test_harness_cost_wiring.py -m gate` |
| Host cost materialization | `python scripts/check_harness_cost_wiring.py` |
| V-COST envelope/quota logic | `pytest tests/unit/runtime/architecture/test_cost_budget.py tests/unit/runtime/architecture/test_cost_quota.py -m gate` |
| Runtime config bridge | `pytest tests/unit/applications/test_runtime_config_bridge.py -m gate` |
| Full gate | `uv run pytest -m gate -q` |

Full audit procedure: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · layer §24: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Anti-patterns

| Do not | Do instead |
|--------|------------|
| Import `intergrax.chat_agent` / `ChatAgent` | Nexus `RuntimeEngine` / `NoPlannerPipeline` |
| Import `intergrax.rag.answers` from runtime | `RetrievalService` |
| Put agent logic in `applications/` | Logic in `agents/`, wiring in application |
| Modify `NexusLoop` for one agent | `registry.register()` + contract/metadata |
| Expect lab app to auto-load new agents | Add `AgentBinding.mount(...)` in `lab_application/manifest.py` + builder |
| Use string `import_path` / `factory_path` in Python manifests | `AgentBinding.mount(AgentClass, factory=callable)` — see `intergrax/applications/USAGE.md` |
| Duplicate LLM/trace/queue stacks | Extend Tier-0 platform |
| Import `integrations/providers/` from `agents/` | Declare `allowed_tools`; wire slugs in Tier-3 `factory.py` |
| Hardcode Slack/Postgres/Redis in agent steps | `ToolRequest` + application `IntegrationProfile` |
| Model a business workflow as one giant `ToolContract` | **Skill** pack (tool_ids + prompts) + UAEP steps on agent |
| Copy prompt + tool lists into every new agent | Reuse `skill_ids` from [Skill Library](SKILLS.md) |
| Tie agent to one product | Reusable capability in `agents/` |
| Document this workflow in multiple files | Update **this guide only** |
| Use `getattr` / `setattr` on harness paths (`runtime/nexus/`, `agents/`) | Explicit `Protocol` / typed fields; CI `scripts/check_harness_no_getattr.py` |
| Import legacy `tools_agent` or `chat_router` modules | `CatalogToolPlanner` + `ToolRuntime` + `allowed_tools` on contract |
| Rely on flat `Task.metadata` keys for options | Typed `Task.options` / `Task.runtime`; opt-in hydrate via `metadata_needs_hydration` |

### CI and import gates (§5.2 reuse)

Run before opening a harness PR (see `scripts/`):

| Script | Enforces |
|--------|----------|
| `check_harness_no_getattr.py` | No new `getattr`/`setattr` under `runtime/nexus/` and `agents/` |
| `check_legacy_modules_removed.py` | Removed modules (`tools_agent`, `chat_router`, `chains`) stay absent; no production imports |
| `check_agent_skill_resolution.py` | Tier-2 agents do not pre-populate `allowed_tools`; skills resolve at register |
| `check_harness_registry_resolution.py` | Tier-3 hosts wire catalogs via `wire_application_environment` / `build_harness_host_runtime` |
| `check_harness_capability_graph_wiring.py` | Hosts materialize environment capability graph at wire time |
| `check_harness_observability_wiring.py` | Hosts wire observability stores from `ObservabilityProfile` |
| `check_harness_reliability_wiring.py` | Hosts wire reliability stores from `ReliabilityProfile` |
| `check_harness_security_wiring.py` | Hosts wire V-SEC middleware from `ApplicationSecurityProfile` |
| `check_harness_cost_wiring.py` | Hosts wire budget policy from `CostProfile` |
| `check_agents_vendor_imports.py` | Agents do not import `integrations/providers/` |
| `check_integration_vendor_imports.py` | Tier-0 does not import application/agent trees incorrectly |
| `check_production_chat_agent_imports.py` | No `ChatAgent` on production paths |
| `check_legacy_package_boundaries.py` | Supervisor not pulled into runtime/applications |

---

## Instructions for LLM coding agents

When asked to create a new Intergrax agent:

1. Read this guide end-to-end.
2. Run `python -m intergrax.scaffold new-agent <slug> --capability <id>`.
3. Edit only `agents/<slug>/` — primarily `steps/`, `prompts/`, `schemas/`, `contract.py`.
4. Register in the appropriate context (§ Step 4). New deployable host: Step **4E** (`new-application`). Shared lab: Step **4C** (`lab_application/manifest.py`).
5. Verify: `uv run pytest agents/<slug>/tests -q` then `uv run pytest -m gate -q`; optionally `python scripts/check_agents_vendor_imports.py`, `python scripts/check_integration_vendor_imports.py`, and `python scripts/check_production_chat_agent_imports.py` (no `ChatAgent` in production paths).
6. Do **not** modify `intergrax/runtime/` unless a reusable Tier-0 gap is proven and approved.
7. Do **not** import `intergrax.integrations.providers.*` from agent code — wire integrations in Tier-3 only (Appendix E).
8. For Tier-3 hosts, configure governance via `ApplicationEnvironmentProfile` + `RuntimePolicyBundle` — see [Appendix H](#appendix-h--governance-policy--observability-control-plane).
9. For multi-agent / graph / delegation behavior, read [Appendix I](#appendix-i--orchestration-control-plane) — never wire cross-agent calls inside `agents/`.
10. For tool/skill catalogs and runtime bridge, read [Appendix J](#appendix-j--tools--skills-control-plane) — enable profiles on environment, not in agent code.
11. For integration backends and RAG retrieval, read [Appendix K](#appendix-k--integration--rag-control-plane) — wire `IntegrationProfile` in Tier-3 only.
12. For context budget and assembly, read [Appendix L](#appendix-l--context-engineering-control-plane) — configure `ContextProfile`, not ad-hoc prompt stitching.
13. For YAML prompt catalogs and registry wiring, read [Appendix M](#appendix-m--prompt-registry-control-plane) — configure `PromptProfile`, not inline prompt strings.
14. For agent contract assembly, skills, and lifecycle, read [Appendix N](#appendix-n--agent-assembly-control-plane) — declare `skills` on `AgentContract`, not raw `allowed_tools`.
15. For registry wiring and catalog resolution, read [Appendix O](#appendix-o--registry-architecture-control-plane) — enable profiles on environment, not direct `ToolRegistry()` in hosts.
16. For capability graph lineage and blast-radius, read [Appendix P](#appendix-p--capability-graph-control-plane) — environment graph is built at wire time from catalog baseline.
17. For observability wiring and assembly validation, read [Appendix Q](#appendix-q--observability-control-plane-closeout) — configure `ObservabilityProfile`, not direct `wire_nexus_observability()` in hosts.
18. For reliability wiring and idempotency/circuit breaker assembly, read [Appendix R](#appendix-r--reliability-control-plane-closeout) — configure `ReliabilityProfile`, not ad-hoc `IdempotencyStore` in hosts.
19. For security wiring and V-SEC middleware assembly, read [Appendix S](#appendix-s--security-control-plane-closeout) — configure `ApplicationSecurityProfile`, not direct middleware in host factories.
20. For cost governance wiring and budget assembly, read [Appendix T](#appendix-t--cost-governance-control-plane-closeout) — configure `CostProfile`, not ad-hoc `BudgetPolicy` in hosts.
21. Do **not** create duplicate workflow documentation — update this file if the process changes.
