# Tier-3 applications (`applications/`) — usage

**Repository path:** `applications/<app_name>/`  
**Composition engine:** [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)  
**Architecture:** `docs/intergrax_runtime_architecture.md` §7.4.8–§7.4.10

> **Documentation boundary:** Platform docs in `docs/` (architecture canon, `intergrax_runtime_architecture.md`) describe the **Harness** and how to host applications. Each product under `applications/<name>/` maintains its own **`docs/ARCHITECTURE.md`**, **`docs/IMPLEMENTATION_PLAN.md`**, and deployment notes — those are **not** duplicated in the platform plan.

> **Authoring rule:** Application authors define product behavior and compose platform capabilities. They do not implement generic platform infrastructure. For ownership decisions see [`docs/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../docs/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md).

Each folder under `applications/` is a **self-contained execution environment**: host, env, agent roster, integrations, **dependency project** (`pyproject.toml`), and (when scaffolded) Docker.  
Tier-2 agent logic lives in `agents/` — not here.

**Dependencies:** each real application owns `applications/<app>/pyproject.toml` (Intergrax workspace package + selected extras). Sync with `uv sync --project applications/<app>`. Canon: [`docs/architecture/APPLICATION_DEPENDENCY_MODEL.md`](../docs/architecture/APPLICATION_DEPENDENCY_MODEL.md).

Isolation (current monorepo): **declaration** and **dependency-graph** isolation per application project are supported; the default physical environment remains the workspace root `.venv` (not one `.venv` per app unless `UV_PROJECT_ENVIRONMENT` is set).

**Proof evidence:** Application-scoped live evidence belongs under
`applications/<app>/docs/proof/` when the execution uses that application as the
reference host and closes an application roadmap gate.
Reusable proof harnesses and platform provider tests remain in platform-owned
locations.

---

## Phase V compliance hooks (harness hardening)

When adding or changing Tier-3 application hosts, include harness hardening hooks where relevant:

- capability graph impact visibility for changed app/agent/tool wiring (`V-CG.*`),
- lifecycle governance metadata path for production-eligible agents (`V-ALG.*`),
- context/prompt/eval regression compatibility in host pipelines (`V-CE.*`, `V-PE.*`, `V-EVAL.*`),
- security and cost policy enforcement in runtime wiring (`V-SEC.*`, `V-COST.*`).

Primary tracker: `docs/intergrax_runtime_architecture.md` Phase V.

---

## Layout of one application

```text
applications/my_lab/
    pyproject.toml           # Application dependency project (Intergrax + extras)
    manifest.py              # ApplicationManifest + AgentBinding.mount(...)
    README.md                # Quickstart (uvicorn, curl, docker) — sole top-level doc entry
    docs/
        ARCHITECTURE.md        # Host purpose, manifest, dependencies
        IMPLEMENTATION_PLAN.md # Local implementation queue
        BUILD_AND_DEPLOY.md    # Operational runbook (when deploy triad present)
        adr/                   # Application architecture decisions
    scripts/
        build-local-docker.sh  # Operator entrypoint → docker/ compose or build
        build-local-docker.bat
    sample_docs/               # Local smoke fixtures (.gitignore keeps repo clean)
    .env.example               # App-prefixed env vars (MY_LAB_*)
    host/
        main.py                # ASGI entry, load_dotenv
        factory.py             # create_*_application() → FastAPI
        settings.py            # Settings dataclass + from_env()
        wiring.py              # build_*_registry() → AgentRegistry
        agent_builders.py      # dict[type[Agent], AgentFactory] (optional)
        agent_factories.py     # typed factories for configured agents (optional)
        integration_wiring.py  # IntegrationProfile → stores/adapters
        tool_wiring.py         # ToolProfile + ToolWiringContext → catalog registry
    serving/
        fastapi_router.py      # HTTP routes → NexusLoop / UnifiedTaskRunner
    docker/                    # Dockerfile (Phase N scaffold)
    tests/              # Host smoke tests
```

**Python path:** `applications/` is on `pythonpath` (`pyproject.toml`). Import as `my_lab.host.main`, not `applications.my_lab`.

### Deploy triad (required per application — Phase AA)

Every Tier-3 host under `applications/<app>/` must ship:

| Piece | Path | Notes |
|-------|------|--------|
| **Docker** | `docker/Dockerfile`, `docker-compose.yml`, `build-docker.sh` / `.bat` | Image build from repo root context |
| **Deploy doc** | `docs/BUILD_AND_DEPLOY.md` | From scaffold `render_build_deploy_doc` or kept in sync manually |
| **Dependencies** | `applications/<app>/pyproject.toml` + `docs/ARCHITECTURE.md` § Dependencies | Application selects Intergrax extras; see `docs/architecture/APPLICATION_DEPENDENCY_MODEL.md` |
| **Implementation plan** | `docs/IMPLEMENTATION_PLAN.md` | Local task queue — scaffold emits on create; links to `docs/ARCHITECTURE.md` |

Gate: `tests/unit/applications/test_application_deploy_triad.py` · doc pair: `tests/unit/applications/test_agent_app_doc_pair.py`.

**Scaffold default vs `--full`:** `python -m intergrax.scaffold new-application …` emits H-APP `factory.py` + `environment_profile.py` without `integration_wiring.py` / `tool_wiring.py`. Use `--full` only when custom catalog wiring is required.

### Progressive disclosure (Phase DX-0.4)

| Stage | How | Notes |
|-------|-----|-------|
| **Minimal** | `python -m intergrax.scaffold new-stack <name> --profile lab --minimal` | Harness-only factory; skip Docker/MCP until promoted |
| **Standard** | `new-stack` or `new-application` without `--minimal` | Full lab/product scaffold (Docker, MCP, deploy doc) |
| **Promote** | `python -m intergrax.scaffold expand <app_slug>` | Upgrade minimal lab tree to standard layout |

Author path: [`docs/guides/AGENT_CREATION_GUIDE.md`](../docs/guides/AGENT_CREATION_GUIDE.md) Step 4E § E.0.

---

## How to define an application

### 1. Manifest — who is active

```python
# applications/my_lab/manifest.py
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

def build_my_lab_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="my_lab",
        name="My Lab",
        route_prefix="/v1/my_lab",
        env_prefix="MY_LAB_",
        agents=[
            AgentBinding.mount(EchoAgent, capabilities=["echo.basic"]),
        ],
    )
```

Dynamic roster (flags from settings) — see `applications/lab_application/manifest.py`.

### 2. Builders — how instances are created

**Simple agents** — type-keyed map:

```python
# applications/my_lab/host/agent_builders.py
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.factory import AgentFactory

MY_LAB_BUILDERS: dict[type, AgentFactory] = {
    EchoAgent: lambda ctx, binding: EchoAgent(),
}
```

**Configured agents** — dedicated factory on the binding:

```python
AgentBinding.mount(LegalAgent, factory=build_legal_agent_from_context)
```

See `applications/legal_application/host/agent_factories.py`.

### 3. Wiring — registry assembly

```python
# applications/my_lab/host/wiring.py
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from my_lab.host.agent_builders import MY_LAB_BUILDERS
from my_lab.host.settings import MyLabSettings
from my_lab.host.tool_wiring import wire_my_lab_tools
from my_lab.manifest import build_my_lab_manifest

def build_my_lab_registry(*, settings: MyLabSettings | None = None):
    settings = settings or MyLabSettings.from_env()
    manifest = build_my_lab_manifest()
    tool_wiring = wire_my_lab_tools(integration_profile=getattr(manifest, "integration_profile", None))
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
    )
    return build_application_registry(manifest, ctx, builders=MY_LAB_BUILDERS)
```

Agent factories that need tools must read `ctx.tool_profile` / `ctx.tool_wiring_context` and pass them into agent config → `RuntimeConfig`. See `legal_application/host/agent_factories.py` and `research_application/host/agent_builders.py`.

### 3b. Tool wiring template

```python
# applications/my_lab/host/tool_wiring.py
from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
from intergrax.tools.registry.profile import ToolProfile

def wire_my_lab_tools(*, integration_profile=None):
    return build_application_tool_wiring(
        ToolProfile(enabled=["rag.retrieve", "websearch.query"]),
        integration_profile=integration_profile,
    )
```

Full guide: [`intergrax/tools/USAGE.md`](../intergrax/tools/USAGE.md) · catalog: [`docs/architecture/TOOLS.md`](../docs/architecture/TOOLS.md)

### 4. Host — HTTP + Nexus

```python
# applications/my_lab/host/factory.py
from my_lab.host.wiring import build_my_lab_registry
from intergrax.applications._shared.plugin_bootstrap import bootstrap_application_plugins
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.plugins.default_plugins import default_lab_plugins

def create_my_lab_application():
    settings = MyLabSettings.from_env()
    registry = build_my_lab_registry(settings=settings)
    nexus_loop = NexusLoop(registry, ...)
    bootstrap_application_plugins(
        default_lab_plugins(trace_store=nexus_loop.trace_store),
        nexus_loop=nexus_loop,
    )
    app = ...
    return app
```

Lab and product scaffolds call `bootstrap_nexus_platform()` from `intergrax/applications/_shared/platform_wiring.py` — registers compatibility telemetry and metrics export on `TASK_COMPLETED`. Legal, research, lab, and poc_template hosts use this pattern.

### 4b. Slack / Teams interaction intake (§18)

Product and lab hosts can expose **`POST {interaction_route_prefix}/intake`** (default `/v1/interactions/intake`) for inbound Slack, Teams, or lab JSON payloads:

```python
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.runtime.interactions.router import create_interaction_intake_router

interaction_service = wire_interaction_intake_service(
    nexus_loop,
    interaction_surface=settings.interaction_surface,  # auto | slack | teams | lab
)
app.include_router(
    create_interaction_intake_router(interaction_service, execute_default=True),
    prefix=settings.interaction_route_prefix,
)
```

`InteractionIntakeService` also accepts an optional `task_executor` (`intergrax.runtime.interactions.task_executor.TaskExecutor`). When set, `execute=true` routes through the executor instead of calling `NexusLoop` directly; `nexus_loop=` remains supported for backward compatibility. LKW mounts a shared `LocalWorkspaceTaskExecutor` so `/run` and interaction intake share one application boundary.

| Host | Env flag | Surface env |
|------|----------|-------------|
| lab | `LAB_INCLUDE_INTERACTIONS` | `LAB_INTERACTION_SURFACE` |
| legal | `LEGAL_INCLUDE_INTERACTIONS` | `LEGAL_INTERACTION_SURFACE` |
| research | `RESEARCH_INCLUDE_INTERACTIONS` | `RESEARCH_INTERACTION_SURFACE` |

Configure signing secrets per vendor (see integration provider `USAGE.md` for slack/teams).

### 5. Environment

- Commit **`host/settings.py`** + **`.env.example`** with `MY_LAB_*` variables.
- Gitignore **`.env`** locally.
- Root `.env.example` is for Tier-0/platform only.

---

## How to run

### Local (development)

```bash
# From repository root
cp applications/my_lab/.env.example applications/my_lab/.env
uv run uvicorn my_lab.host.main:app --host 127.0.0.1 --port 8091
```

### Tests

```bash
uv run pytest applications/my_lab/tests -q
uv run pytest tests/unit/applications/ -q
```

### Docker (when `docker/` exists)

```bash
# Recommended — per-app scripts (monorepo root context, BuildKit if available)
applications/my_lab_application/docker/build-docker.sh
# Windows: applications\my_lab_application\docker\build-docker.bat

docker run --env-file applications/my_lab_application/.env -p 8091:8091 my-lab-application
```

Manual build: see `applications/<app>/docs/BUILD_AND_DEPLOY.md`.

### Docker and CI

| Context | Expectation |
|---------|------------|
| **Gate (`pytest -m gate`)** | Validates scaffold tree, `build-docker.sh` / `.bat` content, runtime E2E — **no** `docker build` |
| **Optional integration** | `tests/integration/applications/test_poc_template_docker_build.py` when Docker is installed |
| **Production** | Run per-app `applications/<pkg>/docker/build-docker.sh` from repo root |

Readiness checklist: [`TIER3_READINESS.md`](TIER3_READINESS.md).

---

## Scaffold commands

Product scaffold (`--profile product`) generates:

- `manifest.py` — `_resolve_integration_profile()` from `INTERGRAX_INTEGRATION_PROFILE_JSON` or SQLite + inmemory + Docling defaults
- `integration_wiring.py` — `wire_*_integrations(integration_profile=…)` → `wire_nexus_observability`
- `tool_wiring.py` — product tools: `rag.*`, `websearch.*` (incl. `fetch_batch`)

Lab scaffold (`--profile lab`) uses `IntegrationProfile.lab()` and tools: `rag.retrieve`, `websearch.query`, `websearch.read_url`, `sandbox.exec`.

```bash
# Full stack (Tier-2 agent + Tier-3 host)
python -m intergrax.scaffold new-stack my_feature --profile lab

# Application only
python -m intergrax.scaffold new-application my_lab --profile lab --agents echo
python -m intergrax.scaffold new-application my_product --profile product --agents echo --port 8000
```

---

## Reference applications

| App | Purpose | Start |
|-----|---------|-------|
| `lab_application` | Universal lab + `/debug/*` | `uv run uvicorn lab_application.host.main:app --port 8090` |
| `legal_application` | Product API + Legal agent | `uv run uvicorn legal_application.host.main:app --port 8000` |
| `research_application` | Research pipeline host | See `research_application/README.md` |
| `local_workspace_application` | **Local Knowledge Workspace (LKW)** — local index, search, synthesis | See [`local_workspace_application/docs/ARCHITECTURE.md`](local_workspace_application/docs/ARCHITECTURE.md) |

Per-app details: each application's `README.md` and `docs/ARCHITECTURE.md` where present.

---

## Relationship to agents and integrations

```text
agents/<slug>/           Tier-2 — domain logic, AgentContract, UAEP
        ↑
applications/<app>/      Tier-3 — manifest + registry + HTTP + env
        ↑
intergrax/applications/    Composition engine (manifest, wiring API)
        ↑
intergrax/integrations/  Tier-0 — IntegrationProfile, providers
intergrax/llm_adapters/    Tier-0 — LLMProfile, LLMAdapterRegistry (not Integration Library)
```

### LLM provider in deployment

Tier-3 hosts should **not** import OpenAI/Anthropic SDKs. Use the adapter registry or a declarative profile:

```python
from intergrax.llm_adapters.registry import LLMProfile, llm_profile_from_env
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

# Explicit (any Tier-3 host)
llm = LLMProfile(
    provider=LLMProvider.GROQ,
    model="llama-3.3-70b-versatile",
    options={"context_window_tokens": 128_000},  # override when catalog miss — see USAGE.md
).create_adapter()

# Env-driven (lab / K8s / deploy)
# INTERGRAX_LLM_PROVIDER=groq  INTERGRAX_LLM_MODEL=llama-3.3-70b-versatile
llm = llm_profile_from_env(prefix="INTERGRAX_LLM").create_adapter()
```

Set API keys via env or `secrets=` on `LLMProfile.create_adapter()`. Enable `INTERGRAX_LLM_METRICS_ENABLED=true`; optional `register_llm_metrics_routes(app)`. Nexus hosts using `bootstrap_nexus_platform()` get automatic tenant-scoped LLM metrics on task completion — no manual `set_llm_tenant_id` required. Optional quota: `INTERGRAX_LLM_TENANT_MAX_TOKENS`.

**Developer guide:** [`intergrax/llm_adapters/USAGE.md`](../intergrax/llm_adapters/USAGE.md) (env matrix, Cohere slugs, failover, catalog override). **Architecture:** [architecture/LLM_ADAPTERS.md](../docs/architecture/LLM_ADAPTERS.md). **Active plan:** [M-LLM-X](../docs/plan/LLM_ADAPTERS.md) (ModelCatalog, routing).

| Task | Where |
|------|--------|
| Create agent | `python -m intergrax.scaffold new-agent …` → `agents/` |
| Register in app | `AgentBinding.mount(...)` in `applications/<app>/manifest.py` |
| Wire backends | `IntegrationProfile` in manifest + `integration_wiring.py` |
| Select LLM provider | `LLMProfile` or env `INTERGRAX_LLM_*` — see [USAGE.md](../intergrax/llm_adapters/USAGE.md) |
| Enable catalog tools | `tool_wiring.py` + pass `tool_profile` via `ApplicationBuildContext` |
| Scaffold app | [`TIER3_READINESS.md`](TIER3_READINESS.md) · Guide Step **4E** — `new-stack` or `new-application --profile lab\|product` |

---

## Related docs

- **Engine API (define / invoke registry):** [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)
- **LLM adapters (providers, env, deployment):** [`intergrax/llm_adapters/USAGE.md`](../intergrax/llm_adapters/USAGE.md) · [`docs/architecture/LLM_ADAPTERS.md`](../docs/architecture/LLM_ADAPTERS.md)
- **Tool catalog wiring:** [`intergrax/tools/USAGE.md`](../intergrax/tools/USAGE.md)
- **Agent creation:** [`docs/guides/AGENT_CREATION_GUIDE.md`](../docs/guides/AGENT_CREATION_GUIDE.md)
- **Phase N plan:** [`docs/intergrax_runtime_architecture.md`](../docs/intergrax_runtime_architecture.md)
