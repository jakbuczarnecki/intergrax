# Tier-3 applications (`applications/`) — usage

**Repository path:** `applications/<app_name>/`  
**Composition engine:** [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)  
**Architecture:** `docs/intergrax_runtime_architecture.md` §7.4.8–§7.4.10

Each folder under `applications/` is a **self-contained execution environment**: host, env, agent roster, integrations, and (when scaffolded) Docker.  
Tier-2 agent logic lives in `agents/` — not here.

---

## Layout of one application

```text
applications/my_lab/
    manifest.py              # ApplicationManifest + AgentBinding.mount(...)
    README.md                  # Quickstart (uvicorn, curl, docker)
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
    my_lab_tests/              # Host smoke tests
```

**Python path:** `applications/` is on `pythonpath` (`pyproject.toml`). Import as `my_lab.host.main`, not `applications.my_lab`.

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

Full guide: [`intergrax/tools/USAGE.md`](../intergrax/tools/USAGE.md) · catalog: [`docs/TOOLS.md`](../docs/TOOLS.md)

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
uv run pytest applications/my_lab/my_lab_tests -q
uv run pytest tests/unit/applications/ -q
```

### Docker (when `docker/` exists)

```bash
# Recommended — per-app scripts (monorepo root context, BuildKit if available)
applications/my_lab_application/docker/build-docker.sh
# Windows: applications\my_lab_application\docker\build-docker.bat

docker run --env-file applications/my_lab_application/.env -p 8091:8091 my-lab-application
```

Manual build: see `applications/<app>/BUILD_AND_DEPLOY.md`.

### Docker and CI

| Context | Expectation |
|---------|------------|
| **Gate (`pytest -m gate`)** | Validates scaffold tree, `build-docker.sh` / `.bat` content, runtime E2E — **no** `docker build` |
| **Optional integration** | `tests/integration/applications/test_poc_template_docker_build.py` when Docker is installed |
| **Production** | Run per-app `applications/<pkg>/docker/build-docker.sh` from repo root |

Readiness checklist: [`TIER3_READINESS.md`](TIER3_READINESS.md).

---

## Scaffold commands

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

Per-app details: each application's `README.md`.

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

# Explicit (product)
llm = LLMProfile(provider=LLMProvider.AZURE_OPENAI, model="gpt-4o-deployment").create_adapter()

# Env-driven (lab / K8s)
# INTERGRAX_LLM_PROVIDER=groq  INTERGRAX_LLM_MODEL=llama-3.3-70b-versatile
llm = llm_profile_from_env(prefix="INTERGRAX_LLM").create_adapter()
```

Set provider API keys in `.env` / secrets (see [LLM_ADAPTERS.md](../docs/LLM_ADAPTERS.md) env tables). Optional live smoke: GitHub workflow `llm-network-smoke.yml`.

| Task | Where |
|------|--------|
| Create agent | `python -m intergrax.scaffold new-agent …` → `agents/` |
| Register in app | `AgentBinding.mount(...)` in `applications/<app>/manifest.py` |
| Wire backends | `IntegrationProfile` in manifest + `integration_wiring.py` |
| Select LLM provider | `LLMProfile` or env `INTERGRAX_LLM_PROVIDER` / `INTERGRAX_LLM_MODEL` — see [LLM_ADAPTERS.md](../docs/LLM_ADAPTERS.md) |
| Enable catalog tools | `tool_wiring.py` + pass `tool_profile` via `ApplicationBuildContext` |
| Scaffold app | [`TIER3_READINESS.md`](TIER3_READINESS.md) · Guide Step **4E** — `new-stack` or `new-application --profile lab\|product` |

---

## Related docs

- **Engine API (define / invoke registry):** [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md)
- **LLM adapters (providers, env, deployment):** [`docs/LLM_ADAPTERS.md`](../docs/LLM_ADAPTERS.md)
- **Tool catalog wiring:** [`intergrax/tools/USAGE.md`](../intergrax/tools/USAGE.md)
- **Agent creation:** [`docs/AGENT_CREATION_GUIDE.md`](../docs/AGENT_CREATION_GUIDE.md)
- **Phase N plan:** [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../docs/INTERGRAX_IMPLEMENTATION_PLAN.md)
