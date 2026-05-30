# Tier-3 application composition engine — usage

**Package:** `intergrax/applications/`  
**Canon:** `docs/intergrax_runtime_architecture.md` §7.4.8–§7.4.10  
**Host examples:** `applications/lab_application/`, `applications/legal_application/`

> Tier-3 **applications** compose Tier-2 **agents** + Tier-0 **integrations** into a runnable HTTP/Docker environment.  
> Tier-2 agents must **not** import application packages or wire env/Docker themselves.

---

## What this engine does

| Piece | Module | Role |
|-------|--------|------|
| Roster contract | `contracts/manifest.py` | **Who** is mounted (`ApplicationManifest`, `AgentBinding`) |
| Build context | `contracts/build_context.py` | **What settings/integrations** factories receive |
| Materialization | `_shared/wiring.py` | **How** bindings become `AgentRegistry` entries |
| Typed factories | `contracts/factory.py` | Protocol `(ctx, binding) -> Agent` |

End-to-end flow:

```text
ApplicationManifest  +  ApplicationBuildContext  +  optional builders map
        →  build_application_registry()
        →  AgentRegistry
        →  NexusLoop / FastAPI host
```

---

## Define a manifest (strongly typed)

Prefer **`AgentBinding.mount()`** with the agent **class** and an optional **factory callable**.  
Do **not** use string `import_path` / `factory_path` in hand-written application code (those are for scaffold-generated manifests only — `AgentBinding.deserialize()`).

### Simple agent (zero-arg constructor)

```python
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

manifest = ApplicationManifest.lab(
    app_id="my_lab",
    name="My Lab",
    agents=[
        AgentBinding.mount(EchoAgent, capabilities=["echo.basic"]),
    ],
)
```

### Agent with Tier-3 configuration (factory)

Pass a callable that receives `ApplicationBuildContext` (and `AgentBinding`).  
The factory must return an instance of the mounted class.

```python
from legal.legal_agent import LegalAgent
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from legal_application.host.agent_factories import build_legal_agent_from_context

manifest = ApplicationManifest.product(
    app_id="legal",
    name="Intergrax Legal API",
    route_prefix="/v1/legal",
    env_prefix="LEGAL_",
    agents=[
        AgentBinding.mount(
            LegalAgent,
            factory=build_legal_agent_from_context,
            capabilities=["legal.review"],
            default=True,
        ),
    ],
)
```

Example factory (application-local):

```python
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding

def build_legal_agent_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> LegalAgent:
    settings = ctx.settings  # LegalBackendSettings from host
    from legal_application.host.wiring import build_legal_agent
    return build_legal_agent(settings)
```

### Type-keyed builders (lab pattern)

When many agents share the same construction pattern, register factories by **`type[Agent]`**:

```python
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.factory import AgentFactory

LAB_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    EchoAgent: lambda ctx, binding: EchoAgent(),
}
```

Bindings stay minimal:

```python
AgentBinding.mount(EchoAgent)  # resolved via LAB_AGENT_BUILDERS[EchoAgent]
```

---

## Build context

```python
from intergrax.applications.contracts.build_context import ApplicationBuildContext

ctx = ApplicationBuildContext.for_manifest(manifest, settings=app_settings)
```

| Field | Content |
|-------|---------|
| `manifest` | Full `ApplicationManifest` |
| `settings` | Application settings dataclass (`LabApplicationSettings`, `LegalBackendSettings`, …) |
| `integration_profile` | Copied from `manifest.integration_profile` |

Factories should read **secrets and product config from `ctx.settings`**, not from `os.environ` directly inside Tier-2 agents.

---

## Materialize `AgentRegistry`

Canonical API:

```python
from intergrax.applications._shared.wiring import build_application_registry

registry = build_application_registry(
    manifest,
    ctx,
    builders=LAB_AGENT_BUILDERS,  # optional: dict[type[Agent], AgentFactory] or dict[str, AgentFactory]
)
```

Shortcut when only manifest + settings are needed:

```python
from intergrax.applications import build_registry_from_manifest

registry = build_registry_from_manifest(manifest, settings=app_settings, builders=builders)
```

### Resolution order (per binding)

| Priority | Source | When |
|----------|--------|------|
| 1 | `binding.factory` | Typed callable on `AgentBinding.mount(..., factory=...)` |
| 2 | `builders[agent_type]` or `builders[builder_key]` | Application builders map |
| 3 | `binding.factory_path` | **Serialized manifests only** (import by string) |
| 4 | `agent_type()` | Zero-arg constructor |

After materialization, optional `contract_id` on the binding overrides `AgentContract.id` at `registry.register()`.

---

## Wire into a FastAPI host

Typical `host/wiring.py`:

```python
def build_my_registry(*, settings: MyAppSettings | None = None) -> AgentRegistry:
    settings = settings or MyAppSettings.from_env()
    manifest = build_my_manifest(settings)  # or constant MY_MANIFEST
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
    return build_application_registry(manifest, ctx, builders=MY_AGENT_BUILDERS)
```

`host/factory.py` then:

```python
registry = build_my_registry(settings)
nexus_loop = NexusLoop(registry, trace_store=..., ...)
app = create_fastapi_app(nexus_loop, ...)
```

See `applications/legal_application/host/factory.py` and `applications/lab_application/host/factory.py`.

---

## Integrations on the manifest

```python
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

manifest = ApplicationManifest.lab(
    ...,
    integration_profile=IntegrationProfile(
        relational_store=IntegrationSlug.SQLITE,
        notification_channel=IntegrationSlug.LOG,
    ),
)
```

Resolve backends in `host/integration_wiring.py` via `profile.resolve(IntegrationCategory.…)`.  
See `intergrax/integrations/providers/<slug>/USAGE.md` per provider.

---

## Serialized bindings (scaffold / YAML only)

```python
AgentBinding.deserialize(
    import_path="echo.echo_agent.EchoAgent",
    factory_path="my_app.host.agent_factories.build_echo",  # optional
)
```

Use **`deserialize()`** only in generated code. Hand-written manifests should use **`mount()`**.

---

## Invoke and verify

### Unit test (registry only)

```python
from intergrax.applications import build_registry_from_manifest

def test_registry_has_echo():
    manifest = ApplicationManifest.lab(
        app_id="t", name="T", agents=[AgentBinding.mount(EchoAgent)]
    )
    registry = build_registry_from_manifest(manifest)
    assert registry.has("echo")
```

### HTTP (lab)

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
curl -s http://127.0.0.1:8090/v1/lab/agents
```

### Full stack

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest tests/acceptance/agent_os/test_lab_application.py -q
```

---

## Reference implementations

| Application | Manifest | Builders / factory |
|-------------|----------|-------------------|
| Lab | `applications/lab_application/manifest.py` | `host/agent_builders.py` (`dict[type[Agent], …]`) |
| Legal | `applications/legal_application/manifest.py` | `host/agent_factories.py` (`factory=` on mount) |

---

## Rules and anti-patterns

| Do | Don't |
|----|--------|
| `AgentBinding.mount(AgentClass, factory=...)` | String `import_path` + `factory_path` in Python source |
| Keep domain logic in `agents/<slug>/` | Pipeline steps or prompts in `applications/` |
| One factory per agent that needs config | `agent_cls()` when Legal-style deps are required |
| `build_application_registry(manifest, ctx, builders=...)` | Ad-hoc `registry.register()` scattered without manifest |
| Per-app `.env` / `settings.from_env()` | Secrets only in root `.env` |

**Terminology:** Tier-3 **application environment** (this package) ≠ Tier-1 **runtime sandbox** (`metadata.sandbox` on tasks). See canon §7.4.9.

---

## Related docs

- Repository Tier-3 folder: [`applications/USAGE.md`](../../applications/USAGE.md)
- Agent workflow: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md)
- Implementation plan Phase N: [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md)
