# Extension Author Guide (Tier-0 Plugin Catalogs)

**Last updated:** 2026-06-17 · Phase P-Ext · **H-APP** · §10 policy rules · §11 runtime signals (OBS-EVOL-9)

Intergrax exposes four **plugin catalogs** (three Tier-0 + Context Engineering Tier-1 contracts with Tier-0 shared types). Shipped providers and third-party pip packages register through the same protocols.

| Layer | Entry point group | Protocol | Register function | Status |
|-------|-------------------|----------|-------------------|--------|
| Integration | `intergrax.integrations` | `IntegrationPlugin` | `register_integration_plugin()` | **Done** |
| Tool | `intergrax.tools` | `ToolPlugin` | `register_tool_plugin()` | **Done** |
| Skill | `intergrax.skills` | `SkillPlugin` | `register_skill_plugin()` | **Done** |
| Context | `intergrax.context` | `ContextPlugin` | `register_context_plugin()` | **Planned** — [CE-2](../plan/CONTEXT_ENGINEERING.md) |

**Architecture:** Integration → Tool → Skill → Agent; **Context Engineering** assembles LLM windows from all sources — see [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) · [plan CE-EXT](../plan/CONTEXT_ENGINEERING.md). **Invariants:** [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) — Tier-0/Tier-2 boundaries (`SYS-INV-*`).

---

## 0. Tier-3 environment vs Tier-2 agent (H-APP, DX)

**LangGraph is not required.** Intergrax ships its own Nexus loop, `HarnessApplication`, and `AgentGraph`. The table below is a **conceptual mapping** for authors coming from LangGraph — not a runtime dependency. Optional `intergrax.supervisor.build_langgraph_from_plan` needs the extra `pip install 'Intergrax-ai[langgraph-legacy]'`.

| LangGraph (analogy) | Intergrax |
|---------------------|-----------|
| `State` fields | `AgentContract` + step metadata |
| Node function | `IntergraxAgent` `@step` / `run_step` |
| Conditional edge | `decide_after_step` → `AgentDecision` |
| `StateGraph.compile()` | `AgentGraph.build()` → `ApplicationGraphSpec` |
| `app.invoke()` | `HarnessApplication.build_fastapi()` + `POST …/run` |

**Responsibility matrix**

| Concern | Agent (`agents/`) | Environment (`applications/` or `HarnessApplication`) |
|---------|-------------------|--------------------------------------------------------|
| Business logic, UAEP steps | Yes | No |
| Tool/skill allow-list on contract | Yes | Enables catalogs via profiles |
| Integration backends (Postgres, S3, …) | No | `IntegrationProfile` / presets |
| Nexus loop, retry, graph routing | No | `ApplicationEnvironmentProfile` (§22.6 nested bundles — same root) |
| HTTP/MCP host, auth, tenant | No | Host factory / `HarnessApplication` |

| Belongs in `applications/<app>/` | Belongs in `agents/<name>/` |
|----------------------------------|-----------------------------|
| `ApplicationManifest`, `ApplicationEnvironmentProfile` | `Agent`, UAEP steps, domain prompts |
| `wire_application_environment()`, host `factory.py` | `AgentContract`, skill manifests on contract |
| Tool/skill **profiles** (which catalog ids are enabled) | Business logic and step graphs |
| Policy bundle, identity, observability profiles | No direct `intergrax.integrations` / `intergrax.tools` imports |

**Forbidden:** `getattr`/`setattr` on manifests in host wiring; Tier-2 agents importing integration or tool modules (use `scripts/check_agent_registry_bypass.py`).

Scaffold: `python -m intergrax.scaffold new-application <name>` emits `host/environment_profile.py` and unified wiring.

---

## 1. Application bootstrap

```python
from intergrax.core.catalog_bootstrap import bootstrap_catalogs

bootstrap_catalogs(
    register_shipped=True,
    integration_preset="full",       # "core" = lab essentials (~12 slugs)
    tool_bundle_ids=None,            # e.g. ("rag", "websearch") for lazy catalog
    skill_bundle_ids=None,           # e.g. ("harness", "legal")
    discover_entry_points=True,
    integration_plugins=(MyIntegrationPlugin,),
    tool_plugins=(MyToolPlugin,),
    skill_plugins=(MySkillPlugin,),
)
```

Tier-3 wiring helpers call this automatically:

- `bootstrap_application_integration_catalog()` — integrations only (`applications/_shared/integration_wiring.py`)
- `build_application_tool_wiring()` — passes `tool_bundle_ids` from `ToolProfile` when set
- `build_application_skill_wiring()` — passes `skill_bundle_ids` from `SkillProfile` when set

Optional env: `INTERGRAX_DISCOVER_PLUGINS=true` enables entry-point discovery when wiring helpers run (default off).

### Production path matrix

| Layer | Shipped registration | External package | Runtime materialization |
|-------|---------------------|------------------|-------------------------|
| Integration | `register_from_manifest` (167 slugs) | `IntegrationPlugin` + EP | `IntegrationProfile.resolve(category)` |
| Tool | `ToolPlugin` (13 bundles) | `ToolPlugin` + EP | `build_registry_from_profile(ToolProfile, ctx)` → invoke / MCP |
| Skill | `SkillPlugin` (3 bundles) | `SkillPlugin` + EP | `build_registry_from_profile(SkillProfile)` → `SkillResolver` |

**Dual model (integrations):** shipped providers use `manifest.py` + `create_*` factory; third-party packages use `IntegrationPlugin`. See `SqliteIntegrationPlugin` in `sqlite/plugin.py` as a reference class (shipped `register.py` still uses manifest path).

**Examples in repo:** `integrations/examples/custom_memory_kv/`, `tools/examples/custom_echo/`, `skills/examples/custom_pack/`.

---

## 2. External integration plugin

Reference: `intergrax/integrations/examples/custom_memory_kv/`

1. **`manifest.py`** — `IntegrationManifest(slug=..., categories=..., ...)`
2. **`plugin.py`** — `integration_manifest()` + `create_integration(**kwargs)`
3. Register at startup or via entry point.

```python
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.contracts.base import IntegrationCategory

register_integration_plugin(MyIntegrationPlugin)
profile = IntegrationProfile(key_value_cache=MyIntegrationPlugin)
cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
```

---

## 3. External tool plugin

Reference: `intergrax/tools/examples/custom_echo/`

Tools are **LLM-invokable** operations: `ToolContract` + handler class. They may compose integrations via `ToolWiringContext`.

```python
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

class MyToolPlugin:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id="my_tools",
            tool_ids=("my_tools.run",),
            status=ToolBundleStatus.BETA,
            description="My extension tools",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(my_contract(), MyHandler(ctx))
```

```python
from intergrax.tools.registry.plugin_register import register_tool_plugin

register_tool_plugin(MyToolPlugin)
```

Shipped bundles are defined in `intergrax/tools/registry/shipped_plugins.py` (factory: `define_tool_plugin`).

**Standalone LLM use:** build `ToolRegistry` from `ToolProfile` — no skills required. Export MCP/OpenAI schemas via `intergrax.tools.exporters.mcp.to_mcp_tools(registry)`.

---

## 4. External skill plugin

Reference: `intergrax/skills/examples/custom_pack/`

Skills are **not** LLM tools. They declare `tool_ids`, prompt refs, and optional policy fragments.

**Cursor `SKILL.md` import:** use `CursorSkillImporter` for one-off markdown packs — not the same as a pip `SkillPlugin`. Prefer `SkillPlugin` for versioned bundles distributed with your package.

```python
class MySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest: ...

    @classmethod
    def skill_manifests(cls) -> tuple: ...

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in cls.skill_manifests():
            registry.register(manifest)
```

Use `requires_skills` on `SkillManifest` for transitive dependencies (resolved by `SkillResolver`).

---

## 4b. Host entry-point tool patterns (TOOL-MAINT-03)

Tier-3 hosts can ship **custom tool packs** via setuptools entry points without
forking the platform catalog. Pattern:

1. Implement `ToolPlugin` with `tool_manifests()` returning `ToolManifest` rows.
2. Register in `pyproject.toml` under `[project.entry-points."intergrax.tools"]`.
3. Enable tool ids on `ToolProfile.enabled` in `ApplicationEnvironmentProfile`.
4. Wire host factory with `bootstrap_catalogs(discover_entry_points=True)` or
   explicit `register_tool_plugin(MyToolPlugin)` before `wire_application_environment()`.

Example host wiring (lab):

```python
from intergrax.applications._shared.environment_wiring import wire_application_environment
from my_product.tools import MyToolPlugin

register_tool_plugin(MyToolPlugin)
bundle = wire_application_environment(env, manifest=manifest)
ctx = bundle.wiring_context  # ToolWiringContext with custom tools resolved
```

Scaffold: `python -m intergrax.scaffold new-stack <name>` emits agent + application
with entry-point placeholders. See `intergrax/tools/USAGE.md` for invoke/MCP paths.

---

## 5. Setuptools entry points (`pyproject.toml`)

```toml
[project.entry-points."intergrax.integrations"]
my_integration = "my_pkg.integration_plugin:MyIntegrationPlugin"

[project.entry-points."intergrax.tools"]
my_tools = "my_pkg.tool_plugin:MyToolPlugin"

[project.entry-points."intergrax.skills"]
my_skills = "my_pkg.skill_plugin:MySkillPlugin"
```

Enable discovery: `bootstrap_catalogs(discover_entry_points=True)`.

Catalog slug conflicts when a plugin replaces a shipped provider:

- `on_conflict="error"` — raise on duplicate catalog slug (default)
- `on_conflict="warn_override"` — log warning and replace the catalog row
- `on_conflict="skip"` — skip the plugin when the slug already exists

---

## 6. Scaffold CLI

```bash
python -m intergrax.scaffold new-integration <slug> --category <integration_category>
python -m intergrax.scaffold new-tool-bundle <bundle_id> [--tool-id <bundle_id>.ping]
python -m intergrax.scaffold new-skill <skill_id> [--domain <folder>]
```

`new-skill-bundle` is an alias for `new-skill`. Generated trees use `IntegrationPlugin`, `ToolPlugin`, and `SkillPlugin` protocols (manifest + `plugin.py` + `bundle.py` / `register.py`).

---

## 7. Validation

```bash
set PYTHONPATH=.
python scripts/check_plugin_catalog.py
pytest tests/unit/core/plugins tests/unit/integrations/test_external_plugin.py -q
```

---

## 8. Do not confuse with Nexus runtime plugins

| Mechanism | Purpose |
|-----------|---------|
| Tier-0 catalog plugins (this guide) | Register integrations, tools, skills in catalog |
| `RuntimePlugin` / `plugin_bootstrap.py` | Nexus middleware, metrics, persistence hooks |

Agents consume **tools** via `ToolRegistry` and **skills** via `SkillResolver` → `allowed_tools`. Agents MUST NOT import vendor SDKs or integration slugs directly when a catalog tool exists.

---

## 9. Memory store plugins (Phase MEM)

Entry point group: `intergrax.memory_stores`

| Protocol | Factory method | Replaces |
|----------|----------------|----------|
| `UserProfileStorePlugin` | `create_user_profile_store(**kwargs)` | Default `InMemoryUserProfileStore` / sqlite bundle / optional Mongo `document_store` (MEM-PERS.2) |
| `SessionStoragePlugin` | `create_session_storage(**kwargs)` | Default `InMemorySessionStorage` / sqlite bundle |
| `SessionTurnIndexStore` (target MEM-VEC-3.1) | `create_session_turn_index(**kwargs)` | Default episodic vector adapter over host `VectorstoreManager` |

Bootstrap: `intergrax.core.memory_bootstrap.bootstrap_memory_stores(discover_entry_points=True)`.

**Vector memory:** LTM and session episodic indexes reuse the host integration **vector store** — memory plugins swap index adapters, not vendor SDKs. See [`architecture/MEMORY.md`](architecture/MEMORY.md) §5.3, §11.5.

Reference fixture: `tests/fixtures/plugin_packages/memory_store_plugin/`.

Swap backends in Tier-3 by registering an EP plugin — agents still use `UserProfileManager` / `SessionManager`; never import store implementations from Tier-2.

---

## 10. Policy rule handler plugins (Phase DX-5.8)

Entry point group: `intergrax.policy_rules`

| Mechanism | Purpose |
|-----------|---------|
| `PolicyRuleHandlerPlugin` | Register custom handlers evaluated by `PolicyEngine` |
| `PolicyRulesProfile.rules_path` | Declarative YAML rules loaded via `load_policy_rules_from_path` |
| `PolicyRulesProfile.inline_rules` | Inline rule dicts on `ApplicationEnvironmentProfile` |

Bootstrap: `intergrax.runtime.policy.rules.plugin_loader.register_policy_rule_plugins(discover_entry_points=True)` (called from policy wiring when enabled).

**Composition:** YAML + EP handlers merge into `RuntimePolicyBundle.domain_fragments["policy_rules"]` via `intergrax/applications/_shared/policy_wiring.py`. They **never** bypass `ToolRuntime` or `ApplicationSecurityProfile` middleware.

**Author map:** [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) [Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) · canon [§42.11](architecture/UNIFIED_EXECUTION_RUNTIME.md#4211-policy-engine).

Lab reference: `applications/lab_application/policy/rules/harness_lab.yaml`.

---

## 11. Runtime signals — spine vs `event_kind` (OBS-EVOL-9)

**Canon:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) §4.4 · [`ADR-OBS-003`](../adr/entries/2026-06-17/ADR-OBS-003.md)

Extension authors (integrations, tools, skills) and agent authors share one observability contract:

| Signal need | API | Register |
|-------------|-----|----------|
| Debug / reconstruction | `DiagnosticPayload` via `AgentEngine` | `register_payload_schema(..., extension=True)` + `agents.<slug>.diag.*` |
| Operator-visible domain fact | `emit_domain_signal(kind, payload)` | `event_kind` + extension payload schema |
| Platform lifecycle | `emit_platform_event` | Platform only — `EventCatalog` + ADR |

### 11.1 `event_kind` namespace rules

| Prefix | Owner | Example |
|--------|-------|---------|
| `agents.<slug>.` | Tier-2 agent | `agents.legal.clause_flagged` |
| `applications.<slug>.` | Tier-3 product | `applications.dispute_sim.risk_threshold_exceeded` |
| `platform.<domain>.` | Harness (via `DOMAIN_SIGNAL`) | `platform.adaptive.signal_recorded` |
| `intergrax.<domain>.` | Reserved — platform spine payloads | `intergrax.graph.checkpoint_persisted` |

Rules:

- Lowercase slug segments; dots only; no wildcards in emitted kinds.
- One kind = one semantics; never reuse `schema_id` for different meaning.
- **Do not** add `RuntimeEventType` from extension or agent packages.

### 11.2 Minimal agent example

```python
from intergrax.runtime.events.signals import emit_domain_signal

emit_domain_signal(
    ctx,
    kind="agents.my_plugin.risk_flagged",
    payload=MyRiskFlaggedPayloadV1(score=0.92),
)
```

Register payload at agent bootstrap:

```python
register_payload_schema(MyRiskFlaggedPayloadV1, extension=True)
```

### 11.3 Integration / tool plugins

Plugins **do not** publish directly to `RuntimeEventBus`. Emit through:

- **ToolRuntime** — existing `TOOL_*` spine events (audit path).
- **Agent step** — `emit_domain_signal` when the product must surface a domain fact.
- **Trace** — `DiagnosticPayload` for implementation detail.

**Author map:** [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) [Appendix Q §Q.5](AGENT_CREATION_GUIDE.md#q5-domain-runtime-signals-event_kind--obs-evol-9) · [`APPLICATION_CREATION_GUIDE.md`](APPLICATION_CREATION_GUIDE.md) §8 (Tier-3 subscribe / adapters).
