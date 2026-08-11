# Extension Author Guide (Tier-0 Plugin Catalogs)

> **Application dependencies:** each Tier-3 host owns applications/<app>/pyproject.toml (Intergrax workspace package + selected extras). Sync with uv sync --project applications/<app>. Canon: [docs/project/architecture/APPLICATION_DEPENDENCY_MODEL.md](../../architecture/APPLICATION_DEPENDENCY_MODEL.md).

**Last updated:** 2026-08-11 · PLATFORM-PLUGIN-5

Intergrax exposes four **core plugin catalogs** plus opt-in RAG component entry points. Shipped providers and third-party pip packages register through the same discovery protocol.

| Layer | Entry point group | Protocol | Register function | Status |
|-------|-------------------|----------|-------------------|--------|
| Integration | `intergrax.integrations` | `IntegrationPlugin` | `register_integration_plugin()` | **Done** |
| Tool | `intergrax.tools` | `ToolPlugin` | `register_tool_plugin()` | **Done** |
| Skill | `intergrax.skills` | `SkillPlugin` | `register_skill_plugin()` | **Done** |
| Context | `intergrax.context` | `ContextPlugin` | `register_context_plugin()` | **Planned** — [CE-2](../../maintainers/plans/CONTEXT_ENGINEERING.md) |
| RAG chunker | `intergrax.rag.chunkers` | `BaseChunkingStrategy` | RAG bootstrap registry | **Done** |
| RAG retriever | `intergrax.rag.retrievers` | `BaseRetriever` | RAG bootstrap registry | **Done** |
| RAG reranker | `intergrax.rag.rerankers` | `BaseReranker` | RAG bootstrap registry | **Done** |

**Architecture:** Integration → Tool → Skill → Agent; **Context Engineering** assembles LLM windows from all sources — see [`architecture/CONTEXT_ENGINEERING.md`](../../architecture/CONTEXT_ENGINEERING.md) · [plan CE-EXT](../../maintainers/plans/CONTEXT_ENGINEERING.md). **Invariants:** [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) — Tier-0/Tier-2 boundaries (`SYS-INV-*`).

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

| Concern | Agent (`agents`) | Environment (`applications` or `HarnessApplication`) |
|---------|-------------------|--------------------------------------------------------|
| Business logic, UAEP steps | Yes | No |
| Tool/skill allow-list on contract | Yes | Enables catalogs via profiles |
| Integration backends (Postgres, S3, …) | No | `IntegrationProfile` / presets |
| Nexus loop, retry, graph routing | No | `ApplicationEnvironmentProfile` (§22.6 nested bundles — same root) |
| HTTP/MCP host, auth, tenant | No | Host factory / `HarnessApplication` |

| Belongs in `applications/<app>` | Belongs in `agents/<name>` |
|----------------------------------|-----------------------------|
| `ApplicationManifest`, `ApplicationEnvironmentProfile` | `Agent`, UAEP steps, domain prompts |
| `wire_application_environment()`, host `factory.py` | `AgentContract`, skill manifests on contract |
| Tool/skill **profiles** (which catalog ids are enabled) | Business logic and step graphs |
| Policy bundle, identity, observability profiles | No direct `intergrax.integrations` / `intergrax.tools` imports |

**Forbidden:** `getattr`/`setattr` on manifests in host wiring; Tier-2 agents importing integration or tool modules (use `scripts/maintenance/check_agent_registry_bypass.py`).

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

**Examples in repo:** `integrations/examples/custom_memory_kv`, `tools/examples/custom_echo`, `skills/examples/custom_pack`.

---

## 2. External integration plugin

Reference: `intergrax/integrations/examples/custom_memory_kv`

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

Reference: `intergrax/tools/examples/custom_echo`

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

Reference: `intergrax/skills/examples/custom_pack`

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

[project.entry-points."intergrax.rag.chunkers"]
my_recursive = "my_pkg.chunking:MyRecursiveChunker"

[project.entry-points."intergrax.rag.retrievers"]
my_retriever = "my_pkg.retrieval:MyRetriever"

[project.entry-points."intergrax.rag.rerankers"]
my_reranker = "my_pkg.reranking:MyReranker"
```

Enable discovery during RAG bootstrap with `INTERGRAX_DISCOVER_PLUGINS=true` or the
bootstrap `discover_entry_points=True` argument. Select the stable component IDs
through the normal profile configuration, for example
`RagProfile(chunking_strategy_id="my_recursive", retriever_id="my_retriever",
reranker_id="my_reranker")`. RAG plugins must implement the existing contract and
are registered after built-ins; duplicate IDs fail instead of silently replacing
core defaults.

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
python scripts/maintenance/check_plugin_catalog.py
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

Reference fixture: `tests/fixtures/plugin_packages/memory_store_plugin`.

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

**Canon:** [`architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §4.4 · [`ADR-OBS-003`](../adr/entries/2026-06-17/ADR-OBS-003.md)

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

---

## 12. Security defense plugins (Phase SEC-PLANES)

Entry point group: `intergrax.security_defenses`

| Mechanism | Purpose |
|-----------|---------|
| `SecurityDefensePlugin` | S2 runtime inspection at declared `HookPoint`s |
| `defense_bundle_ids` | Shipped bundles on `ApplicationSecurityProfile` (e.g. `harness.strict_injection`) |
| `defense_plugin_ids` | Explicit EP plugin ids on profile |
| `bootstrap_security_providers()` | Load shipped bundles + optional EP discovery |

Bootstrap: `intergrax.core.security_bootstrap.bootstrap_security_providers(discover_entry_points=True)` — also invoked from `bootstrap_catalogs()`.

**Composition:** plugins merge into `MiddlewarePipeline` via `security_runtime_bridge` — after native V-SEC middleware, before `ToolRuntime`. They **never** bypass `PolicyEngine`.

**Author checklist:**

| Rule | Detail |
|------|--------|
| Tenant scope | Read `tenant_id` from `HookContext.runtime_state`; do not exfiltrate cross-tenant data |
| Hook coverage | Declare only `HookPoint`s you inspect — undeclared points are skipped |
| Fail mode | Default `FAIL_CLOSED`; `FAIL_OPEN` requires explicit product justification |
| Performance | Inspection runs under a wall-clock budget (default 100ms); slow plugins are blocked |
| Observability | Blocks emit `platform.security.defense_blocked` on the runtime bus |

**Lab fixture:** `tests/fixtures/plugin_packages/intergrax_security_defense_fixture` — reference EP package for CI discovery tests.

**Author map:** [Appendix H §H.3.1](guides/AGENT_CREATION_GUIDE.md#h31-security--trust-planes-operator-index) · canon [§42.45](architecture/UNIFIED_EXECUTION_RUNTIME.md#4245-security-and-data-governance) · [ADR-SEC-001](../adr/entries/2026-06-19/ADR-SEC-001.md).

---

## 13. Platform Plugin package manifest (PLATFORM-PLUGIN-3)

**Canon:** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

A **Platform Plugin** is a **package-level coordination contract** — not a universal runtime wrapper. One installable Python distribution (plugin package) may expose zero or more **capabilities** across domains. Each capability remains governed by its domain contract (`IntegrationPlugin`, `ToolPlugin`, `SkillPlugin`, …).

| Layer | Authoritative for |
|-------|-------------------|
| `pyproject.toml` `[project]` | Distribution identity, dependencies, setuptools entry points |
| Setuptools entry points | Machine discovery (`intergrax.integrations`, `intergrax.tools`, …) |
| Domain manifests | Runtime capability semantics (`IntegrationManifest`, tool/skill bundles, …) |
| **Optional** `[tool.intergrax.plugin]` | Package coordination metadata only (identity, compatibility, capability pointers) |

For a real `pyproject.toml`, `[project].name` and `[project].version` are **authoritative** distribution identity. Platform Plugin manifest identity fields must match them after Python packaging normalization — contradictory manifest metadata is rejected.

**Important:** manifest-valid ≠ discovered ≠ enabled ≠ qualified ≠ production-qualified. Declare `intergrax_version` using Python packaging version specifiers (e.g. `>=1.0,<2`); PLUGIN-6 checks compatibility via `check_platform_compatibility` — **compatible does not mean qualified**. Incompatible metadata must block activation once an approved host gate invokes the checker (PLUGIN-8 reference host). Installation alone does not prove platform compatibility. Secrets must **never** appear in Platform Plugin manifests.

### 13.1 Multi-capability external package example

One distribution `acme-intergrax` may register multiple domain entry points and optionally declare coordination metadata:

```toml
[project]
name = "acme-intergrax"
version = "1.0.0"
dependencies = ["Intergrax-ai"]

[project.entry-points."intergrax.integrations"]
acme_foo = "acme_intergrax.integration:AcmeFooIntegrationPlugin"

[project.entry-points."intergrax.tools"]
acme_tool = "acme_intergrax.tool:AcmeToolPlugin"

[project.entry-points."intergrax.skills"]
acme_skill = "acme_intergrax.skill:AcmeSkillPlugin"

[tool.intergrax.plugin]
name = "acme-intergrax"
version = "1.0.0"
intergrax_version = ">=1.0,<2"
author = "Acme Corp"
documentation_uri = "https://docs.example.com/acme-intergrax"

[[tool.intergrax.plugin.capabilities]]
domain = "integrations"
entry_point_group = "intergrax.integrations"
entry_point_name = "acme_foo"
capability_ids = ["acme_foo"]

[[tool.intergrax.plugin.capabilities]]
domain = "tools"
entry_point_group = "intergrax.tools"
entry_point_name = "acme_tool"

[[tool.intergrax.plugin.capabilities]]
domain = "skills"
entry_point_group = "intergrax.skills"
entry_point_name = "acme_skill"
```

Domain manifests and plugin classes remain separate and authoritative. The Platform Plugin manifest only lists capability **pointers**.

### 13.2 Python contract API

```python
from intergrax.core.plugins import (
    build_platform_plugin_manifest,
    parse_platform_plugin_pyproject_toml,
)

manifest = parse_platform_plugin_pyproject_toml(pyproject_text)
# or construct programmatically:
manifest = build_platform_plugin_manifest(
    name="acme-intergrax",
    version="1.0.0",
    intergrax_version=">=1.0,<2",
)
```

Parsing and validation are side-effect free — they do not scan installed packages or register plugins.

---

## 14. Configuration, credentials and dependency injection (PLATFORM-PLUGIN-5)

**Canon:** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) §12–§13 · cross-surface matrix §12.3.

Platform Plugin packages **declare** capabilities; the **host/application** selects and resolves deployment settings; **domain profiles and wiring contexts** materialize runtime objects. Capability code should consume **resolved** configuration — not read arbitrary process globals as the primary path.

### 14.1 Author rules (all PEP surfaces)

| Do | Do not |
|----|--------|
| Put non-secret coordination metadata in optional `[tool.intergrax.plugin]` | Put secrets, API keys, tokens, or connection strings in `[tool.intergrax.plugin]` |
| Use setuptools entry points to identify your plugin class | Put credentials or runtime config in entry-point values |
| Use the domain profile / wiring API for your surface (see §14.2) | Assume access to all application services or secrets |
| Accept explicit constructor/factory parameters from host wiring | Rely on module-level global registries or `get_service(...)` patterns |
| Let the host choose which capabilities are enabled | Treat `installed` or `discovered` as `enabled` |

**Logging:** never log secret values; do not serialize resolved credentials into manifests, metadata, or discovery output.

### 14.2 Domain-specific configuration and DI

Use the **domain-owned** primitive for your surface — the platform does not provide one universal config object.

| Surface | Enablement / config | Credentials | Materialization |
|---------|---------------------|-------------|-----------------|
| Integration | `IntegrationProfile` + `ApplicationEnvironmentProfile` | Host resolves; optional `IntegrationManifest.env_prefix` (domain contract) | `profile.resolve(IntegrationCategory.…)` |
| Tool | `ToolProfile` | `ToolWiringContext` integration slots | `register_tools(registry, ctx)` |
| Skill | `SkillProfile` | Via tools at invoke time | `register_skills(registry)` |
| Context | `ContextProfile` | Typically none at plugin boundary | `register(registry)` |
| RAG component | `RagProfile` + bootstrap kwargs | Via integrations passed into bootstrap | `BaseRetrieverPlugin.create(…)` / registry bootstrap |
| Memory store | `MemoryProfile` | Host `**kwargs` to factory | `create_user_profile_store(**kwargs)` |
| Security defense | Host security profile | `HookContext` only | Middleware wraps plugin at host |
| Policy rule | `PolicyRulesProfile` / YAML | None in EP | `evaluate(rule, context=…)` |
| Tool invocation pattern | `ToolInvocationMode` | None | `execute(state, invoker, …)` |
| Vendor Knowledge | Host builder + bindings | Scoped `credential_ref` per binding | See VK author guide — not Tier-0 catalog |

### 14.3 Integration `env_prefix` (documented exception)

Integrations may declare `env_prefix` on `IntegrationManifest`. The host/profile selects the provider; the integration factory may read environment variables under that prefix. This is **domain-owned** behavior — not a portable pattern for tools, skills, or other surfaces.

```python
# Host resolves provider; integration may read INTERGRAX_MY_KV_* when env_prefix is set.
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.contracts.base import IntegrationCategory

profile = IntegrationProfile(key_value_cache=MyIntegrationPlugin)
cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
```

### 14.4 Tool wiring example

```python
from intergrax.tools.registry.wiring import ToolWiringContext

class MyHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._cache = ctx.key_value_cache  # host-injected; do not call os.environ here

class MyToolPlugin:
    @classmethod
    def register_tools(cls, registry, ctx: ToolWiringContext) -> None:
        registry.register(my_contract(), MyHandler(ctx))
```

Configuration parsing (`parse_platform_plugin_pyproject_toml`, profile models) is **side-effect free** — it does not discover or register plugins.

---

## 15. Trust, qualification and production gates (PLATFORM-PLUGIN-7)

**Canon:** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) §16, §18.

Qualification is **evidence that a subject meets domain/program thresholds** — not installation, discovery, compatibility, or lifecycle enablement.

| State | Meaning |
|-------|---------|
| installed / discovered / loadable / contract-valid / enabled | Prerequisite/runtime states (§14) — **not** qualification |
| qualified | Domain/program evidence threshold met |
| production-qualified | Approved for production host profiles |
| live-qualified | Optional domain-specific label (e.g. VK, RAG live backends) |

### 15.1 External package author

- Wheel + setuptools entry points make capabilities **discoverable** — discoverable ≠ qualified.
- Optional `[tool.intergrax.plugin]` manifest supports package-level coordination only.
- `check_platform_compatibility` (PLUGIN-6) produces compatibility evidence; **compatible ≠ qualified**.
- Host/domain collects capability/domain evidence and sets qualification status.
- Production host profiles must require **production-qualified** evidence via `require_production_qualification` (PLUGIN-8 reference host will invoke before activation).

### 15.2 Application developer — host-embedded extension

Local modules (e.g. `applications/my_app/extensions/my_tool.py`) may implement domain contracts (`ToolPlugin`, `IntegrationPlugin`, …) and enter via **explicit host registration** (`register_tool_plugin`, `register_integration_plugin`, …).

- Packaging as a wheel is **not required** for qualification.
- Entry-point discovery is **not required** when the host registers the class directly.
- The same capability/domain qualification model applies; use `PluginDeliverySource.HOST_EMBEDDED_EXTENSION` subject identity with `host_registration_path`.
- Production use still requires production-qualified evidence and host/domain gates — same as external packages.

### 15.3 Python contract API

```python
from intergrax.core.plugins import (
    PluginQualificationLevel,
    PluginQualificationStatus,
    build_external_package_subject,
    build_host_embedded_capability_subject,
    build_qualification_result,
    require_production_qualification,
)
```

Evidence records are immutable and safe to log — never include secrets or raw credential-bearing payloads.

---

## 16. Dual-mode developer quickstarts (PLATFORM-PLUGIN-8)

**Canon:** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) §20.3–§20.4 · executable proof: [`tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py`](../../../../tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py)

Both delivery modes converge on the same domain contract and runtime (`ToolPlugin` → catalog → `ToolWiringContext` → `RuntimeToolInvoker`). Choose **external package** when distributing a reusable installable plugin; choose **host-embedded** when the code lives in your application tree.

### 16.1 External package quickstart (Tools)

Working reference: [`examples/platform_plugins/intergrax_reference_tool_plugin/`](../../../../examples/platform_plugins/intergrax_reference_tool_plugin/)

1. **Create package** — own `pyproject.toml`, Python namespace under `src/`, outside the Intergrax repository.
2. **Implement `ToolPlugin`** — `tool_bundle_manifest()` + `register_tools(registry, ctx: ToolWiringContext)`.
3. **Declare entry point** — `[project.entry-points."intergrax.tools"]` mapping to your plugin class.
4. **Optional Platform Plugin metadata** — `[tool.intergrax.plugin]` with package identity, `intergrax_version`, and capability descriptors (must match `[project].name` / `version`).
5. **Build wheel** — `uv build --wheel` (or equivalent setuptools build) in your package directory.
6. **Install wheel** — `uv pip install ./dist/*.whl` (or host deployment mechanism); isolated tests use `uv pip install --target <dir> --no-deps`.
7. **Enable discovery** — host calls `bootstrap_catalogs(discover_entry_points=True)` or sets `INTERGRAX_DISCOVER_PLUGINS=true` where the application wiring supports it.
8. **Configure host / DI** — build `ToolWiringContext` (integrations, managers, `extras`) before `build_registry_from_profile`.
9. **Qualification** — collect PLUGIN-6 compatibility evidence (`check_platform_compatibility` with explicit host platform version) and PLUGIN-7 production gates (`evaluate_package_production_admission`, `require_production_qualification`) before production activation.
10. **Run** — enable bundle/tool ids on `ToolProfile`; invoke tools via `RuntimeToolInvoker`.

### 16.2 Local application extension quickstart (Tools)

Working reference: [`examples/platform_plugins/local_embedded_tool_extension/`](../../../../examples/platform_plugins/local_embedded_tool_extension/) · scaffold emits `extensions/` automatically.

1. **Create application** — `python -m intergrax.scaffold new-application <name>`.
2. **Add local module** — implement the same `ToolPlugin` contract under `<app_pkg>/extensions/` (see generated `extensions/README.md`).
3. **Explicit registration** — in `host/tool_wiring.py`, call `register_tool_plugin(YourToolPlugin)` before `build_application_tool_wiring`.
4. **Provide DI** — pass host values via `ToolWiringContext.extras` (reference plugins use `echo_prefix`).
5. **Enable tools** — add your `tool_id` or bundle id to `ToolProfile.enabled` / `enabled_bundles`.
6. **Qualification** — use `PluginDeliverySource.HOST_EMBEDDED_EXTENSION` via `build_host_embedded_capability_subject`; capability/domain production qualification still required (`require_production_qualification`). No package manifest or PLUGIN-6 package compatibility required.
7. **No wheel / entry point** — local modules are imported directly; do not fabricate setuptools entry points for host-only code.

### 16.3 Public extension matrix

See architecture hub §20.3 for all twelve canonical entry-point surfaces, local-registration availability, and documented gaps.
