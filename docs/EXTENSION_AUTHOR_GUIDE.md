# Extension Author Guide (Tier-0 Plugin Catalogs)

**Last updated:** 2026-06-02 · Phase P-Ext

Intergrax exposes three **Tier-0 plugin catalogs**. Shipped providers and third-party pip packages register through the same protocols.

| Layer | Entry point group | Protocol | Register function |
|-------|-------------------|----------|-------------------|
| Integration | `intergrax.integrations` | `IntegrationPlugin` | `register_integration_plugin()` |
| Tool | `intergrax.tools` | `ToolPlugin` | `register_tool_plugin()` |
| Skill | `intergrax.skills` | `SkillPlugin` | `register_skill_plugin()` |

**Architecture:** Integration → Tool → Skill → Agent. See [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.5.1, §7.1.6–§7.1.8.

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
| Integration | `register_from_manifest` (~99 slugs) | `IntegrationPlugin` + EP | `IntegrationProfile.resolve(category)` |
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

---

## 6. Validation

```bash
set PYTHONPATH=.
python scripts/check_plugin_catalog.py
pytest tests/unit/core/plugins tests/unit/integrations/test_external_plugin.py -q
```

---

## 7. Do not confuse with Nexus runtime plugins

| Mechanism | Purpose |
|-----------|---------|
| Tier-0 catalog plugins (this guide) | Register integrations, tools, skills in catalog |
| `RuntimePlugin` / `plugin_bootstrap.py` | Nexus middleware, metrics, persistence hooks |

Agents consume **tools** via `ToolRegistry` and **skills** via `SkillResolver` → `allowed_tools`. Agents MUST NOT import vendor SDKs or integration slugs directly when a catalog tool exists.
