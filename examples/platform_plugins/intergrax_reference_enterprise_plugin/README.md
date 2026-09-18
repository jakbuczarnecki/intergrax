# Intergrax reference enterprise plugin

One Python distribution contributing **four independent domain surfaces** through separate setuptools entry-point groups:

```text
intergrax-reference-enterprise-plugin
        |
        +-- intergrax.tools                  -> ReferenceEnterpriseEchoToolPlugin
        +-- intergrax.skills                 -> ReferenceEnterprisePackSkillPlugin
        +-- intergrax.context                -> ReferenceEnterpriseContextPlugin
        +-- intergrax.tool_invocation_patterns -> ReferenceEnterpriseSinglePassPattern
```

There is **no** universal `PlatformPlugin.execute()`, no shared runtime dispatcher, and no package-global DI container. Each domain owns its contract, discovery, and materialization path.

## Install

```bash
uv pip install ./examples/platform_plugins/intergrax_reference_enterprise_plugin
```

Build wheel:

```bash
uv build --wheel --project examples/platform_plugins/intergrax_reference_enterprise_plugin
```

## Enable discovery

Tier-0 catalogs (tools, skills, context):

```python
from intergrax.core.catalog_bootstrap import bootstrap_catalogs

bootstrap_catalogs(discover_entry_points=True)
```

Tool invocation patterns load lazily by id:

```python
from intergrax.tools.invocation_pattern.registry import load_tool_invocation_pattern

pattern = load_tool_invocation_pattern("reference_enterprise_single_pass")
```

Set `RuntimeConfig.tool_invocation_pattern_id` (or host override) so Nexus resolves the public pattern internally.

## Qualification

`installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`. Hosts enable subsets via domain profiles (`ToolProfile`, `SkillProfile`, `ContextProfile`, `ToolInvocationMode`).

Offline proof: `tests/unit/platform_plugins/test_reference_enterprise_plugin.py`

## Platform manifest

`pyproject.toml` includes `[tool.intergrax.plugin]` capability descriptors for package-level inventory - domain qualification remains per surface.

Canon: [`docs/project/architecture/PLATFORM_PLUGINS.md`](../../../docs/project/architecture/PLATFORM_PLUGINS.md) §21
