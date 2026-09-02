# Acme reference Vendor Knowledge plugin

Installable reference package for third-party **Vendor Knowledge** providers. It demonstrates the full contribution ABI (adapter, connection factory, discovery, indexed materializer) without coupling to Intergrax core.

**Not** a Tier-0 catalog plugin - discovery uses `intergrax.vendor_knowledge.providers` and host builder composition with `discover_entry_points=True`.

## Install

From the repository root:

```bash
uv pip install ./examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin
```

Or build a wheel:

```bash
uv build --wheel --project examples/platform_plugins/intergrax_reference_vendor_knowledge_plugin
uv pip install dist/acme_reference_vk_plugin-*.whl
```

## Entry point

| Group | Name | Target |
|-------|------|--------|
| `intergrax.vendor_knowledge.providers` | `acme_reference` | `acme_reference_vk_plugin.contribution:build_acme_reference_contribution` |

## Enable discovery

Vendor Knowledge external providers load only when the host enables EP discovery, for example:

```python
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
)

catalog = build_default_vendor_knowledge_contribution_catalog(
    discover_entry_points=True,
)
```

`installed` ≠ `discovered` ≠ `enabled` ≠ `production-qualified`.

## Qualification

Full qualification evidence remains in repository tests:

- `tests/unit/runtime/vendor_knowledge/test_acme_reference_plugin.py`
- `tests/integration/vendor_knowledge/test_acme_reference_external_provider_proof.py`

Author guide: [`docs/project/technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md`](../../../docs/project/technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md)
