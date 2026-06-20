# Architecture hub satellites

**Parent directory:** [`../`](../) — domain architecture hubs (`<DOMAIN>.md`)

Load **one** satellite per session when a task or audit gap cites extended § blocks.

| Pattern | When to load |
|---------|----------------|
| `<DOMAIN>_extended_depth.md` | §22–§39 depth (ACP, Tier-3, platform layers) |
| `<DOMAIN>_production_gates.md` | §40+ production / release gates |
| `TOOLS_selection_and_plugins.md` | Tool selection + plugin model |
| `TOOLS_catalog_and_index.md` | Full tool catalog tables |

**Regenerate splits:** `uv run python scripts/split_domain_architecture.py [DOMAIN ...]`

**CI gate:** `uv run python scripts/check_arch_hub_size.py`

**Audit compact context:** [`../../guides/audit_slices/`](../../guides/audit_slices/)
