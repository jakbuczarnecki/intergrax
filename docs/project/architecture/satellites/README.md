# Architecture satellites (`docs/project/architecture/satellites/`)

**Parent directory:** [`../`](..) — token-efficient **hubs** (one file per domain).

This folder holds **extended architecture canon** split out of hubs for Cursor context budget (F4). Nothing here is deleted from the platform — it is the **full-depth** reference for § blocks that are too large for default agent reads.

| Layer | Path | Cursor indexing | When to load |
|-------|------|-----------------|--------------|
| **Hub** | `docs/project/architecture/<DOMAIN>.md` | Yes (default) | Implement / audit — read **Cursor read scope** block only |
| **Satellite** | `docs/project/architecture/satellites/<DOMAIN>_*.md` | No (`.cursorignore`) | Explicit `@` or `Read` when read-scope, TOC, or audit cites extended § |

**Naming:**

| Suffix | Typical contents |
|--------|------------------|
| `_extended_depth.md` | Main extended § (numbered sections moved out of hub) |
| `_runtime_extended.md` | UAEP §42.x+ runtime depth |
| `_production_gates.md` | §40+ production / release gates |
| `_provider_catalog.md`, `_selection_and_plugins.md`, … | Domain-specific catalogs and reference tables |

**Regenerate splits:**

```bash
uv run python scripts/docs/split_domain_architecture.py [DOMAIN ...]
uv run python scripts/maintenance/check_arch_hub_size.py
uv run python scripts/docs/verify_arch_split_content.py
```
