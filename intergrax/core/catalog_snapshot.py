# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only snapshots of Tier-0 catalog registration state."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.registry.catalog import catalog_snapshot as integration_catalog_snapshot
from intergrax.skills.registry.catalog import iter_bundles as iter_skill_bundles
from intergrax.tools.registry.catalog import iter_bundles as iter_tool_bundles


@dataclass(frozen=True)
class CatalogSnapshot:
    integration_slugs: tuple[str, ...]
    tool_bundle_ids: tuple[str, ...]
    tool_ids: tuple[str, ...]
    skill_bundle_ids: tuple[str, ...]
    skill_ids: tuple[str, ...]


def snapshot_catalogs() -> CatalogSnapshot:
    """Return sorted slug/bundle identifiers currently registered in memory."""
    integration_slugs = tuple(sorted(integration_catalog_snapshot().keys()))
    tool_bundle_ids: list[str] = []
    tool_ids: list[str] = []
    for entry in iter_tool_bundles():
        tool_bundle_ids.append(entry.bundle_id)
        tool_ids.extend(entry.tool_ids)
    skill_bundle_ids: list[str] = []
    skill_ids: list[str] = []
    for entry in iter_skill_bundles():
        skill_bundle_ids.append(entry.bundle_id)
        skill_ids.extend(entry.skill_ids)
    return CatalogSnapshot(
        integration_slugs=integration_slugs,
        tool_bundle_ids=tuple(sorted(tool_bundle_ids)),
        tool_ids=tuple(sorted(set(tool_ids))),
        skill_bundle_ids=tuple(sorted(skill_bundle_ids)),
        skill_ids=tuple(sorted(set(skill_ids))),
    )
