# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

from intergrax.skills.registry.plugin_register import register_skill_plugin
from intergrax.skills.registry.shipped_plugins import SHIPPED_SKILL_BUNDLE_IDS, SHIPPED_SKILL_PLUGINS

_BOOTSTRAPPED = False


def register_default_skills(
    *,
    bundle_ids: Sequence[str] | None = None,
    override: bool = False,
) -> None:
    """Idempotent registration of shipped skill bundles via ``SkillPlugin``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override and bundle_ids is None:
        return

    allowed = None
    if bundle_ids is not None:
        allowed = {bid.strip().lower() for bid in bundle_ids if bid.strip()}
        unknown = allowed - SHIPPED_SKILL_BUNDLE_IDS
        if unknown:
            raise ValueError(f"Unknown skill bundle_id(s): {', '.join(sorted(unknown))}")

    for plugin_type in SHIPPED_SKILL_PLUGINS:
        manifest = plugin_type.skill_bundle_manifest()
        if allowed is not None and manifest.bundle_id not in allowed:
            continue
        register_skill_plugin(plugin_type, override=override)

    if bundle_ids is None:
        _BOOTSTRAPPED = True


def reset_default_skills_for_tests() -> None:
    global _BOOTSTRAPPED
    from intergrax.core.catalog_bootstrap import reset_tier0_catalog_bootstrap_for_tests
    from intergrax.skills.registry.catalog import clear_skill_catalog

    clear_skill_catalog()
    reset_tier0_catalog_bootstrap_for_tests()
    _BOOTSTRAPPED = False
