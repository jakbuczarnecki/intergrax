# © Artur Czarnecki. All rights reserved.

"""Load security defense plugins from entry points (Phase SEC-EXT-2)."""

from __future__ import annotations

from intergrax.core.plugins.discovery import (
    EP_SECURITY_DEFENSES,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_value,
)
from intergrax.runtime.security.defense_plugin import SecurityDefensePlugin
from intergrax.runtime.security.defense_registry import register_security_defense_plugin


def load_security_defense_plugins(*, discover_entry_points: bool = True) -> int:
    """Register plugins from ``intergrax.security_defenses`` entry points."""
    if not discover_entry_points:
        return 0
    count = 0
    for spec in iter_entry_point_specs(EP_SECURITY_DEFENSES):
        loaded = load_entry_point_value(spec.value)
        plugin = instantiate_entry_point_target(loaded)
        if not isinstance(plugin, SecurityDefensePlugin):
            raise TypeError(
                f"Security defense entry point {spec.name!r} must return SecurityDefensePlugin"
            )
        register_security_defense_plugin(plugin, override=True)
        count += 1
    return count
