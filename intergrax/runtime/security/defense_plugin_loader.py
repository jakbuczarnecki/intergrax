# © Artur Czarnecki. All rights reserved.

"""Load security defense plugins from entry points (Phase SEC-EXT-2)."""

from __future__ import annotations

from importlib.metadata import entry_points

from intergrax.runtime.security.defense_plugin import SecurityDefensePlugin
from intergrax.runtime.security.defense_registry import register_security_defense_plugin


def load_security_defense_plugins(*, discover_entry_points: bool = True) -> int:
    """Register plugins from ``intergrax.security_defenses`` entry points."""
    if not discover_entry_points:
        return 0
    try:
        eps = entry_points(group="intergrax.security_defenses")
    except TypeError:  # pragma: no cover — Python 3.11
        eps = entry_points().select(group="intergrax.security_defenses")
    count = 0
    for ep in eps:
        loaded = ep.load()
        if isinstance(loaded, type):
            plugin: SecurityDefensePlugin = loaded()
        else:
            plugin = loaded
        register_security_defense_plugin(plugin, override=True)
        count += 1
    return count
