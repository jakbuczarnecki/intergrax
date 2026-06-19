# © Artur Czarnecki. All rights reserved.

"""Bootstrap shipped security defense bundles and optional entry points (Phase SEC-BUNDLE-3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.security.defense_plugin_loader import load_security_defense_plugins
from intergrax.runtime.security.defense_registry import list_shipped_defense_bundle_ids


@dataclass(frozen=True, slots=True)
class SecurityBootstrapResult:
    shipped_bundle_ids: tuple[str, ...]
    entry_point_plugins: int


def bootstrap_security_providers(*, discover_entry_points: bool = False) -> SecurityBootstrapResult:
    """
    Register shipped defense bundles (always available) and optional EP plugins.

    Shipped bundles are registered at import time via ``defense_registry``;
    this function loads third-party entry points when requested.
    """
    loaded = load_security_defense_plugins(discover_entry_points=discover_entry_points)
    return SecurityBootstrapResult(
        shipped_bundle_ids=list_shipped_defense_bundle_ids(),
        entry_point_plugins=loaded,
    )
