# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 integration catalog bootstrap helpers (Phase P-Ext.1.11)."""

from __future__ import annotations

from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.integrations.registry.bootstrap import IntegrationPreset


def bootstrap_application_integration_catalog(
    *,
    integration_preset: IntegrationPreset = "full",
    discover_entry_points: bool | None = None,
) -> None:
    """
    Register shipped integration catalog (and optional entry-point plugins).

    Prefer this over bare ``register_default_integrations()`` in Tier-3 hosts.
    """
    discover = discover_plugins_enabled() if discover_entry_points is None else discover_entry_points
    bootstrap_catalogs(
        register_shipped=True,
        integration_preset=integration_preset,
        discover_entry_points=discover,
    )
