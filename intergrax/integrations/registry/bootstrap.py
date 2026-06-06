# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register default integration providers (Phase M.4+, P-Ext lazy preset)."""

from __future__ import annotations

from typing import Literal

from intergrax.integrations.registry.bootstrap_core import register_core_integrations

IntegrationPreset = Literal["full", "core"]

_BOOTSTRAPPED = False


def register_default_integrations(
    *,
    preset: IntegrationPreset = "full",
    override: bool = False,
) -> None:
    """
    Idempotent registration of shipped integration providers.

    ``preset="core"`` registers lab essentials only (~12 slugs).
    ``preset="full"`` registers the complete shipped catalog (135 slugs).
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override:
        return

    register_core_integrations(override=override)
    if preset == "full":
        from intergrax.integrations.registry.bootstrap_extended import register_extended_integrations

        register_extended_integrations(override=override)

    _BOOTSTRAPPED = True


def reset_default_integrations_state() -> None:
    """Test helper — allow re-bootstrap after ``clear_catalog()``."""
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False
