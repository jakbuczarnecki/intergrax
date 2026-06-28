# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete lab JSON integration bundle — composition root for laboratory interaction intake.

All runtime wiring MUST use this module or
``profile.resolve(IntegrationCategory.INTERACTION_SURFACE)`` with ``"lab_json"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.providers.interaction_surface.lab_json.config import LabJsonIntegrationConfig
from intergrax.integrations.providers.interaction_surface.lab_json.opens import open_lab_json_interaction_surface
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter


@dataclass(frozen=True)
class LabJsonIntegrationBundle:
    """Lab JSON interaction surface + config."""

    config: LabJsonIntegrationConfig
    interaction_surface: InteractionAdapter


def resolve_lab_json_config(**overrides: object) -> LabJsonIntegrationConfig:
    return LabJsonIntegrationConfig.from_env(**overrides)


def create_lab_json_integration(
    *,
    default_source: Optional[str] = None,
    interaction_adapter: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> LabJsonIntegrationBundle:
    """Single entry point for lab JSON interaction intake."""
    overrides: dict[str, object] = dict(config_overrides)
    if default_source is not None:
        overrides["default_source"] = default_source

    config = resolve_lab_json_config(**overrides)
    interaction = open_lab_json_interaction_surface(
        config,
        implementation=interaction_adapter,
    )

    return LabJsonIntegrationBundle(
        config=config,
        interaction_surface=interaction,
    )


def create_lab_json_interaction_surface(
    *,
    interaction_adapter: Optional[InteractionAdapter] = None,
    **config_overrides: object,
) -> InteractionAdapter:
    """Catalog factory for ``"lab_json"`` / ``INTERACTION_SURFACE``."""
    return create_lab_json_integration(
        interaction_adapter=interaction_adapter,
        **config_overrides,
    ).interaction_surface

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.interaction_surface.lab_json.integration import (
    LAB_JSON_INTERACTION_SURFACE_PROVIDER_ID,
    LabJsonInteractionSurfaceIntegration,
    LabJsonInteractionSurfaceIntegrationConfig,
    LabJsonInteractionSurfaceClient,
)


def create_lab_json_interaction_surface_integration(
    *,
    client: LabJsonInteractionSurfaceClient | None = None,
    enabled: bool = False,
) -> LabJsonInteractionSurfaceIntegration:
    """
    Build a contract-based Lab Json interaction surface integration.

    The legacy facade (create_lab_json_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Lab Json interaction surface integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LabJsonInteractionSurfaceIntegration.from_client(client, enabled=enabled)
    return LabJsonInteractionSurfaceIntegration.for_provider(
        provider_id=LAB_JSON_INTERACTION_SURFACE_PROVIDER_ID,
        display_name="Lab Json",
        config=LabJsonInteractionSurfaceIntegrationConfig(enabled=enabled),
    )
