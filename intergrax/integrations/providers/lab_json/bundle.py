# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete lab JSON integration bundle — composition root for laboratory interaction intake.

All runtime wiring MUST use this module or
``profile.resolve(IntegrationCategory.INTERACTION_SURFACE)`` with ``IntegrationSlug.LAB_JSON``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.providers.lab_json.config import LabJsonIntegrationConfig
from intergrax.integrations.providers.lab_json.opens import open_lab_json_interaction_surface
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
    """Catalog factory for ``IntegrationSlug.LAB_JSON`` / ``INTERACTION_SURFACE``."""
    return create_lab_json_integration(
        interaction_adapter=interaction_adapter,
        **config_overrides,
    ).interaction_surface
