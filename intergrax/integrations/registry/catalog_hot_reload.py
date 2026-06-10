# © Artur Czarnecki. All rights reserved.

"""Integration catalog hot-reload without host restart (AUDIT-IDEAL-13.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.integrations.registry.bootstrap import IntegrationPreset, register_default_integrations
from intergrax.integrations.registry.catalog import catalog_snapshot


class CatalogHotReloadReport(BaseModel):
    schema_version: str = "1.0.0"
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    reloaded: bool


def reload_integration_catalog(
    *,
    preset: IntegrationPreset = "core",
) -> CatalogHotReloadReport:
    """Re-register shipped integrations into the in-process catalog."""
    before_count = len(catalog_snapshot())
    register_default_integrations(preset=preset, override=True)
    after_count = len(catalog_snapshot())
    return CatalogHotReloadReport(
        before_count=before_count,
        after_count=after_count,
        reloaded=after_count >= before_count and after_count > 0,
    )
