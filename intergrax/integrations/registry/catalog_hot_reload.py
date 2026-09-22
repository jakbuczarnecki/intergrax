# © Artur Czarnecki. All rights reserved.

"""Integration catalog hot-reload contracts (AUDIT-IDEAL-13.2).

Production consequential reload must use ``CatalogHotReloadService`` (applications composition).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CatalogHotReloadReport(BaseModel):
    """Legacy diagnostic report — not authorization evidence."""

    schema_version: str = "1.0.0"
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    reloaded: bool


def reload_integration_catalog(*_args: object, **_kwargs: object) -> CatalogHotReloadReport:
    """Blocked production bypass — use governed ``CatalogHotReloadService.reload``."""
    raise RuntimeError(
        "reload_integration_catalog is not a legal production path; "
        "use CatalogHotReloadService via catalog hot-reload composition wiring"
    )
