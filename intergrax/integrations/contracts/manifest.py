# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration catalog manifests — plugin registration without central enums (§7.1.4)."""

from __future__ import annotations

from typing import Self, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus


class IntegrationManifest(BaseModel):
    """
    Declarative metadata for one integration provider (catalog row identity).

    Register with :func:`intergrax.integrations.registry.catalog.register_from_manifest`
  or :func:`intergrax.integrations.registry.plugin.register_integration_plugin`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str
    categories: tuple[IntegrationCategory, ...]
    status: IntegrationStatus = IntegrationStatus.STABLE
    env_prefix: str = ""
    description: str = ""
    requires_local_container: bool = False

    @field_validator("slug")
    @classmethod
    def _normalize_slug(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("integration slug must be non-empty")
        return normalized

    @field_validator("categories", mode="before")
    @classmethod
    def _coerce_categories(cls, value: object) -> tuple[IntegrationCategory, ...]:
        if value is None:
            return ()
        if isinstance(value, IntegrationCategory):
            return (value,)
        return tuple(
            item if isinstance(item, IntegrationCategory) else IntegrationCategory(str(item))
            for item in value
        )

    @property
    def primary_category(self) -> IntegrationCategory:
        if not self.categories:
            raise ValueError(f"integration manifest {self.slug!r} has no categories")
        return self.categories[0]

    def with_description(self, description: str) -> Self:
        return self.model_copy(update={"description": description})
