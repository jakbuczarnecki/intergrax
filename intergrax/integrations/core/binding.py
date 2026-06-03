# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration selection for Tier-3 profiles — manifest, plugin class, or live instance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.plugin import IntegrationPlugin, integration_manifest_for_plugin


class IntegrationBinding(BaseModel):
    """
    Normalized integration slot for :class:`IntegrationProfile`.

    Exactly one of:

    - ``manifest`` / ``plugin`` → resolve via catalog factory
    - ``instance`` → use pre-built integration object (no factory)
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    manifest: IntegrationManifest | None = None
    plugin: type[IntegrationPlugin] | None = None
    slug: str | None = None
    instance: Any | None = Field(default=None, repr=False)

    @classmethod
    def from_manifest(cls, manifest: IntegrationManifest) -> IntegrationBinding:
        return cls(manifest=manifest, plugin=None, slug=None, instance=None)

    @classmethod
    def from_plugin(cls, plugin: type[IntegrationPlugin]) -> IntegrationBinding:
        return cls(manifest=None, plugin=plugin, slug=None, instance=None)

    @classmethod
    def from_instance(cls, instance: Any) -> IntegrationBinding:
        return cls(manifest=None, plugin=None, slug=None, instance=instance)

    @classmethod
    def from_slug(cls, slug: str) -> IntegrationBinding:
        normalized = slug.strip().lower()
        if not normalized:
            raise ValueError("integration slug must be non-empty")
        return cls(manifest=None, plugin=None, slug=normalized, instance=None)

    def uses_factory(self) -> bool:
        return self.instance is None

    def resolved_slug(self) -> str | None:
        if self.instance is not None:
            return None
        if self.slug is not None:
            return self.slug
        if self.manifest is not None:
            return self.manifest.slug
        if self.plugin is not None:
            return integration_manifest_for_plugin(self.plugin).slug
        return None

    def catalog_manifest(self) -> IntegrationManifest | None:
        if self.manifest is not None:
            return self.manifest
        if self.plugin is not None:
            return integration_manifest_for_plugin(self.plugin)
        return None
