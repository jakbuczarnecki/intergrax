# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure cloud platform adapter — ``CloudPlatform`` facade (no SDK here)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Optional

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationCategory,
    UnknownIntegrationCategoryError,
    normalize_category,
)
from intergrax.integrations.providers.cloud_platform.azure.config import AZURE_MANAGEMENT_SCOPE, AzureIntegrationConfig
from intergrax.integrations.registry.slugs import CLOUD_PLATFORM_DEFAULTS

_AZURE_DEFAULTS = CLOUD_PLATFORM_DEFAULTS["azure"]


class AzureCloudPlatform:
    """
    Catalog facade over an Azure ``TokenCredential``.

    Instantiate via ``create_azure_cloud_platform()`` — not from agent code.
    """

    slug = "azure"

    def __init__(self, config: AzureIntegrationConfig, credential: Any) -> None:
        self._config = config
        self._credential = credential

    @property
    def config(self) -> AzureIntegrationConfig:
        return self._config

    @property
    def credential(self) -> Any:
        """Underlying Azure credential for Tier-3 infrastructure wiring."""
        return self._credential

    @property
    def default_region(self) -> Optional[str]:
        location = self._config.location.strip()
        return location or None

    def resolve(self, category: str) -> Optional[str]:
        try:
            normalized = normalize_category(category)
        except UnknownIntegrationCategoryError:
            return None
        slug = _AZURE_DEFAULTS.get(normalized)
        return slug if slug is not None else None

    def health(self) -> HealthStatus:
        try:
            token = self._credential.get_token(AZURE_MANAGEMENT_SCOPE)
            expires_on = attribute_access.optional(token, "expires_on", None)
            detail = f"token_expires={expires_on}" if expires_on is not None else "ok"
            if self._config.subscription_id:
                detail = f"subscription={self._config.subscription_id}; {detail}"
            return HealthStatus(slug=self.slug, healthy=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 — health probe must not raise
            return HealthStatus(slug=self.slug, healthy=False, detail=str(exc))

    def default_slug_for(self, category: IntegrationCategory) -> Optional[str]:
        slug = _AZURE_DEFAULTS.get(category)
        return slug if slug is not None else None
