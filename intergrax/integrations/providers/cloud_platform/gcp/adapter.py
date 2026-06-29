# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GCP cloud platform adapter — ``CloudPlatform`` facade (no google.auth here)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Optional

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationCategory,
    UnknownIntegrationCategoryError,
    normalize_category,
)
from intergrax.integrations.providers.cloud_platform.gcp.config import GcpIntegrationConfig
from intergrax.integrations.registry.slugs import CLOUD_PLATFORM_DEFAULTS

_GCP_DEFAULTS = CLOUD_PLATFORM_DEFAULTS["gcp"]


class _GcpCloudPlatform:
    """
    Catalog facade over Google ``Credentials``.

    Instantiate via ``create_gcp_cloud_platform()`` — not from agent code.
    """

    slug = "gcp"

    def __init__(
        self,
        config: GcpIntegrationConfig,
        credentials: Any,
        *,
        resolved_project_id: str = "",
    ) -> None:
        self._config = config
        self._credentials = credentials
        self._resolved_project_id = resolved_project_id.strip()

    @property
    def config(self) -> GcpIntegrationConfig:
        return self._config

    @property
    def credentials(self) -> Any:
        """Underlying Google credentials for Tier-3 infrastructure wiring."""
        return self._credentials

    @property
    def project_id(self) -> Optional[str]:
        project = self._config.project_id or self._resolved_project_id
        return project or None

    @property
    def default_region(self) -> Optional[str]:
        region = self._config.region.strip()
        return region or None

    def resolve(self, category: str) -> Optional[str]:
        try:
            normalized = normalize_category(category)
        except UnknownIntegrationCategoryError:
            return None
        slug = _GCP_DEFAULTS.get(normalized)
        return slug if slug is not None else None

    def health(self) -> HealthStatus:
        try:
            if not attribute_access.optional(self._credentials, "valid", False):
                raise RuntimeError("GCP credentials are not valid")
            detail_parts: list[str] = []
            if self.project_id:
                detail_parts.append(f"project={self.project_id}")
            detail_parts.append("credentials=valid")
            return HealthStatus(
                slug=self.slug,
                healthy=True,
                detail="; ".join(detail_parts),
            )
        except Exception as exc:  # noqa: BLE001 — health probe must not raise
            return HealthStatus(slug=self.slug, healthy=False, detail=str(exc))

    def default_slug_for(self, category: IntegrationCategory) -> Optional[str]:
        slug = _GCP_DEFAULTS.get(category)
        return slug if slug is not None else None
