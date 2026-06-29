# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AWS cloud platform adapter — ``CloudPlatform`` facade (no boto3 here)."""

from __future__ import annotations

from typing import Any, Optional

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationCategory,
    UnknownIntegrationCategoryError,
    normalize_category,
)
from intergrax.integrations.providers.cloud_platform.aws.config import AwsIntegrationConfig
from intergrax.integrations.registry.slugs import CLOUD_PLATFORM_DEFAULTS

_AWS_DEFAULTS = CLOUD_PLATFORM_DEFAULTS["aws"]


class _AwsCloudPlatform:
    """
    Catalog facade over a boto3 ``Session``.

    Instantiate via ``create_aws_cloud_platform()`` — not from agent code.
    """

    slug = "aws"

    def __init__(self, config: AwsIntegrationConfig, session: Any) -> None:
        self._config = config
        self._session = session

    @property
    def config(self) -> AwsIntegrationConfig:
        return self._config

    @property
    def boto_session(self) -> Any:
        """Underlying boto3 session for Tier-3 infrastructure wiring."""
        return self._session

    @property
    def default_region(self) -> Optional[str]:
        region = self._config.region or self._session.region_name
        return str(region) if region else None

    def resolve(self, category: str) -> Optional[str]:
        try:
            normalized = normalize_category(category)
        except UnknownIntegrationCategoryError:
            return None
        slug = _AWS_DEFAULTS.get(normalized)
        return slug if slug is not None else None

    def health(self) -> HealthStatus:
        try:
            sts = self._session.client("sts", region_name=self.default_region)
            identity = sts.get_caller_identity()
            arn = identity.get("Arn") if isinstance(identity, dict) else None
            detail = str(arn) if arn else "ok"
            return HealthStatus(slug=self.slug, healthy=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 — health probe must not raise
            return HealthStatus(slug=self.slug, healthy=False, detail=str(exc))

    def default_slug_for(self, category: IntegrationCategory) -> Optional[str]:
        return _AWS_DEFAULTS.get(category)
