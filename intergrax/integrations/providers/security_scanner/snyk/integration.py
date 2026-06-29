# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Snyk security scanner integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SNYK_SECURITY_SCANNER_PROVIDER_ID = "snyk"


class SnykSecurityScannerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Snyk security scanner integration."""

    pass


SnykSecurityScannerClient = SecurityScannerBackend

class SnykSecurityScannerIntegration(SecurityScannerIntegrationContract):
    """
    Single public Snyk security scanner entrypoint.

    Legacy catalog factory (create_snyk_security_scanner) owns catalog behavior; legacy factories use from_client().
    """

    config: SnykSecurityScannerIntegrationConfig = SnykSecurityScannerIntegrationConfig()
    _client: SnykSecurityScannerClient | None = PrivateAttr(default=None)
    

    def scan_image(self, image_ref):
        return self._require_client().scan_image(image_ref)

    def scan_repo(self, repo_path):
        return self._require_client().scan_repo(repo_path)

    def _require_client(self) -> SecurityScannerBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: SnykSecurityScannerClient,
        *,
        enabled: bool = False,
    ) -> SnykSecurityScannerIntegration:
        integration = cls.for_provider(
            provider_id=SNYK_SECURITY_SCANNER_PROVIDER_ID,
            display_name="Snyk",
            config=SnykSecurityScannerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SnykSecurityScannerClient | None:
        return self._client

SecurityScannerBackend.register(SnykSecurityScannerIntegration)
