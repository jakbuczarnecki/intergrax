# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trivy security scanner integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TRIVY_SECURITY_SCANNER_PROVIDER_ID = "trivy"


class TrivySecurityScannerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Trivy security scanner integration."""

    pass


TrivySecurityScannerClient = SecurityScannerBackend

class TrivySecurityScannerIntegration(SecurityScannerIntegrationContract):
    """
    Single public Trivy security scanner entrypoint.

    Legacy catalog factory (create_trivy_security_scanner) owns catalog behavior; legacy factories use from_client().
    """

    config: TrivySecurityScannerIntegrationConfig = TrivySecurityScannerIntegrationConfig()
    _client: TrivySecurityScannerClient | None = PrivateAttr(default=None)
    

    def scan_image(self, image_ref):
        return self._require_client().scan_image(image_ref)

    def scan_repo(self, repo_path):
        return self._require_client().scan_repo(repo_path)

    def health(self):
        return self._require_client().health()

    def _require_client(self) -> SecurityScannerBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: TrivySecurityScannerClient,
        *,
        enabled: bool = False,
    ) -> TrivySecurityScannerIntegration:
        integration = cls.for_provider(
            provider_id=TRIVY_SECURITY_SCANNER_PROVIDER_ID,
            display_name="Trivy",
            config=TrivySecurityScannerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TrivySecurityScannerClient | None:
        return self._client

SecurityScannerBackend.register(TrivySecurityScannerIntegration)
