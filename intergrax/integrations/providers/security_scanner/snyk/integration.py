# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Snyk security scanner integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SNYK_SECURITY_SCANNER_PROVIDER_ID = "snyk"


class SnykSecurityScannerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Snyk security scanner integration."""

    pass


@runtime_checkable
class SnykSecurityScannerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SnykSecurityScannerIntegration(SecurityScannerIntegrationContract):
    """
    Snyk security scanner integration.

    The legacy facade (create_snyk_security_scanner) remains separate and backward-compatible.
    """

    config: SnykSecurityScannerIntegrationConfig = SnykSecurityScannerIntegrationConfig()
    _client: SnykSecurityScannerClient | None = PrivateAttr(default=None)

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
