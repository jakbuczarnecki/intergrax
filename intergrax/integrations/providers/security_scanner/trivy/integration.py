# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trivy security scanner integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TRIVY_SECURITY_SCANNER_PROVIDER_ID = "trivy"


class TrivySecurityScannerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Trivy security scanner integration."""

    pass


@runtime_checkable
class TrivySecurityScannerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TrivySecurityScannerIntegration(SecurityScannerIntegrationContract):
    """
    Trivy security scanner integration.

    The legacy facade (create_trivy_security_scanner) remains separate and backward-compatible.
    """

    config: TrivySecurityScannerIntegrationConfig = TrivySecurityScannerIntegrationConfig()
    _client: TrivySecurityScannerClient | None = PrivateAttr(default=None)

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
