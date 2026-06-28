# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semgrep security scanner integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import SecurityScannerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SEMGREP_SECURITY_SCANNER_PROVIDER_ID = "semgrep"


class SemgrepSecurityScannerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Semgrep security scanner integration."""

    pass


@runtime_checkable
class SemgrepSecurityScannerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SemgrepSecurityScannerIntegration(SecurityScannerIntegrationContract):
    """
    Semgrep security scanner integration.

    The legacy facade (create_semgrep_security_scanner) remains separate and backward-compatible.
    """

    config: SemgrepSecurityScannerIntegrationConfig = SemgrepSecurityScannerIntegrationConfig()
    _client: SemgrepSecurityScannerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SemgrepSecurityScannerClient,
        *,
        enabled: bool = False,
    ) -> SemgrepSecurityScannerIntegration:
        integration = cls.for_provider(
            provider_id=SEMGREP_SECURITY_SCANNER_PROVIDER_ID,
            display_name="Semgrep",
            config=SemgrepSecurityScannerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SemgrepSecurityScannerClient | None:
        return self._client
