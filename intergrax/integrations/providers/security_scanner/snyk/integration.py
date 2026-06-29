# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Snyk security scanner integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
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
    Single public Snyk security scanner entrypoint.

    Legacy catalog factory (create_snyk_security_scanner) delegates to this class.
    """

    config: SnykSecurityScannerIntegrationConfig = SnykSecurityScannerIntegrationConfig()
    _client: SnykSecurityScannerClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> SnykSecurityScannerIntegration:
        integration = cls.for_provider(
            provider_id=SNYK_SECURITY_SCANNER_PROVIDER_ID,
            display_name="Snyk",
            config=SnykSecurityScannerIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Snyk integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SecurityScannerBackend.register(SnykSecurityScannerIntegration)
