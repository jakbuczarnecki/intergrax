# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semgrep security scanner integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
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
    Single public Semgrep security scanner entrypoint.

    Legacy catalog factory (create_semgrep_security_scanner) delegates to this class.
    """

    config: SemgrepSecurityScannerIntegrationConfig = SemgrepSecurityScannerIntegrationConfig()
    _client: SemgrepSecurityScannerClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> SemgrepSecurityScannerIntegration:
        integration = cls.for_provider(
            provider_id=SEMGREP_SECURITY_SCANNER_PROVIDER_ID,
            display_name="Semgrep",
            config=SemgrepSecurityScannerIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Semgrep integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

SecurityScannerBackend.register(SemgrepSecurityScannerIntegration)
