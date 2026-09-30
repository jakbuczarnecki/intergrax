# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed physical effect executor boundary (AW-7C-CLOSURE-R1-R1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.credential import ScopedCredentialResolutionResult
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationEffectRequest,
    ScopedAdaptedIntegrationOperationEvidence,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapable


@runtime_checkable
class ScopedAdaptedIntegrationEffectExecutionIngress(Protocol):
    """Executor ingress — validated request plus sanctioned sandbox and credential resources."""

    @property
    def effect_request(self) -> ScopedAdaptedIntegrationEffectRequest: ...

    @property
    def execution_id(self) -> str: ...

    @property
    def sandbox_resource(self) -> SandboxSecurityCapable: ...

    @property
    def credential_resolution(self) -> ScopedCredentialResolutionResult: ...

    @property
    def sandbox_session_id(self) -> int: ...

    @property
    def credential_use_evidence_grant_id(self) -> str: ...

    @property
    def credential_use_evidence_fingerprint(self) -> str: ...


@runtime_checkable
class ScopedAdaptedIntegrationEffectExecutor(Protocol):
    """Canonical physical adapted-integration effect boundary (Integrations-owned, not a plugin SPI)."""

    def execute(
        self,
        ingress: ScopedAdaptedIntegrationEffectExecutionIngress,
    ) -> ScopedAdaptedIntegrationOperationEvidence: ...


__all__ = [
    "ScopedAdaptedIntegrationEffectExecutionIngress",
    "ScopedAdaptedIntegrationEffectExecutor",
]
