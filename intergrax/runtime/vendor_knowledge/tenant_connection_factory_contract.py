# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed tenant-connection factory capability contracts (P1.7A)."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.credential import CredentialResolutionMode
from intergrax.runtime.vendor_knowledge.models import JsonValue


class EagerTenantConnectionIntegrationFactoryMixin:
    """Shared explicit eager credential mode for legacy provider factories."""

    @property
    def credential_resolution_mode(self) -> CredentialResolutionMode:
        return CredentialResolutionMode.RESOLVED_MATERIAL

    def credential_resolution_mode_for(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> CredentialResolutionMode:
        return self.credential_resolution_mode

    def create_late_bound_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> object:
        raise ValueError(
            "factory does not support late-bound credential resolution",
        )


def require_valid_credential_resolution_mode(mode: object) -> CredentialResolutionMode:
    """Fail fast when a factory declares an invalid credential resolution mode."""
    if not isinstance(mode, CredentialResolutionMode):
        raise TypeError("factory must declare a valid CredentialResolutionMode")
    return mode


__all__ = [
    "EagerTenantConnectionIntegrationFactoryMixin",
    "require_valid_credential_resolution_mode",
]
