# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral tenant connection authorization contract (PRODUCT-5B)."""

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.models import JsonValue


class TenantConnectionAuthMode(StrEnum):
    OAUTH_DELEGATED = "oauth_delegated"
    MANUAL_CREDENTIAL_BINDING = "manual_credential_binding"


class TenantConnectionAuthQualification(StrEnum):
    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"


@dataclass(frozen=True, slots=True)
class TenantConnectionAuthProviderDescriptor:
    provider_id: str
    integration_kind: IntegrationCategory
    auth_mode: TenantConnectionAuthMode
    safe_display_name: str
    supported_scopes_summary: str
    qualification: TenantConnectionAuthQualification = TenantConnectionAuthQualification.QUALIFIED


@dataclass(frozen=True, slots=True)
class TenantConnectionAuthBeginResult:
    authorization_url: str | None
    code_verifier: str | None
    correlation_state: str
    required_user_action: Literal["redirect", "present_manual_instructions"]
    manual_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class TenantConnectionAuthExchangeResult:
    credential_bundle_json: str
    connected_principal_ref: str | None


@dataclass(frozen=True, slots=True)
class TenantConnectionAuthManualBindResult:
    credential_bundle_json: str
    connected_principal_ref: str | None


def generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for OAuth S256 PKCE."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def generate_correlation_state() -> str:
    return secrets.token_urlsafe(32)


@runtime_checkable
class TenantConnectionAuthProvider(Protocol):
    provider_id: str
    integration_kind: IntegrationCategory
    auth_mode: TenantConnectionAuthMode
    qualification: TenantConnectionAuthQualification

    def describe(self) -> TenantConnectionAuthProviderDescriptor:
        ...

    def begin_authorization(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        reconnect_connection_ref: str | None,
    ) -> TenantConnectionAuthBeginResult:
        ...

    def exchange_authorization_code(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        authorization_code: str,
        code_verifier: str,
        correlation_state: str,
    ) -> TenantConnectionAuthExchangeResult:
        ...

    def bind_manual_credentials(
        self,
        *,
        tenant_id: str,
        credential_payload: Mapping[str, JsonValue],
    ) -> TenantConnectionAuthManualBindResult:
        ...

    def build_secret_free_config(
        self,
        *,
        tenant_id: str,
        reconnect_connection: object | None,
    ) -> Mapping[str, JsonValue]:
        ...

    def revoke_remote_credentials(
        self,
        *,
        tenant_id: str,
        credential_bundle_json: str,
    ) -> None:
        ...


class TenantConnectionAuthProviderRegistry:
    """Route provider_id to a TenantConnectionAuthProvider adapter."""

    def __init__(
        self,
        providers: Mapping[str, TenantConnectionAuthProvider] = {},
    ) -> None:
        self._providers: dict[str, TenantConnectionAuthProvider] = {}
        for provider_id, provider in providers.items():
            self.register(provider)

    def register(self, provider: TenantConnectionAuthProvider) -> None:
        cleaned = provider.provider_id.strip()
        if not cleaned:
            raise ValueError("provider_id must be a non-empty string")
        if cleaned in self._providers:
            raise ValueError("tenant connection auth provider is already registered")
        self._providers[cleaned] = provider

    def get(self, provider_id: str) -> TenantConnectionAuthProvider | None:
        return self._providers.get(provider_id.strip())

    def list_descriptors(self) -> tuple[TenantConnectionAuthProviderDescriptor, ...]:
        return tuple(provider.describe() for provider in self._providers.values())


__all__ = [
    "TenantConnectionAuthBeginResult",
    "TenantConnectionAuthExchangeResult",
    "TenantConnectionAuthManualBindResult",
    "TenantConnectionAuthMode",
    "TenantConnectionAuthProvider",
    "TenantConnectionAuthProviderDescriptor",
    "TenantConnectionAuthProviderRegistry",
    "TenantConnectionAuthQualification",
    "generate_correlation_state",
    "generate_pkce_pair",
]
