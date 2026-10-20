# © Artur Czarnecki. All rights reserved.

"""Shared API-key fixtures for strict production HTTP qualification (P3-A2)."""

from __future__ import annotations

from dataclasses import replace

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.protocol import ApiHeaders

_QUAL_API_KEY: str = "qual-strict-http-key"
_QUAL_USER: str = "http-inv-qual-principal"
_QUAL_API_KEY_B: str = "qual-strict-http-key-b"
_QUAL_USER_B: str = "http-inv-qual-principal-unauthorized-b"
_WRONG_TENANT_KEY: str = "qual-wrong-tenant-key"


def strict_http_auth_headers(api_key: str = _QUAL_API_KEY) -> dict[str, str]:
    return {ApiHeaders.API_KEY: api_key}


def with_strict_http_api_keys(
    settings: GovernedContractorBackendSettings,
    *,
    execution_tenant_id: str,
    wrong_tenant_id: str | None = None,
) -> GovernedContractorBackendSettings:
    keys: dict[str, ApiKeyIdentity] = {
        _QUAL_API_KEY: ApiKeyIdentity(
            tenant_id=execution_tenant_id,
            user_id=_QUAL_USER,
            scopes=("*",),
        ),
        _QUAL_API_KEY_B: ApiKeyIdentity(
            tenant_id=execution_tenant_id,
            user_id=_QUAL_USER_B,
            scopes=("*",),
        ),
    }
    if wrong_tenant_id is not None:
        keys[_WRONG_TENANT_KEY] = ApiKeyIdentity(
            tenant_id=wrong_tenant_id,
            user_id="qual-wrong-tenant-user",
            scopes=("*",),
        )
    return replace(
        settings,
        identity_source="context_only",
        api_keys_map=keys,
    )


__all__ = [
    "_QUAL_API_KEY",
    "_QUAL_API_KEY_B",
    "_QUAL_USER",
    "_QUAL_USER_B",
    "_WRONG_TENANT_KEY",
    "strict_http_auth_headers",
    "with_strict_http_api_keys",
]
