# © Artur Czarnecki. All rights reserved.

"""Apply identity profile and optional OIDC provider to harness HTTP hosts."""

from __future__ import annotations

import os

from fastapi import FastAPI

from intergrax.applications._shared.harness_auth import HarnessAuthState, apply_harness_auth_middleware
from intergrax.applications.contracts.environment_profile import IdentityProfile
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.integrations.registry.profile import IntegrationProfile


def resolve_identity_provider_backend(
    integration_profile: IntegrationProfile | None,
) -> IdentityProviderBackend | None:
    """Resolve configured ``identity_provider`` integration when present."""
    if integration_profile is None:
        return None
    backend = integration_profile.instance_for_category(IntegrationCategory.IDENTITY_PROVIDER)
    if backend is None:
        slug = integration_profile.slug_for_category(IntegrationCategory.IDENTITY_PROVIDER)
        if slug is None:
            return None
        backend = integration_profile.resolve(IntegrationCategory.IDENTITY_PROVIDER)
    if isinstance(backend, IdentityProviderBackend):
        return backend
    return None


def wire_application_identity(
    app: FastAPI,
    profile: IdentityProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
) -> HarnessAuthState:
    """
    Configure harness API key middleware and optional OIDC identity provider.

    When ``require_api_key`` is true and no key is configured, startup should fail
    in the host factory (same guard as lab strict harness).
    """
    resolved_api_key = (os.environ.get(profile.api_key_env) or "").strip() or None
    identity_provider = resolve_identity_provider_backend(integration_profile)
    if profile.require_api_key:
        if not resolved_api_key and identity_provider is None:
            raise ValueError(
                f"{profile.api_key_env} is required when identity_profile.require_api_key=true "
                "and no identity_provider integration is configured"
            )
    api_key_service_id = profile.service_identities.get("harness") or "harness-api-key"
    api_key_tenant_id = (os.environ.get("INTERGRAX_HARNESS_TENANT_ID") or "").strip() or None
    state = HarnessAuthState(
        identity_provider=identity_provider,
        require_api_key=profile.require_api_key,
        resolved_api_key=resolved_api_key,
        tenant_required=profile.tenant_required,
        api_key_principal_tenant_id=api_key_tenant_id,
        api_key_principal_service_id=api_key_service_id,
    )
    app.state.harness_auth = state
    apply_harness_auth_middleware(app, require_auth=profile.require_api_key)
    return state
