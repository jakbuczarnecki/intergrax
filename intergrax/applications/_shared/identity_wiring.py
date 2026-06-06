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
    if profile.require_api_key:
        key = (os.environ.get(profile.api_key_env) or "").strip()
        if not key and resolve_identity_provider_backend(integration_profile) is None:
            raise ValueError(
                f"{profile.api_key_env} is required when identity_profile.require_api_key=true "
                "and no identity_provider integration is configured"
            )
    state = HarnessAuthState(identity_provider=resolve_identity_provider_backend(integration_profile))
    app.state.harness_auth = state
    apply_harness_auth_middleware(app)
    return state
