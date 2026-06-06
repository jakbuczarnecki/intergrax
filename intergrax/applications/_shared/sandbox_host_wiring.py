# © Artur Czarnecki. All rights reserved.

"""Resolve hosted sandbox sessions from ``sandbox_host`` integration category."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession


def resolve_hosted_sandbox_session(
    integration_profile: IntegrationProfile,
    *,
    tenant_id: str,
    task_id: str,
) -> SandboxExecCapable | None:
    """Open a cloud sandbox session when ``sandbox_host`` is configured."""
    backend = integration_profile.instance_for_category(IntegrationCategory.SANDBOX_HOST)
    if backend is None:
        slug = integration_profile.slug_for_category(IntegrationCategory.SANDBOX_HOST)
        if slug is None:
            return None
        backend = integration_profile.resolve(IntegrationCategory.SANDBOX_HOST)
    if not isinstance(backend, SandboxHostBackend):
        return None
    return HostedSandboxSession.open(
        backend,
        tenant_id=tenant_id,
        task_id=task_id,
    )
