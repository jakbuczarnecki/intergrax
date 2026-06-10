# © Artur Czarnecki. All rights reserved.

"""Critical action signing wiring for product hosts (AUDIT-IDEAL-4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.security.critical_action_signing import (
    CriticalActionKind,
    CriticalActionPayload,
    CriticalActionSignature,
    sign_critical_action,
    verify_critical_action_signature,
)


@dataclass(frozen=True, slots=True)
class CriticalActionSigningWiring:
    enabled: bool
    bootstrap_signature: CriticalActionSignature | None


def resolve_critical_action_signing_wiring(
    env: ApplicationEnvironmentProfile,
) -> CriticalActionSigningWiring:
    """Enable HMAC signing for critical actions on product hosts."""
    identity = env.identity_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return CriticalActionSigningWiring(enabled=False, bootstrap_signature=None)
    if not identity.critical_action_signing_enabled:
        return CriticalActionSigningWiring(enabled=False, bootstrap_signature=None)

    secret = os.environ.get(identity.critical_action_signing_secret_env, "harness-dev-signing-key")
    payload = CriticalActionPayload(
        action_id=f"{env.profile_id}:bootstrap",
        action_kind=CriticalActionKind.SECURITY_CONFIG_CHANGE,
        tenant_id=env.profile_id,
        actor_id="harness.bootstrap",
        resource="identity.signing",
        details={"host": env.profile_id},
    )
    signature = sign_critical_action(secret=secret, payload=payload)
    if not verify_critical_action_signature(secret=secret, payload=payload, signature=signature):
        return CriticalActionSigningWiring(enabled=False, bootstrap_signature=None)
    return CriticalActionSigningWiring(enabled=True, bootstrap_signature=signature)
