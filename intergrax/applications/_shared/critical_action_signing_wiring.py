# © Artur Czarnecki. All rights reserved.

"""Critical action signing wiring for product hosts (AUDIT-IDEAL-4.1, PDC-D6-R008)."""

from __future__ import annotations

import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.security.security_configuration import (
    LAB_OPT_IN_FORBIDDEN_IN_PRODUCT,
    SIGNING_SECRET_ENV_INVALID,
    SIGNING_SECRET_MISSING,
    SecurityConfigurationError,
)
from intergrax.runtime.security.critical_action_signing import (
    CriticalActionKind,
    CriticalActionPayload,
    CriticalActionSignature,
    sign_critical_action,
    verify_critical_action_signature,
)

_ITEM = "critical_action_signing"


@dataclass(frozen=True, slots=True)
class LabEphemeralSigningOptIn:
    """Named, explicit lab opt-in: sign with a per-process random key (never a literal).

    Only honoured for ``ApplicationProfile.LAB``. A product profile that is handed this
    opt-in is a configuration error. ``reason`` is recorded for operators and must be
    non-empty.
    """

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("LabEphemeralSigningOptIn.reason must be non-empty")


@dataclass(frozen=True, slots=True)
class CriticalActionSigningWiring:
    enabled: bool
    bootstrap_signature: CriticalActionSignature | None


def _bootstrap(env: ApplicationEnvironmentProfile, secret: str) -> CriticalActionSigningWiring:
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
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=SIGNING_SECRET_MISSING,
            message="bootstrap signature failed self-verification",
        )
    return CriticalActionSigningWiring(enabled=True, bootstrap_signature=signature)


def resolve_critical_action_signing_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    environ: Mapping[str, str] | None = None,
    lab_opt_in: LabEphemeralSigningOptIn | None = None,
) -> CriticalActionSigningWiring:
    """Enable HMAC signing for critical actions; fail closed when the secret is missing.

    * Product profile with signing enabled: the secret is read from the configured
      environment variable at call time (use site). A missing or blank value raises
      :class:`SecurityConfigurationError`; there is no default value.
    * Lab profile: signing stays disabled unless the caller passes an explicit
      :class:`LabEphemeralSigningOptIn`, which signs with a random per-process key.
    * A lab opt-in on a product profile raises.
    """
    identity = env.identity_profile
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    if lab_opt_in is not None and is_product:
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=LAB_OPT_IN_FORBIDDEN_IN_PRODUCT,
            message="lab signing opt-in is not accepted on a product profile",
        )
    if not is_product:
        if lab_opt_in is None:
            return CriticalActionSigningWiring(enabled=False, bootstrap_signature=None)
        return _bootstrap(env, secrets.token_hex(32))
    if not identity.critical_action_signing_enabled:
        return CriticalActionSigningWiring(enabled=False, bootstrap_signature=None)

    secret_env = (identity.critical_action_signing_secret_env or "").strip()
    if not secret_env:
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=SIGNING_SECRET_ENV_INVALID,
            message="identity_profile.critical_action_signing_secret_env is blank",
        )
    source = environ if environ is not None else os.environ
    secret = source.get(secret_env)
    if secret is None or not secret.strip():
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=SIGNING_SECRET_MISSING,
            message=f"signing secret is not set in environment variable {secret_env!r}",
        )
    return _bootstrap(env, secret)
