# © Artur Czarnecki. All rights reserved.

"""Typed fail-closed security configuration errors and defaults (PDC-D6-R008/R014/R024)."""

from __future__ import annotations

from typing import Final

SIGNING_SECRET_MISSING: Final = "signing_secret_missing"
SIGNING_SECRET_ENV_INVALID: Final = "signing_secret_env_invalid"
LAB_OPT_IN_FORBIDDEN_IN_PRODUCT: Final = "lab_opt_in_forbidden_in_product_profile"
AUDIT_SINK_MISSING: Final = "audit_sink_missing"
AUDIT_REGIONS_INVALID: Final = "audit_regions_invalid"
AUDIT_REPLICATION_FAILED: Final = "audit_replication_failed"
RETRIEVAL_POISONING_PROFILE_MALFORMED: Final = "retrieval_poisoning_profile_malformed"
PLUGIN_TRUST_POLICY_MISSING: Final = "plugin_trust_policy_missing"
DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE: Final = "demo_seed_forbidden_in_strict_profile"


class SecurityConfigurationError(ValueError):
    """A required security configuration item is missing or malformed.

    Raised instead of silently defaulting. The message names the configuration item
    and a stable ``reason_code``; it never contains secret material.
    """

    def __init__(self, *, item: str, reason_code: str, message: str) -> None:
        self.item = item
        self.reason_code = reason_code
        super().__init__(f"{item}: {message} [{reason_code}]")


def retrieval_poisoning_defense_enabled_or_default(profile: object | None) -> bool:
    """Resolve the retrieval-poisoning toggle; the defense is ON unless explicitly disabled.

    * absent profile (``None``) -> enabled
    * profile without the attribute -> enabled
    * only a literal ``False`` disables the defense (explicit profile value)
    * any other value (malformed, non-bool) is treated as enabled (fail closed)
    """
    if profile is None:
        return True
    from intergrax.utils import attribute_access

    value = attribute_access.optional(profile, "retrieval_poisoning_defense_enabled", True)
    return value is not False


__all__ = [
    "AUDIT_REGIONS_INVALID",
    "AUDIT_REPLICATION_FAILED",
    "AUDIT_SINK_MISSING",
    "DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE",
    "LAB_OPT_IN_FORBIDDEN_IN_PRODUCT",
    "PLUGIN_TRUST_POLICY_MISSING",
    "RETRIEVAL_POISONING_PROFILE_MALFORMED",
    "SIGNING_SECRET_ENV_INVALID",
    "SIGNING_SECRET_MISSING",
    "SecurityConfigurationError",
    "retrieval_poisoning_defense_enabled_or_default",
]
