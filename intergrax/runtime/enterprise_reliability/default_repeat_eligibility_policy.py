# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default external-effect repeat policy — fail-closed (GR-7-A5)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatPolicy,
    ExternalEffectRepeatPolicyDecision,
    ExternalEffectRepeatPolicyRequest,
)

__all__ = [
    "DEFAULT_DENY_EXTERNAL_EFFECT_REPEAT_POLICY_ID",
    "DefaultDenyExternalEffectRepeatPolicy",
    "default_deny_external_effect_repeat_policy",
]

DEFAULT_DENY_EXTERNAL_EFFECT_REPEAT_POLICY_ID = "default_deny_external_effect_repeat"


class DefaultDenyExternalEffectRepeatPolicy:
    """Production default: never authorize idempotent repeat without explicit policy."""

    @property
    def policy_id(self) -> str:
        return DEFAULT_DENY_EXTERNAL_EFFECT_REPEAT_POLICY_ID

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        _ = request
        return ExternalEffectRepeatPolicyDecision(allow_repeat=False)


_DEFAULT = DefaultDenyExternalEffectRepeatPolicy()


def default_deny_external_effect_repeat_policy() -> ExternalEffectRepeatPolicy:
    """Shared deny-all policy for runtime composition."""
    return _DEFAULT
