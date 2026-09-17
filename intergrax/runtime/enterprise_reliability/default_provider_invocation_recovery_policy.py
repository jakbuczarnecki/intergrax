# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default provider invocation recovery policy — fail-closed (GR-7-A7)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryPolicy,
    ProviderInvocationRecoveryPolicyDecision,
    ProviderInvocationRecoveryPolicyRequest,
)

__all__ = [
    "DEFAULT_FAIL_CLOSED_PROVIDER_INVOCATION_RECOVERY_POLICY_ID",
    "DefaultFailClosedProviderInvocationRecoveryPolicy",
    "default_fail_closed_provider_invocation_recovery_policy",
]

DEFAULT_FAIL_CLOSED_PROVIDER_INVOCATION_RECOVERY_POLICY_ID = (
    "default_fail_closed_provider_invocation_recovery"
)


class DefaultFailClosedProviderInvocationRecoveryPolicy:
    """Never auto-repeat; prefer reconcile then HITL within allowed_actions."""

    @property
    def policy_id(self) -> str:
        return DEFAULT_FAIL_CLOSED_PROVIDER_INVOCATION_RECOVERY_POLICY_ID

    def decide(
        self,
        request: ProviderInvocationRecoveryPolicyRequest,
    ) -> ProviderInvocationRecoveryPolicyDecision:
        allowed = request.allowed_actions
        if ProviderInvocationRecoveryAction.RECONCILE in allowed:
            return ProviderInvocationRecoveryPolicyDecision(
                selected_action=ProviderInvocationRecoveryAction.RECONCILE,
            )
        if ProviderInvocationRecoveryAction.ESCALATE_HITL in allowed:
            return ProviderInvocationRecoveryPolicyDecision(
                selected_action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            )
        return ProviderInvocationRecoveryPolicyDecision(
            selected_action=allowed[0],
        )


_DEFAULT = DefaultFailClosedProviderInvocationRecoveryPolicy()


def default_fail_closed_provider_invocation_recovery_policy() -> (
    ProviderInvocationRecoveryPolicy
):
    return _DEFAULT
