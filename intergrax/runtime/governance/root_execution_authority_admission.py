# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime/Governance trusted root execution authority admission (AW-5A seam).

Mints trusted ``ParentExecutionAuthority`` for canonical root execution intake.
Autonomous Work must consume this port — it must not mint trusted authority.
"""

from __future__ import annotations

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionPort,
    RootExecutionAuthorityAdmissionRequest,
    RootExecutionAuthorityAdmissionResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision


class RootExecutionAuthorityAdmissionService:
    """Default runtime admission — collaborative ALLOW is necessary but not sufficient."""

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        collaborative = request.effective_authority_decision
        decision = collaborative.decision
        if decision.action is PolicyAction.DENY:
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
                policy_decision=decision,
            )
        if decision.action is PolicyAction.REQUIRE_HUMAN:
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN,
                policy_decision=decision,
            )
        if decision.action is PolicyAction.ESCALATE:
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.ESCALATE,
                policy_decision=decision,
            )
        if decision.action is PolicyAction.MODIFY:
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
                policy_decision=decision,
            )
        if decision.action is not PolicyAction.ALLOW:
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE,
                policy_decision=decision,
            )

        trusted = ParentExecutionAuthority.scoped(request.collaborative_authority_scopes)
        return RootExecutionAuthorityAdmissionResult(
            disposition=RootExecutionAuthorityAdmissionDisposition.ALLOWED,
            trusted_parent_execution_authority=trusted,
            policy_decision=decision,
        )


class DenyingRootExecutionAuthorityAdmission:
    """Test/reference adapter that always denies runtime admission."""

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        del request
        return RootExecutionAuthorityAdmissionResult(
            disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
            policy_decision=PolicyDecision(
                action=PolicyAction.DENY,
                reason="runtime_authority_denied",
            ),
        )


class UnavailableRootExecutionAuthorityAdmission:
    """Fail-closed adapter when runtime admission is unavailable."""

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        del request
        return RootExecutionAuthorityAdmissionResult(
            disposition=RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE,
        )
