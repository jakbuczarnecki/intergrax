# © Artur Czarnecki. All rights reserved.

"""MP-4D Approval authorization error contracts — fail-closed typed failures."""

from __future__ import annotations

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementResult


class ApprovalError(Exception):
    """Base Approval domain error."""


class ApprovalAuthorizationError(ApprovalError):
    """Approval mutation blocked by authority validation or enforcement."""


class ApprovalAuthorizationDenied(ApprovalAuthorizationError):
    """Approval mutation denied by the collaborative work enforcement gate."""

    def __init__(
        self, *, enforcement_result: CollaborativeWorkEnforcementResult
    ) -> None:
        self.enforcement_result = enforcement_result
        reason = (
            enforcement_result.composition.decision.reason or "authorization denied"
        )
        super().__init__(reason)


class ApprovalAuthorityContextError(ApprovalAuthorizationError):
    """Approval authority context could not be constructed or aligned."""
