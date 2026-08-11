# © Artur Czarnecki. All rights reserved.

"""Authoritative effective-authority state resolver (COLLAB-WORK-1C).

Resolves only the Collaborative Work portion of effective authority:

    acting principal
      ∩ authoritative active WorkspaceMembership
      ∩ authoritative active/valid AuthorityDelegation when acting for another principal
      ∩ requested collaborative authority scopes (structural delegation subset checks)

Workspace policy, resource policy, runtime/tool policy, principal base authority, and
membership role→capability mapping are **out of scope** for this slice. When structural
collaborative evidence is valid but final authority cannot be proven, the resolver fails
closed with ``SCOPE_ONLY_INSUFFICIENT``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from intergrax.collaborative_work.repository import (
    AuthorityDelegationRepository,
    WorkspaceMembershipRepository,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    DelegationStatus,
    EffectiveAuthorityDecision,
    EffectiveAuthorityDenialReason,
    EffectiveAuthorityRequest,
    MembershipStatus,
    WorkspaceMembership,
    fail_closed_effective_authority_decision,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

_POLICY_RULE_SCOPE_ONLY_INSUFFICIENT = (
    "collaborative_work.effective_authority.scope_only_insufficient"
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _authoritative_clock_result(raw_now: object) -> datetime | None:
    """Return a timezone-aware datetime with usable offset, or None when invalid."""
    if not isinstance(raw_now, datetime):
        return None
    if raw_now.tzinfo is None:
        return None
    if raw_now.tzinfo.utcoffset(raw_now) is None:
        return None
    return raw_now


class CollaborativeWorkAuthorityResolver:
    """Resolve collaborative authority state from authoritative repository records.

    Caller-supplied embedded ``WorkspaceMembership`` and ``AuthorityDelegation`` objects
    are locator hints only; authority-bearing fields are always reloaded from repository
    ports and never trusted from the request payload.
    """

    def __init__(
        self,
        *,
        membership_repository: WorkspaceMembershipRepository,
        delegation_repository: AuthorityDelegationRepository,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._membership_repository = membership_repository
        self._delegation_repository = delegation_repository
        self._clock = clock or _utc_now

    def resolve(self, request: EffectiveAuthorityRequest) -> EffectiveAuthorityDecision:
        if not request.acting_principal_id.strip():
            return fail_closed_effective_authority_decision(
                reason="acting principal is required",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_ACTING_PRINCIPAL,
            )

        membership_denial = self._resolve_membership(request)
        if membership_denial is not None:
            return membership_denial

        if request.delegator_principal_id is not None:
            delegation_denial = self._resolve_delegation(request)
            if delegation_denial is not None:
                return delegation_denial

        return self._deny_unresolved_authority_source()

    def _resolve_membership(
        self,
        request: EffectiveAuthorityRequest,
    ) -> EffectiveAuthorityDecision | None:
        if request.membership is None:
            return fail_closed_effective_authority_decision(
                reason="workspace membership locator is required",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP,
            )

        locator = request.membership
        authoritative = self._membership_repository.get(
            tenant_id=locator.tenant_id,
            workspace_id=locator.workspace_id,
            membership_id=locator.membership_id,
        )
        if authoritative is None:
            return fail_closed_effective_authority_decision(
                reason="authoritative workspace membership not found",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP,
            )

        if not self._membership_matches_request(request, authoritative):
            return fail_closed_effective_authority_decision(
                reason="authoritative membership does not match request scope",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP,
            )

        if authoritative.status is not MembershipStatus.ACTIVE:
            return fail_closed_effective_authority_decision(
                reason="workspace membership is not active",
                denial_reason=EffectiveAuthorityDenialReason.MEMBERSHIP_NOT_ACTIVE,
            )

        return None

    @staticmethod
    def _membership_matches_request(
        request: EffectiveAuthorityRequest,
        membership: WorkspaceMembership,
    ) -> bool:
        return (
            membership.tenant_id == request.tenant_id
            and membership.workspace_id == request.workspace_id
            and membership.principal_id == request.acting_principal_id
        )

    def _resolve_delegation(
        self,
        request: EffectiveAuthorityRequest,
    ) -> EffectiveAuthorityDecision | None:
        if request.delegation is None:
            return fail_closed_effective_authority_decision(
                reason="authority delegation locator is required for delegated acting",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_DELEGATION,
            )

        locator = request.delegation
        authoritative = self._delegation_repository.get(
            tenant_id=locator.tenant_id,
            workspace_id=locator.workspace_id,
            delegation_id=locator.delegation_id,
        )
        if authoritative is None:
            return fail_closed_effective_authority_decision(
                reason="authoritative authority delegation not found",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_DELEGATION,
            )

        if not self._delegation_matches_request(request, authoritative):
            return fail_closed_effective_authority_decision(
                reason="authoritative delegation does not match delegated acting request",
                denial_reason=EffectiveAuthorityDenialReason.MISSING_DELEGATION,
            )

        if authoritative.status is not DelegationStatus.ACTIVE:
            return fail_closed_effective_authority_decision(
                reason="authority delegation is not active",
                denial_reason=EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE,
            )

        now = _authoritative_clock_result(self._clock())
        if now is None:
            return fail_closed_effective_authority_decision(
                reason=(
                    "authority cannot be safely established because authoritative "
                    "temporal context is unavailable or invalid"
                ),
                denial_reason=EffectiveAuthorityDenialReason.AUTHORITY_TEMPORAL_CONTEXT_UNAVAILABLE,
            )
        if authoritative.valid_from is not None and now < authoritative.valid_from:
            return fail_closed_effective_authority_decision(
                reason="authority delegation is not yet valid",
                denial_reason=EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE,
            )
        if authoritative.valid_until is not None and now >= authoritative.valid_until:
            return fail_closed_effective_authority_decision(
                reason="authority delegation has expired",
                denial_reason=EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE,
            )

        delegation_scopes = set(authoritative.authority_scopes)
        for scope in request.requested_authority_scopes:
            if scope not in delegation_scopes:
                return fail_closed_effective_authority_decision(
                    reason="requested authority scope exceeds delegation authority",
                    denial_reason=EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE,
                )

        resource_denial = self._check_resource_scope(request, authoritative)
        if resource_denial is not None:
            return resource_denial

        return None

    @staticmethod
    def _delegation_matches_request(
        request: EffectiveAuthorityRequest,
        delegation: AuthorityDelegation,
    ) -> bool:
        if request.delegator_principal_id is None:
            return False
        return (
            delegation.tenant_id == request.tenant_id
            and delegation.workspace_id == request.workspace_id
            and delegation.delegator_principal_id == request.delegator_principal_id
            and delegation.delegate_principal_id == request.acting_principal_id
        )

    @staticmethod
    def _check_resource_scope(
        request: EffectiveAuthorityRequest,
        delegation: AuthorityDelegation,
    ) -> EffectiveAuthorityDecision | None:
        if delegation.resource_scope is None:
            return None
        if request.resource_scope is None:
            return fail_closed_effective_authority_decision(
                reason="request resource scope required for resource-limited delegation",
                denial_reason=EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE,
            )
        if request.resource_scope != delegation.resource_scope:
            return fail_closed_effective_authority_decision(
                reason="request resource scope is incompatible with delegation resource scope",
                denial_reason=EffectiveAuthorityDenialReason.INSUFFICIENT_DELEGATION_SCOPE,
            )
        return None

    @staticmethod
    def _deny_unresolved_authority_source() -> EffectiveAuthorityDecision:
        return EffectiveAuthorityDecision(
            decision=PolicyDecision(
                action=PolicyAction.DENY,
                reason=(
                    "collaborative membership and delegation evidence validated; "
                    "principal base authority and policy composition are not yet resolved"
                ),
                policy_rule_id=_POLICY_RULE_SCOPE_ONLY_INSUFFICIENT,
            ),
            denial_reason=EffectiveAuthorityDenialReason.SCOPE_ONLY_INSUFFICIENT,
        )
