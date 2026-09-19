# © Artur Czarnecki. All rights reserved.

"""MP-6E read authorization — MP-1 effective authority then injectable read policy."""

from __future__ import annotations

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.repository import AuthorityDelegationRepository
from intergrax.contracts.collaborative_activity_read import (
    COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
    CollaborativeActivityReadAuthorizationDecision,
    CollaborativeActivityReadAuthorizationOutcome,
    CollaborativeActivityReadAuthorizationPolicy,
    CollaborativeActivityReadAuthorizationPolicyInput,
    CollaborativeActivityReadDenialReason,
    CollaborativeActivityReadRequest,
    DefaultCollaborativeActivityReadAuthorizationPolicyConfig,
    fail_closed_collaborative_activity_read_decision,
)
from intergrax.contracts.collaborative_work import (
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.runtime_policy import PolicyAction


def build_collaborative_activity_read_effective_authority_request(
    request: CollaborativeActivityReadRequest,
    *,
    required_authority_scope: str = COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
) -> EffectiveAuthorityRequest:
    query = request.query
    if request.authority_request is not None:
        auth = request.authority_request
        scopes = tuple(
            dict.fromkeys((*auth.requested_authority_scopes, required_authority_scope)),
        )
        return auth.model_copy(
            update={
                "requested_authority_scopes": scopes,
                "tenant_id": query.tenant_id,
                "workspace_id": query.workspace_id,
                "acting_principal_id": request.acting_principal_id,
                "resource_scope": auth.resource_scope,
            },
        )

    return EffectiveAuthorityRequest(
        tenant_id=query.tenant_id,
        workspace_id=query.workspace_id,
        acting_principal_id=request.acting_principal_id,
        requested_authority_scopes=(required_authority_scope,),
        delegator_principal_id=request.delegator_principal_id,
        resource_scope=None,
        membership=request.membership,
        membership_resolution_mode=request.membership_resolution_mode,
        delegation=request.delegation,
    )


class CollaborativeActivityReadAuthorizationEvaluator:
    """Resolve MP-1 authority, then delegate to injectable read authorization policy."""

    def __init__(
        self,
        *,
        authority_resolver: CollaborativeWorkAuthorityResolver,
        read_authorization_policy: CollaborativeActivityReadAuthorizationPolicy,
        delegation_repository: AuthorityDelegationRepository | None = None,
    ) -> None:
        self._authority_resolver = authority_resolver
        self._read_authorization_policy = read_authorization_policy
        self._delegation_repository = delegation_repository

    def evaluate(
        self,
        request: CollaborativeActivityReadRequest,
    ) -> CollaborativeActivityReadAuthorizationDecision:
        policy_id = self._read_authorization_policy.policy_id
        query = request.query

        if not request.acting_principal_id.strip():
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.MISSING_AUTHORITY_RESOLUTION,
            )

        if request.membership_resolution_mode is MembershipResolutionMode.LOCATOR:
            if request.membership is None and request.delegator_principal_id is None:
                return fail_closed_collaborative_activity_read_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityReadDenialReason.MISSING_AUTHORITY_RESOLUTION,
                )

        if request.membership is not None:
            membership = request.membership
            if membership.tenant_id.strip() != query.tenant_id.strip():
                return fail_closed_collaborative_activity_read_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityReadDenialReason.SCOPE_ISOLATION,
                )
            if membership.workspace_id.strip() != query.workspace_id.strip():
                return fail_closed_collaborative_activity_read_decision(
                    policy_id=policy_id,
                    denial_reason=CollaborativeActivityReadDenialReason.SCOPE_ISOLATION,
                )

        authority_request = build_collaborative_activity_read_effective_authority_request(request)
        if authority_request.tenant_id.strip() != query.tenant_id.strip():
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.SCOPE_ISOLATION,
            )
        if authority_request.workspace_id.strip() != query.workspace_id.strip():
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.SCOPE_ISOLATION,
            )

        authority_decision = self._authority_resolver.resolve(authority_request)
        if authority_decision.decision.action is not PolicyAction.ALLOW:
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.AUTHORITY_DENIED,
            )

        policy_input = CollaborativeActivityReadAuthorizationPolicyInput(
            request=request,
            effective_authority=authority_decision,
        )
        return self._read_authorization_policy.evaluate(policy_input)


class DefaultCollaborativeActivityReadAuthorizationPolicy:
    """Platform default — workspace-scoped read when MP-1 collaborative slice allows."""

    def __init__(
        self,
        config: DefaultCollaborativeActivityReadAuthorizationPolicyConfig | None = None,
    ) -> None:
        self._config = config or DefaultCollaborativeActivityReadAuthorizationPolicyConfig()

    @property
    def policy_id(self) -> str:
        return self._config.policy_id

    def evaluate(
        self,
        policy_input: CollaborativeActivityReadAuthorizationPolicyInput,
    ) -> CollaborativeActivityReadAuthorizationDecision:
        request = policy_input.request
        query = request.query
        policy_id = self.policy_id

        if policy_input.effective_authority.decision.action is not PolicyAction.ALLOW:
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.AUTHORITY_DENIED,
            )

        auth_request = build_collaborative_activity_read_effective_authority_request(request)
        if auth_request.tenant_id.strip() != query.tenant_id.strip():
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.SCOPE_ISOLATION,
            )
        if auth_request.workspace_id.strip() != query.workspace_id.strip():
            return fail_closed_collaborative_activity_read_decision(
                policy_id=policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.SCOPE_ISOLATION,
            )

        return CollaborativeActivityReadAuthorizationDecision(
            outcome=CollaborativeActivityReadAuthorizationOutcome.ALLOW,
            policy_id=policy_id,
            authorized_query=query,
        )


def build_default_collaborative_activity_read_authorization_policy(
    *,
    config: DefaultCollaborativeActivityReadAuthorizationPolicyConfig | None = None,
) -> DefaultCollaborativeActivityReadAuthorizationPolicy:
    return DefaultCollaborativeActivityReadAuthorizationPolicy(config=config)
