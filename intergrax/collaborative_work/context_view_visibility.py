# © Artur Czarnecki. All rights reserved.

"""MP-5C principal ContextView visibility policy (Collaborative Work).

Flow: ContextViewRequest → MP-1 effective authority → visibility policy → decision.
No retrieval, composition, or domain store access.
"""

from __future__ import annotations

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.repository import AuthorityDelegationRepository
from intergrax.contracts.collaborative_work import (
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewOperationScope,
    ContextViewRequest,
    ContextViewScope,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_visibility_policy import (
    CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
    ContextViewCategoryDenial,
    ContextViewCategoryDenialReason,
    ContextViewPolicyDecision,
    ContextViewPolicyDenialReason,
    ContextViewPolicyOutcome,
    ContextViewVisibilityPolicy,
    ContextViewVisibilityPolicyInput,
    DefaultContextViewVisibilityPolicyConfig,
    fail_closed_context_view_policy_decision,
)
from intergrax.contracts.runtime_policy import PolicyAction


def build_context_view_effective_authority_request(
    request: ContextViewRequest,
    *,
    required_authority_scope: str = CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
) -> EffectiveAuthorityRequest:
    """Map a ContextView request to MP-1 effective-authority input."""
    resource_scope: str | None = None
    operation_scope = request.scope.operation_scope
    if operation_scope is not None:
        resource_scope = operation_scope.resource_scope

    if request.authority_request is not None:
        auth = request.authority_request
        scopes = tuple(
            dict.fromkeys((*auth.requested_authority_scopes, required_authority_scope)),
        )
        return auth.model_copy(
            update={
                "requested_authority_scopes": scopes,
                "resource_scope": resource_scope if resource_scope is not None else auth.resource_scope,
            },
        )

    return EffectiveAuthorityRequest(
        tenant_id=request.scope.tenant_id,
        workspace_id=request.scope.workspace_id,
        acting_principal_id=request.acting_principal_id,
        requested_authority_scopes=(required_authority_scope,),
        delegator_principal_id=request.delegator_principal_id,
        resource_scope=resource_scope,
        membership=request.membership,
        membership_resolution_mode=request.membership_resolution_mode,
        delegation=request.delegation,
    )


class ContextViewVisibilityEvaluator:
    """Resolve MP-1 authority, then delegate eligibility to an injected policy."""

    def __init__(
        self,
        *,
        authority_resolver: CollaborativeWorkAuthorityResolver,
        visibility_policy: ContextViewVisibilityPolicy,
        delegation_repository: AuthorityDelegationRepository | None = None,
    ) -> None:
        self._authority_resolver = authority_resolver
        self._visibility_policy = visibility_policy
        self._delegation_repository = delegation_repository

    def evaluate(self, request: ContextViewRequest) -> ContextViewPolicyDecision:
        policy_id = self._visibility_policy.policy_id
        scope = request.scope

        if not request.acting_principal_id.strip():
            return fail_closed_context_view_policy_decision(
                policy_id=policy_id,
                effective_scope=scope,
                denial_reason=ContextViewPolicyDenialReason.MISSING_AUTHORITY_RESOLUTION,
            )

        if request.membership_resolution_mode is MembershipResolutionMode.LOCATOR:
            if request.membership is None and request.delegator_principal_id is None:
                return fail_closed_context_view_policy_decision(
                    policy_id=policy_id,
                    effective_scope=scope,
                    denial_reason=ContextViewPolicyDenialReason.MISSING_AUTHORITY_RESOLUTION,
                )

        authority_request = build_context_view_effective_authority_request(
            request,
            required_authority_scope=CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
        )
        authority_decision = self._authority_resolver.resolve(authority_request)
        if authority_decision.decision.action is not PolicyAction.ALLOW:
            return fail_closed_context_view_policy_decision(
                policy_id=policy_id,
                effective_scope=scope,
                denial_reason=ContextViewPolicyDenialReason.AUTHORITY_DENIED,
            )

        isolation_denial = _scope_isolation_denial(request)
        if isolation_denial is not None:
            return fail_closed_context_view_policy_decision(
                policy_id=policy_id,
                effective_scope=scope,
                denial_reason=ContextViewPolicyDenialReason.SCOPE_ISOLATION,
                denied_categories=isolation_denial,
            )

        delegation_scopes = _load_authoritative_delegation_scopes(
            request=request,
            delegation_repository=self._delegation_repository,
        )
        if request.delegator_principal_id is not None and delegation_scopes is None:
            return fail_closed_context_view_policy_decision(
                policy_id=policy_id,
                effective_scope=scope,
                denial_reason=ContextViewPolicyDenialReason.AUTHORITY_DENIED,
            )

        policy_input = ContextViewVisibilityPolicyInput(
            request=request,
            effective_authority=authority_decision,
            authoritative_delegation_scopes=delegation_scopes,
        )
        return self._visibility_policy.evaluate(policy_input)


def _load_authoritative_delegation_scopes(
    *,
    request: ContextViewRequest,
    delegation_repository: AuthorityDelegationRepository | None,
) -> tuple[str, ...] | None:
    if request.delegator_principal_id is None:
        return None
    if request.delegation is None or delegation_repository is None:
        return None
    locator = request.delegation
    authoritative = delegation_repository.get(
        tenant_id=locator.tenant_id,
        workspace_id=locator.workspace_id,
        delegation_id=locator.delegation_id,
    )
    if authoritative is None:
        return None
    return authoritative.authority_scopes


def _scope_isolation_denial(
    request: ContextViewRequest,
) -> tuple[ContextViewCategoryDenial, ...] | None:
    scope = request.scope
    operation_scope = scope.operation_scope
    if operation_scope is not None and operation_scope.operation_id != request.operation_id:
        return tuple(
            ContextViewCategoryDenial(
                category=category,
                reason=ContextViewCategoryDenialReason.OPERATION_SCOPE_MISMATCH,
            )
            for category in request.requested_categories
        )
    return None


class DefaultContextViewVisibilityPolicy:
    """Platform default fail-closed visibility eligibility policy."""

    def __init__(self, config: DefaultContextViewVisibilityPolicyConfig | None = None) -> None:
        self._config = config or DefaultContextViewVisibilityPolicyConfig()

    @property
    def policy_id(self) -> str:
        return self._config.policy_id

    def evaluate(self, policy_input: ContextViewVisibilityPolicyInput) -> ContextViewPolicyDecision:
        request = policy_input.request
        scope = request.scope
        acting = request.acting_principal_id

        if policy_input.effective_authority.decision.action is not PolicyAction.ALLOW:
            return fail_closed_context_view_policy_decision(
                policy_id=self.policy_id,
                effective_scope=scope,
                denial_reason=ContextViewPolicyDenialReason.AUTHORITY_DENIED,
            )

        effective_scope = _least_context_scope(request)
        visibility_classes = _eligible_visibility_classes(
            request=request,
            delegation_scopes=policy_input.authoritative_delegation_scopes,
            config=self._config,
        )

        eligible: list[ContextViewCategory] = []
        denied: list[ContextViewCategoryDenial] = []
        for category in request.requested_categories:
            category_denial = _category_eligibility_denial(
                category=category,
                request=request,
                visibility_classes=visibility_classes,
                delegation_scopes=policy_input.authoritative_delegation_scopes,
                config=self._config,
            )
            if category_denial is None:
                eligible.append(category)
            else:
                denied.append(category_denial)

        if not eligible:
            return fail_closed_context_view_policy_decision(
                policy_id=self.policy_id,
                effective_scope=effective_scope,
                denial_reason=ContextViewPolicyDenialReason.NO_ELIGIBLE_CATEGORIES,
                denied_categories=tuple(denied),
            )

        private_principal: str | None = None
        if ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL in visibility_classes:
            private_principal = acting

        return ContextViewPolicyDecision(
            outcome=ContextViewPolicyOutcome.ALLOW,
            policy_id=self.policy_id,
            effective_scope=effective_scope,
            eligible_categories=tuple(eligible),
            denied_categories=tuple(denied),
            eligible_visibility_classes=visibility_classes,
            private_visibility_principal_id=private_principal,
        )


def _least_context_scope(request: ContextViewRequest) -> ContextViewScope:
    scope = request.scope
    operation_scope = scope.operation_scope
    if operation_scope is None:
        return scope
    narrowed_operation = ContextViewOperationScope(
        operation_id=operation_scope.operation_id,
        resource_scope=operation_scope.resource_scope,
    )
    return ContextViewScope(
        tenant_id=scope.tenant_id,
        workspace_id=scope.workspace_id,
        work_item_id=scope.work_item_id,
        operation_scope=narrowed_operation,
    )


def _eligible_visibility_classes(
    *,
    request: ContextViewRequest,
    delegation_scopes: tuple[str, ...] | None,
    config: DefaultContextViewVisibilityPolicyConfig,
) -> tuple[ContextViewVisibilityClass, ...]:
    classes: list[ContextViewVisibilityClass] = [ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL]

    if delegation_scopes is None:
        classes.append(ContextViewVisibilityClass.WORKSPACE_SHARED)
        classes.append(ContextViewVisibilityClass.PLATFORM_VISIBLE)
    else:
        scope_set = set(delegation_scopes)
        if config.workspace_shared_delegation_scope in scope_set:
            classes.append(ContextViewVisibilityClass.WORKSPACE_SHARED)
        if config.platform_visible_delegation_scope in scope_set:
            classes.append(ContextViewVisibilityClass.PLATFORM_VISIBLE)
        classes.append(ContextViewVisibilityClass.DELEGATED_VISIBLE)

    if request.scope.work_item_id is not None:
        classes.append(ContextViewVisibilityClass.WORK_ITEM)

    return tuple(dict.fromkeys(classes))


def _category_eligibility_denial(
    *,
    category: ContextViewCategory,
    request: ContextViewRequest,
    visibility_classes: tuple[ContextViewVisibilityClass, ...],
    delegation_scopes: tuple[str, ...] | None,
    config: DefaultContextViewVisibilityPolicyConfig,
) -> ContextViewCategoryDenial | None:
    if category is ContextViewCategory.MEMORY:
        return None

    if category is ContextViewCategory.KNOWLEDGE:
        if ContextViewVisibilityClass.WORKSPACE_SHARED not in visibility_classes:
            if delegation_scopes is not None:
                return ContextViewCategoryDenial(
                    category=category,
                    reason=ContextViewCategoryDenialReason.DELEGATION_SCOPE_INSUFFICIENT,
                )
            return ContextViewCategoryDenial(
                category=category,
                reason=ContextViewCategoryDenialReason.AUTHORITY_INSUFFICIENT,
            )
        return None

    if category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
        if ContextViewVisibilityClass.WORKSPACE_SHARED not in visibility_classes:
            return ContextViewCategoryDenial(
                category=category,
                reason=ContextViewCategoryDenialReason.DELEGATION_SCOPE_INSUFFICIENT
                if delegation_scopes is not None
                else ContextViewCategoryDenialReason.AUTHORITY_INSUFFICIENT,
            )
        return None

    if category is ContextViewCategory.COLLABORATIVE_WORK:
        if request.scope.work_item_id is None:
            if ContextViewVisibilityClass.WORKSPACE_SHARED not in visibility_classes:
                return ContextViewCategoryDenial(
                    category=category,
                    reason=ContextViewCategoryDenialReason.DELEGATION_SCOPE_INSUFFICIENT
                    if delegation_scopes is not None
                    else ContextViewCategoryDenialReason.AUTHORITY_INSUFFICIENT,
                )
            return None
        if ContextViewVisibilityClass.WORK_ITEM not in visibility_classes:
            return ContextViewCategoryDenial(
                category=category,
                reason=ContextViewCategoryDenialReason.WORK_ITEM_SCOPE_REQUIRED,
            )
        return None

    return ContextViewCategoryDenial(
        category=category,
        reason=ContextViewCategoryDenialReason.UNSUPPORTED_CATEGORY,
    )
