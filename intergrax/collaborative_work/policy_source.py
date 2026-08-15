# © Artur Czarnecki. All rights reserved.

"""Authoritative workspace and resource policy evaluation (COLLAB-WORK-1F).

Provides deterministic, fail-closed ``PolicyDecision`` values for collaborative
workspace and resource policy layers. Does not execute side effects, does not
classify operation applicability, and does not duplicate Runtime Policy.
"""

from __future__ import annotations

from typing import Any

from intergrax.collaborative_work.repository import CollaborativePolicyRepository
from intergrax.contracts.collaborative_work import (
    CollaborativePolicyRule,
    CollaborativePolicyRuleStatus,
    PolicyCompositionLayer,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

_POLICY_RULE_WORKSPACE_MISSING = "collaborative_work.workspace_policy.missing"
_POLICY_RULE_WORKSPACE_INACTIVE = "collaborative_work.workspace_policy.inactive"
_POLICY_RULE_RESOURCE_MISSING = "collaborative_work.resource_policy.missing"
_POLICY_RULE_RESOURCE_INACTIVE = "collaborative_work.resource_policy.inactive"


def _audit_payload(
    *,
    layer: PolicyCompositionLayer,
    tenant_id: str,
    workspace_id: str,
    authority_scope: str,
    resource_scope: str | None,
    revision: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "layer": layer.value,
        "tenant_id": tenant_id,
        "workspace_id": workspace_id,
        "authority_scope": authority_scope,
    }
    if resource_scope is not None:
        payload["resource_scope"] = resource_scope
    if revision is not None:
        payload["rule_revision"] = revision
    return payload


def _deny_missing(
    *,
    layer: PolicyCompositionLayer,
    tenant_id: str,
    workspace_id: str,
    authority_scope: str,
    resource_scope: str | None,
    policy_rule_id: str,
) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason=f"{layer.value} rule missing",
        policy_rule_id=policy_rule_id,
        audit_payload=_audit_payload(
            layer=layer,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
            revision=None,
        ),
    )


def _deny_inactive(
    *,
    rule: CollaborativePolicyRule,
    layer: PolicyCompositionLayer,
    authority_scope: str,
    resource_scope: str | None,
    policy_rule_id: str,
) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason=f"{layer.value} rule inactive",
        policy_rule_id=policy_rule_id,
        audit_payload=_audit_payload(
            layer=layer,
            tenant_id=rule.tenant_id,
            workspace_id=rule.workspace_id,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
            revision=rule.revision,
        ),
    )


def _decision_from_rule(
    *,
    rule: CollaborativePolicyRule,
    layer: PolicyCompositionLayer,
    authority_scope: str,
    resource_scope: str | None,
) -> PolicyDecision:
    return PolicyDecision(
        action=rule.action,
        reason=f"{layer.value}:{rule.action.value}",
        policy_rule_id=rule.policy_rule_id,
        audit_payload=_audit_payload(
            layer=layer,
            tenant_id=rule.tenant_id,
            workspace_id=rule.workspace_id,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
            revision=rule.revision,
        ),
    )


class CollaborativePolicyEvaluator:
    """Evaluate authoritative workspace and resource policy rules."""

    def __init__(self, repository: CollaborativePolicyRepository) -> None:
        self._repository = repository

    def evaluate_workspace_policy(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_scope: str,
    ) -> PolicyDecision:
        """Return workspace policy decision for an exact authority scope."""
        layer = PolicyCompositionLayer.WORKSPACE_POLICY
        rule = self._repository.get_effective_rule(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            layer=layer,
            authority_scope=authority_scope,
        )
        return self._evaluate_rule(
            rule=rule,
            layer=layer,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_scope=authority_scope,
            resource_scope=None,
            missing_rule_id=_POLICY_RULE_WORKSPACE_MISSING,
            inactive_rule_id=_POLICY_RULE_WORKSPACE_INACTIVE,
        )

    def evaluate_resource_policy(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        resource_scope: str,
        authority_scope: str,
    ) -> PolicyDecision:
        """Return resource policy decision for an exact resource and authority scope."""
        layer = PolicyCompositionLayer.RESOURCE_POLICY
        rule = self._repository.get_effective_rule(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            layer=layer,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
        )
        return self._evaluate_rule(
            rule=rule,
            layer=layer,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
            missing_rule_id=_POLICY_RULE_RESOURCE_MISSING,
            inactive_rule_id=_POLICY_RULE_RESOURCE_INACTIVE,
        )

    @staticmethod
    def _evaluate_rule(
        *,
        rule: CollaborativePolicyRule | None,
        layer: PolicyCompositionLayer,
        tenant_id: str,
        workspace_id: str,
        authority_scope: str,
        resource_scope: str | None,
        missing_rule_id: str,
        inactive_rule_id: str,
    ) -> PolicyDecision:
        if rule is None:
            return _deny_missing(
                layer=layer,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                authority_scope=authority_scope,
                resource_scope=resource_scope,
                policy_rule_id=missing_rule_id,
            )
        if rule.status is not CollaborativePolicyRuleStatus.ACTIVE:
            return _deny_inactive(
                rule=rule,
                layer=layer,
                authority_scope=authority_scope,
                resource_scope=resource_scope,
                policy_rule_id=inactive_rule_id,
            )
        return _decision_from_rule(
            rule=rule,
            layer=layer,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
        )
