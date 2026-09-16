# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default and configured Decision requirement policy implementations (GR-6-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_requirement_policy import (
    DecisionRequirement,
    DecisionRequirementContext,
    DecisionRequirementPolicy,
)


@dataclass(frozen=True, slots=True)
class DecisionRequirementRule:
    """Declarative match rule — all set fields must match; unset fields are wildcards."""

    tenant_id: str | None = None
    operation_id: str | None = None
    action: str | None = None
    side_effect_scope_id: str | None = None

    def matches(self, context: DecisionRequirementContext) -> bool:
        if self.tenant_id is not None and context.tenant_id != self.tenant_id:
            return False
        if self.operation_id is not None and context.operation_id != self.operation_id:
            return False
        if self.action is not None and context.action != self.action:
            return False
        if (
            self.side_effect_scope_id is not None
            and context.side_effect_scope_id != self.side_effect_scope_id
        ):
            return False
        return True


@dataclass(frozen=True, slots=True)
class PermissiveDecisionRequirementPolicy:
    """Platform default — Decision provenance not mandatory unless explicitly configured."""

    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        _ = context
        return DecisionRequirement.NOT_REQUIRED


@dataclass(frozen=True, slots=True)
class ConfiguredDecisionRequirementPolicy:
    """Explicit composition rules — first matching rule yields ``REQUIRED``."""

    rules: tuple[DecisionRequirementRule, ...] = ()

    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        for rule in self.rules:
            if rule.matches(context):
                return DecisionRequirement.REQUIRED
        return DecisionRequirement.NOT_REQUIRED


def normalize_decision_requirement(
    raw: DecisionRequirement,
) -> DecisionRequirement:
    if raw is DecisionRequirement.NOT_REQUIRED:
        return DecisionRequirement.NOT_REQUIRED
    if raw is DecisionRequirement.REQUIRED:
        return DecisionRequirement.REQUIRED
    if raw is DecisionRequirement.UNDETERMINED:
        return DecisionRequirement.UNDETERMINED
    return DecisionRequirement.UNDETERMINED


def classify_decision_requirement(
    policy: DecisionRequirementPolicy,
    context: DecisionRequirementContext,
) -> DecisionRequirement:
    """Evaluate policy with fail-closed normalization on unknown outputs."""
    result = policy.evaluate(context)
    return normalize_decision_requirement(result)


def decision_governed_side_effect_requirement_policy(
    *,
    required_actions: frozenset[str],
    required_operation_ids: frozenset[str] | None = None,
    required_side_effect_scope_ids: frozenset[str] | None = None,
    required_tenant_ids: frozenset[str] | None = None,
) -> DecisionRequirementPolicy:
    """Build explicit requirement policy for Decision-bound consequential effects."""
    rules: list[DecisionRequirementRule] = []
    operation_ids = required_operation_ids if required_operation_ids is not None else frozenset()
    scope_ids = (
        required_side_effect_scope_ids
        if required_side_effect_scope_ids is not None
        else frozenset()
    )
    tenant_ids = required_tenant_ids if required_tenant_ids is not None else frozenset()

    if operation_ids or scope_ids or tenant_ids:
        for action in sorted(required_actions):
            if operation_ids and scope_ids and tenant_ids:
                for tenant_id in sorted(tenant_ids):
                    for operation_id in sorted(operation_ids):
                        for scope_id in sorted(scope_ids):
                            rules.append(
                                DecisionRequirementRule(
                                    tenant_id=tenant_id,
                                    operation_id=operation_id,
                                    action=action,
                                    side_effect_scope_id=scope_id,
                                ),
                            )
            elif tenant_ids:
                for tenant_id in sorted(tenant_ids):
                    rules.append(
                        DecisionRequirementRule(tenant_id=tenant_id, action=action),
                    )
            elif operation_ids:
                for operation_id in sorted(operation_ids):
                    rules.append(
                        DecisionRequirementRule(operation_id=operation_id, action=action),
                    )
            elif scope_ids:
                for scope_id in sorted(scope_ids):
                    rules.append(
                        DecisionRequirementRule(
                            action=action,
                            side_effect_scope_id=scope_id,
                        ),
                    )
            else:
                rules.append(DecisionRequirementRule(action=action))
    else:
        for action in sorted(required_actions):
            rules.append(DecisionRequirementRule(action=action))
    return ConfiguredDecisionRequirementPolicy(rules=tuple(rules))


__all__ = [
    "ConfiguredDecisionRequirementPolicy",
    "DecisionRequirementRule",
    "PermissiveDecisionRequirementPolicy",
    "classify_decision_requirement",
    "decision_governed_side_effect_requirement_policy",
    "normalize_decision_requirement",
]
