# © Artur Czarnecki. All rights reserved.

"""Deterministic, provider-neutral meaningful side-effect policy fake (GEC-5)."""

from __future__ import annotations

from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision


class DeterministicMeaningfulSideEffectPolicy:
    """Recording fake — no business rules, no provider awareness."""

    def __init__(
        self,
        *,
        default: PolicyAction = PolicyAction.ALLOW,
        by_action: dict[str, PolicyAction] | None = None,
        raise_on_evaluate: Exception | None = None,
    ) -> None:
        self.default = default
        self.by_action = dict(by_action or {})
        self.raise_on_evaluate = raise_on_evaluate
        self.calls: list[MeaningfulSideEffectRequest] = []
        self.call_log: list[str] = []

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        self.call_log.append("policy.evaluate")
        self.calls.append(request)
        if self.raise_on_evaluate is not None:
            raise self.raise_on_evaluate
        action = self.by_action.get(request.action, self.default)
        return PolicyDecision(
            action=action,
            reason=f"deterministic:{action.value}",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id=f"fake.meaningful_side_effect.{request.action}",
            audit_payload={"action": request.action},
        )
