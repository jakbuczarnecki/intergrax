# © Artur Czarnecki. All rights reserved.

"""Direct interpreter of ``ImmutableRuntimePolicyBundle`` rules (PC-1).

Model: **interprets rules from the immutable pack directly** — not a binding of
an independent live PolicyEngine result to a later snapshot. The evaluator
receives a concrete ``ImmutableRuntimePolicyBundle``, matches a rule, emits
``PolicyDecision`` with pack identity/digest already set, and wraps the result
as ``EvaluatedPolicyDecision``. Pack identity cannot be attached after
evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from intergrax.contracts.evaluated_policy_decision import (
    EvaluatedPolicyDecision,
    request_digest_for_payload,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import (
    ImmutableRuntimePolicyBundle,
    PolicyBundleRule,
)


class RuntimePolicyBundleEvaluator:
    """Provider-neutral bundle-backed meaningful side-effect evaluator."""

    def __init__(
        self,
        bundle: ImmutableRuntimePolicyBundle,
        *,
        clock: Callable[[], datetime] | None = None,
        decision_id_prefix: str = "eval",
    ) -> None:
        if not bundle.canonical_digest:
            bundle = bundle.with_canonical_digest()
        else:
            recomputed = bundle.compute_digest()
            if bundle.canonical_digest != recomputed:
                raise ValueError("bundle_canonical_digest_mismatch")
        self._bundle = bundle
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._decision_id_prefix = decision_id_prefix
        self.last_evaluation: EvaluatedPolicyDecision | None = None
        self.calls: list[MeaningfulSideEffectRequest] = []

    @property
    def bundle(self) -> ImmutableRuntimePolicyBundle:
        return self._bundle

    def evaluate(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> EvaluatedPolicyDecision:
        """Evaluate ``request`` against the bound immutable pack."""
        self.calls.append(request)
        evaluated_at = self._clock()
        req_digest = request_digest_for_payload(request.model_dump(mode="json"))
        rule = self._match_rule(request.action)
        if rule is None:
            decision = PolicyDecision(
                action=PolicyAction.DENY,
                reason="no_matching_rule",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="bundle.no_match",
                policy_bundle_id=self._bundle.bundle_id,
                policy_bundle_version=self._bundle.version,
                policy_bundle_digest=self._bundle.canonical_digest,
                decision_id=f"{self._decision_id_prefix}:deny:no_match",
                audit_payload={
                    "request_digest": req_digest,
                    "evaluated_at": evaluated_at.isoformat(),
                },
            )
            # Fail closed with an explicit deny rule id that is not in the pack
            # — callers must not treat this as an attested allow path.
            # For EvaluatedPolicyDecision construction we use a synthetic deny
            # that still binds pack identity (matched_rule_id = bundle.no_match).
            evaluated = EvaluatedPolicyDecision(
                decision=decision,
                bundle_id=self._bundle.bundle_id,
                bundle_version=self._bundle.version,
                bundle_digest=self._bundle.canonical_digest,
                matched_rule_id="bundle.no_match",
                evaluated_at=evaluated_at,
                request_digest=req_digest,
            )
            self.last_evaluation = evaluated
            return evaluated

        action = self._effect_to_action(rule.effect)
        decision = PolicyDecision(
            action=action,
            reason=f"bundle_rule:{rule.rule_id}",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id=rule.rule_id,
            policy_bundle_id=self._bundle.bundle_id,
            policy_bundle_version=self._bundle.version,
            policy_bundle_digest=self._bundle.canonical_digest,
            decision_id=f"{self._decision_id_prefix}:{rule.rule_id}:{uuid4().hex[:8]}",
            audit_payload={
                "request_digest": req_digest,
                "evaluated_at": evaluated_at.isoformat(),
                "match_action": rule.match_action or request.action,
            },
        )
        evaluated = EvaluatedPolicyDecision(
            decision=decision,
            bundle_id=self._bundle.bundle_id,
            bundle_version=self._bundle.version,
            bundle_digest=self._bundle.canonical_digest,
            matched_rule_id=rule.rule_id,
            evaluated_at=evaluated_at,
            request_digest=req_digest,
        )
        evaluated.assert_consistent_with_bundle(self._bundle)
        self.last_evaluation = evaluated
        return evaluated

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        """``MeaningfulSideEffectEvaluator`` Protocol surface."""
        return self.evaluate(request).decision

    def _match_rule(self, action: str) -> PolicyBundleRule | None:
        normalized = action.strip()
        for rule in self._bundle.rules:
            if rule.match_action and rule.match_action == normalized:
                return rule
        # Backward-compatible: rule_id suffix ``.{ACTION}``.
        suffix = f".{normalized}"
        for rule in self._bundle.rules:
            if rule.rule_id.endswith(suffix):
                return rule
        return None

    @staticmethod
    def _effect_to_action(effect: str) -> PolicyAction:
        token = (effect or "deny").strip().lower()
        if not token:
            return PolicyAction.DENY
        try:
            return PolicyAction(token)
        except ValueError:
            return PolicyAction.DENY
