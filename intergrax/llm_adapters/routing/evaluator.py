# © Artur Czarnecki. All rights reserved.

"""LLM routing rule evaluator (M-LLM-X.9.2b)."""

from __future__ import annotations

from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing.contracts import (
    LLMRoutingProfile,
    LLMRoutingRule,
    RoutingContext,
    RoutingEvaluation,
    RoutingTarget,
)


class AllowlistViolationError(ValueError):
    """Raised when a rule resolves to a profile outside ``allowed_profiles``."""


def profile_identity(profile: LLMProfile) -> str:
    model = profile.model or "default"
    return f"{profile.provider.value}:{model}"


def effective_allowlist(profile: LLMRoutingProfile) -> tuple[LLMProfile, ...]:
    if profile.allowed_profiles:
        return profile.allowed_profiles
    return (profile.default_profile,)


def is_profile_allowed(candidate: LLMProfile, allowlist: tuple[LLMProfile, ...]) -> bool:
    allowed_ids = {profile_identity(item) for item in allowlist}
    return profile_identity(candidate) in allowed_ids


class LLMRoutingEvaluator:
    """First-match rule evaluation with mandatory allowlist guard."""

    def evaluate(
        self,
        routing_profile: LLMRoutingProfile,
        context: RoutingContext,
    ) -> RoutingEvaluation:
        allowlist = effective_allowlist(routing_profile)
        ordered_rules = sorted(
            routing_profile.rules,
            key=lambda rule: rule.priority,
            reverse=True,
        )
        for rule in ordered_rules:
            if not rule.matches(context):
                continue
            target = rule.resolve(context)
            return self._build_evaluation(
                routing_profile=routing_profile,
                allowlist=allowlist,
                matched_rule_id=rule.rule_id,
                target=target,
            )
        return self._build_evaluation(
            routing_profile=routing_profile,
            allowlist=allowlist,
            matched_rule_id=None,
            target=RoutingTarget(profile=routing_profile.default_profile, reason="default_profile"),
        )

    def _build_evaluation(
        self,
        *,
        routing_profile: LLMRoutingProfile,
        allowlist: tuple[LLMProfile, ...],
        matched_rule_id: str | None,
        target: RoutingTarget,
    ) -> RoutingEvaluation:
        selected = target.profile or routing_profile.default_profile
        if target.profile is not None and not is_profile_allowed(selected, allowlist):
            raise AllowlistViolationError(
                f"rule target {profile_identity(selected)} not in allowed_profiles",
            )
        hint = target.hint.value if target.hint is not None else None
        if matched_rule_id is None:
            reason = "default_profile"
        elif target.reason:
            reason = f"rule:{matched_rule_id}:{target.reason}"
        else:
            reason = f"rule:{matched_rule_id}"
        return RoutingEvaluation(
            matched_rule_id=matched_rule_id,
            target=target,
            routing_reason=reason,
            selected_profile=selected,
            policy_route_hint=hint,
        )

    @staticmethod
    def validate_rules(rules: tuple[LLMRoutingRule, ...]) -> list[str]:
        """Return validation errors for duplicate rule ids."""
        errors: list[str] = []
        seen: set[str] = set()
        for rule in rules:
            if rule.rule_id in seen:
                errors.append(f"duplicate rule_id: {rule.rule_id}")
            seen.add(rule.rule_id)
        return errors
