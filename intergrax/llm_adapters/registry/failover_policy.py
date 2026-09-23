# © Artur Czarnecki. All rights reserved.

"""Platform default failover policy implementations."""

from __future__ import annotations

from intergrax.llm_adapters._shared.retry import is_retriable_provider_error
from intergrax.llm_adapters.contracts.failover_policy import (
    FailoverCandidateEligibility,
    FailoverCandidateEligibilityContext,
    FailoverCandidateNotAuthorizedError,
    FailoverDecision,
    FailoverEligibilityPolicy,
    FailoverPolicy,
    FailoverProgressionContext,
    FailoverRoutingAuthorisationContext,
)
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.routing_profile import LLMRoutingProfile
from intergrax.llm_adapters.routing.evaluator import effective_allowlist, is_profile_allowed, profile_identity


class PlatformDefaultFailoverPolicy:
    """Retryable provider errors with remaining candidates → TRY_NEXT; otherwise STOP."""

    def decide_after_failure(self, context: FailoverProgressionContext) -> FailoverDecision:
        if context.is_last_candidate:
            return FailoverDecision.STOP
        if is_retriable_provider_error(context.error, context.failover_retry_config):
            return FailoverDecision.TRY_NEXT
        return FailoverDecision.STOP


class PermissiveFailoverEligibilityPolicy:
    """When authorisation is absent, allow; otherwise delegate to routing allowlist."""

    def evaluate(self, context: FailoverCandidateEligibilityContext) -> FailoverCandidateEligibility:
        if context.routing_authorisation is None:
            return FailoverCandidateEligibility.ALLOWED
        return RoutingAllowlistFailoverEligibilityPolicy().evaluate(context)


class RoutingAllowlistFailoverEligibilityPolicy:
    """Fail-closed: candidates must belong to the routing allowlist when authorisation is set."""

    def evaluate(self, context: FailoverCandidateEligibilityContext) -> FailoverCandidateEligibility:
        authorisation = context.routing_authorisation
        if authorisation is None:
            return FailoverCandidateEligibility.ALLOWED
        if is_profile_allowed(context.candidate, authorisation.allowed_profiles):
            return FailoverCandidateEligibility.ALLOWED
        return FailoverCandidateEligibility.DENIED


def default_failover_policy() -> FailoverPolicy:
    return PlatformDefaultFailoverPolicy()


def default_failover_eligibility_policy() -> FailoverEligibilityPolicy:
    return PermissiveFailoverEligibilityPolicy()


def routing_authorisation_context(
    routing_profile: LLMRoutingProfile | None,
) -> FailoverRoutingAuthorisationContext | None:
    """Build authorisation context when ``LLMRoutingProfile`` governs the session."""
    if routing_profile is None:
        return None
    return FailoverRoutingAuthorisationContext(
        allowed_profiles=effective_allowlist(routing_profile),
    )


def assert_failover_chain_authorized(
    ordered_profiles: tuple[LLMProfile, ...],
    *,
    routing_authorisation: FailoverRoutingAuthorisationContext | None,
    eligibility_policy: FailoverEligibilityPolicy | None = None,
) -> None:
    """Validate every candidate before adapter materialization; fail closed on denial."""
    if not ordered_profiles:
        raise ValueError("failover chain must contain at least one profile")
    policy = eligibility_policy or default_failover_eligibility_policy()
    primary = ordered_profiles[0]
    for index, candidate in enumerate(ordered_profiles):
        verdict = policy.evaluate(
            FailoverCandidateEligibilityContext(
                candidate=candidate,
                candidate_index=index,
                primary_selected=primary,
                routing_authorisation=routing_authorisation,
            )
        )
        if verdict is FailoverCandidateEligibility.DENIED:
            raise FailoverCandidateNotAuthorizedError(
                f"failover candidate {profile_identity(candidate)!r} is not authorized "
                "by canonical routing policy"
            )


__all__ = [
    "PlatformDefaultFailoverPolicy",
    "PermissiveFailoverEligibilityPolicy",
    "RoutingAllowlistFailoverEligibilityPolicy",
    "assert_failover_chain_authorized",
    "routing_authorisation_context",
    "default_failover_eligibility_policy",
    "default_failover_policy",
]
