# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map failure classes to resilience policy actions (REL-ADV.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.resilience_policy import (
    FailureClass,
    FailureResponse,
    RebootStrategy,
    ResiliencePolicy,
    default_resilience_policy,
)
@dataclass(frozen=True, slots=True)
class ResilienceResolution:
    response: FailureResponse
    reboot_strategy: RebootStrategy
    policy_id: str
    reason: str


def resolve_resilience_policy(profile: ResiliencePolicy | None) -> ResiliencePolicy:
    return profile if profile is not None else default_resilience_policy()


def resolve_failure_action(
    failure_class: FailureClass,
    *,
    policy: ResiliencePolicy | None = None,
    attempt: int = 0,
) -> ResilienceResolution:
    resolved = resolve_resilience_policy(policy)
    response = resolved.action_for(failure_class)
    if attempt >= resolved.max_attempts and response in {
        FailureResponse.RETRY,
        FailureResponse.RETRY_ALTERNATE,
        FailureResponse.RETRY_RUN,
        FailureResponse.RECOVERY_REBOOT,
    }:
        response = FailureResponse.ESCALATE
    return ResilienceResolution(
        response=response,
        reboot_strategy=resolved.reboot_strategy,
        policy_id=resolved.policy_id,
        reason=f"{failure_class.value}:{response.value}",
    )


