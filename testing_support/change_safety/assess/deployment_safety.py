# © Artur Czarnecki. All rights reserved.

from testing_support.change_safety.contracts.enums import DeployRefusalReason, DeploySafetyVerdict


def assess_rollback_eligibility(
    *, durable_state_forward_only: bool
) -> tuple[DeploySafetyVerdict, DeployRefusalReason | None]:
    if durable_state_forward_only:
        return DeploySafetyVerdict.REFUSE, DeployRefusalReason.DURABLE_STATE_FORWARD_ONLY
    return DeploySafetyVerdict.ELIGIBLE, None
