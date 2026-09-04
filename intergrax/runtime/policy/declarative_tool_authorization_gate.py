# © Artur Czarnecki. All rights reserved.

"""Phase-1 declarative tool authorization gate for meaningful side effects."""

from __future__ import annotations

from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
    SideEffectAuthorizationFailureReason,
)
from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.tools.core.contracts import ToolContract


def require_meaningful_side_effect_authorization(
    *,
    contract: ToolContract,
    enforcer: DeclarativePolicyEnforcer | None,
    run_id: str,
    agent_id: str,
) -> None:
    """
    Fail closed when a side-effecting tool lacks recognized enforcing authorization.

    Phase 1 recognizes only ``DeclarativePolicyEnforcer`` in ``ENFORCE`` mode.
    """
    if not contract.side_effects:
        return

    if enforcer is None:
        raise MeaningfulSideEffectAuthorizationRequiredError(
            run_id=run_id,
            agent_id=agent_id,
            tool_id=contract.tool_id,
            reason=SideEffectAuthorizationFailureReason.NOT_CONFIGURED,
        )

    if enforcer.runtime.enforcement_mode is not PolicyEnforcementMode.ENFORCE:
        raise MeaningfulSideEffectAuthorizationRequiredError(
            run_id=run_id,
            agent_id=agent_id,
            tool_id=contract.tool_id,
            reason=SideEffectAuthorizationFailureReason.NON_ENFORCING_MODE,
        )
