# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Per-evaluation bridge from DecisionFlowResult to authoritative exposure contracts."""

from __future__ import annotations

from intergrax.contracts.decision_authoritative_exposure import (
    AuthoritativeDecisionExposure,
    DecisionEvaluationScope,
    ExposureAccepted,
    ExposureResolution,
)
from intergrax.runtime.decision_flow import (
    DecisionFlowHostAction,
    DecisionFlowResult,
    DecisionFlowScope,
)


class DecisionFlowExposureMappingError(ValueError):
    """Fail-closed mapping failure for illegal DecisionFlowResult authority state."""


def decision_flow_scope_to_evaluation_scope(
    flow_scope: DecisionFlowScope,
) -> DecisionEvaluationScope:
    """Map runtime host scope to contracts-layer evaluation scope."""
    if type(flow_scope) is not DecisionFlowScope:
        raise TypeError("flow_scope must be DecisionFlowScope")
    if flow_scope is DecisionFlowScope.GRAPH_FINAL:
        return DecisionEvaluationScope.GRAPH_FINAL
    if flow_scope is DecisionFlowScope.UAEP_STEP:
        return DecisionEvaluationScope.UAEP_STEP
    raise ValueError(f"unsupported DecisionFlowScope: {flow_scope!r}")


def decision_flow_result_to_authoritative_exposure(
    result: DecisionFlowResult[object],
) -> AuthoritativeDecisionExposure[object] | None:
    """Map one DecisionFlowResult to exposure, or None when non-terminal (HITL)."""
    if type(result) is not DecisionFlowResult:
        raise TypeError("result must be DecisionFlowResult")
    if result.host_action is DecisionFlowHostAction.PENDING_HUMAN:
        return None
    evaluation_scope = decision_flow_scope_to_evaluation_scope(result.flow_scope)
    accepted = result.accepted_decision
    resolution = result.resolution_record
    if accepted is not None and resolution is not None:
        raise DecisionFlowExposureMappingError(
            "DecisionFlowResult cannot expose both accepted_decision and "
            "resolution_record",
        )
    if accepted is not None:
        return ExposureAccepted(scope=evaluation_scope, accepted=accepted)
    if resolution is not None:
        return ExposureResolution(scope=evaluation_scope, resolution=resolution)
    raise DecisionFlowExposureMappingError(
        "DecisionFlowResult has no terminal authoritative outcome to expose",
    )
