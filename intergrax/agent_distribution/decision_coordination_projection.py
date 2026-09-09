# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure projection from Decision coordination artifacts to NPSC coordination intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.agent_distribution.capability_matching import (
    CapabilityRequirementError,
    build_agent_capability_requirement,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContribution,
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntent,
    CoordinationIntentContractError,
    CoordinationIntentId,
    validate_coordination_intent,
    validate_coordination_intent_id,
)
from intergrax.agent_distribution.multi_agent_coordination import CoordinationPolicy
from intergrax.agent_distribution.task_capability_resolution import (
    AgentDistributionCapabilityNeed,
    TaskCapabilityResolutionContractError,
    resolved_agent_distribution_capability_need,
)
from intergrax.contracts.decision_coordination import (
    DECISION_COORDINATION_ARTIFACT_KIND,
    DecisionCoordinationContribution,
    DecisionCoordinationSemantic,
    DecisionCoordinationShape,
    decision_coordination_artifact_kind,
)
from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
)

PayloadT = TypeVar("PayloadT")

_COORDINATION_INTENT_ID_NAMESPACE = "coordination_intent"


class DecisionCoordinationProjectionError(ValueError):
    """Projection from Decision coordination artifact to CoordinationIntent failed."""


@dataclass(frozen=True, slots=True)
class DecisionCoordinationProjectionContext:
    """Explicit projection identity when the artifact carrier lacks Decision lineage."""

    source_decision_identity: DecisionIdentity | None = None
    coordination_intent_id: CoordinationIntentId | None = None

    def __post_init__(self) -> None:
        has_identity = self.source_decision_identity is not None
        has_intent_id = self.coordination_intent_id is not None
        if has_identity and has_intent_id:
            raise ValueError(
                "DecisionCoordinationProjectionContext must not set both "
                "source_decision_identity and coordination_intent_id",
            )
        if not has_identity and not has_intent_id:
            raise ValueError(
                "DecisionCoordinationProjectionContext requires either "
                "source_decision_identity or coordination_intent_id",
            )
        if self.source_decision_identity is not None:
            if type(self.source_decision_identity) is not DecisionIdentity:
                raise TypeError(
                    "source_decision_identity must be DecisionIdentity",
                )
        if self.coordination_intent_id is not None:
            validate_coordination_intent_id(self.coordination_intent_id)


def coordination_intent_id_from_decision_identity(
    identity: DecisionIdentity,
) -> CoordinationIntentId:
    """Deterministic namespaced projection of canonical Decision identity."""
    if type(identity) is not DecisionIdentity:
        raise TypeError("identity must be DecisionIdentity")
    return CoordinationIntentId(
        f"{_COORDINATION_INTENT_ID_NAMESPACE}:{identity.decision_id}:"
        f"v{identity.version.value}",
    )


def _resolve_projection_intent_id(
    context: DecisionCoordinationProjectionContext,
) -> CoordinationIntentId:
    if context.source_decision_identity is not None:
        return coordination_intent_id_from_decision_identity(
            context.source_decision_identity,
        )
    if context.coordination_intent_id is None:
        raise DecisionCoordinationProjectionError(
            "projection context must provide decision identity or intent id",
        )
    return context.coordination_intent_id


def _validate_coordination_artifact_kind(
    artifact: DecisionArtifact[DecisionCoordinationSemantic[PayloadT]],
) -> None:
    if type(artifact) is not DecisionArtifact:
        raise TypeError("artifact must be DecisionArtifact")
    expected_kind = decision_coordination_artifact_kind()
    if artifact.kind != expected_kind:
        raise DecisionCoordinationProjectionError(
            "artifact kind must be "
            f"{DECISION_COORDINATION_ARTIFACT_KIND!r}, got {artifact.kind!r}",
        )


def _project_execution_mode(
    shape: DecisionCoordinationShape,
) -> CoordinationExecutionMode:
    if shape is DecisionCoordinationShape.SINGLE:
        return CoordinationExecutionMode.SINGLE
    if shape is DecisionCoordinationShape.FAN_OUT:
        return CoordinationExecutionMode.FAN_OUT
    raise DecisionCoordinationProjectionError(
        f"unsupported DecisionCoordinationShape: {shape!r}",
    )


def _project_capability_need(
    contribution: DecisionCoordinationContribution[PayloadT],
) -> AgentDistributionCapabilityNeed:
    capability_id = str(contribution.capability_requirement.capability_id)
    try:
        requirement = build_agent_capability_requirement(required=(capability_id,))
    except CapabilityRequirementError as exc:
        raise DecisionCoordinationProjectionError(
            f"invalid capability projection for contribution "
            f"{contribution.contribution_id!r}",
        ) from exc
    try:
        return resolved_agent_distribution_capability_need(requirement)
    except TaskCapabilityResolutionContractError as exc:
        raise DecisionCoordinationProjectionError(
            f"invalid resolved capability need for contribution "
            f"{contribution.contribution_id!r}",
        ) from exc


def _project_contribution(
    contribution: DecisionCoordinationContribution[PayloadT],
) -> CoordinationContribution[PayloadT]:
    if type(contribution) is not DecisionCoordinationContribution:
        raise TypeError("contribution must be DecisionCoordinationContribution")
    return CoordinationContribution(
        contribution_id=CoordinationContributionId(str(contribution.contribution_id)),
        payload=contribution.payload,
        capability_need=_project_capability_need(contribution),
        policy=CoordinationPolicy(),
    )


def project_decision_coordination_artifact(
    artifact: DecisionArtifact[DecisionCoordinationSemantic[PayloadT]],
    context: DecisionCoordinationProjectionContext,
) -> CoordinationIntent[PayloadT]:
    """Project one typed Decision coordination artifact into a CoordinationIntent."""
    _validate_coordination_artifact_kind(artifact)
    semantic = artifact.content
    if type(semantic) is not DecisionCoordinationSemantic:
        raise TypeError("artifact content must be DecisionCoordinationSemantic")

    intent_id = _resolve_projection_intent_id(context)
    mode = _project_execution_mode(semantic.shape)
    contributions = tuple(_project_contribution(item) for item in semantic.contributions)

    intent = CoordinationIntent(
        intent_id=intent_id,
        mode=mode,
        contributions=contributions,
        requested_max_concurrency=None,
    )
    object_contributions: tuple[CoordinationContribution[object], ...] = tuple(
        CoordinationContribution(
            contribution_id=contribution.contribution_id,
            payload=contribution.payload,
            capability_need=contribution.capability_need,
            policy=contribution.policy,
        )
        for contribution in intent.contributions
    )
    object_intent = CoordinationIntent(
        intent_id=intent.intent_id,
        mode=intent.mode,
        contributions=object_contributions,
        requested_max_concurrency=intent.requested_max_concurrency,
    )
    try:
        validate_coordination_intent(object_intent)
    except CoordinationIntentContractError as exc:
        raise DecisionCoordinationProjectionError(
            "projected coordination intent failed validation",
        ) from exc
    return intent


def project_authoritative_accepted_decision_coordination(
    accepted: AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[PayloadT]],
) -> CoordinationIntent[PayloadT]:
    """Project one authoritative accepted Decision coordination outcome."""
    if type(accepted) is not AuthoritativeAcceptedDecision:
        raise TypeError("accepted must be AuthoritativeAcceptedDecision")
    context = DecisionCoordinationProjectionContext(
        source_decision_identity=accepted.identity,
    )
    return project_decision_coordination_artifact(accepted.artifact, context)


__all__ = [
    "DecisionCoordinationProjectionContext",
    "DecisionCoordinationProjectionError",
    "coordination_intent_id_from_decision_identity",
    "project_authoritative_accepted_decision_coordination",
    "project_decision_coordination_artifact",
]
