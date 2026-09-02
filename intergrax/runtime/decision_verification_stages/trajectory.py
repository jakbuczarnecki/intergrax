# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trajectory probabilistic Decision Verification stage (DS-VER-STAGE-TRAJ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_record import (
    CandidateDecision,
    candidate_decision_ref,
)
from intergrax.contracts.decision_verification import (
    VerificationStageKind,
    VerificationStageOutcome,
    VerificationStageRecord,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageUnavailableError,
)
from intergrax.contracts.trajectory_verification import (
    TrajectoryAgentIdProvider,
    TrajectoryEvaluator,
    TrajectoryVerificationStageConfig,
    validate_trajectory_agent_id,
)
from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput

T = TypeVar("T")

TRAJECTORY_VERIFICATION_STAGE_KIND = validate_verification_stage_kind("trajectory")

_AGENT_ID_MISSING_REQUIREMENT = validate_verification_requirement_code(
    "verification.trajectory.agent_id_missing",
)
_AGENT_ID_MISSING_FINDING = validate_verification_finding_code(
    "verification.trajectory.agent_id_missing",
)
_BELOW_REQUIREMENT = validate_verification_requirement_code(
    "verification.trajectory.below_requirement",
)
_BELOW_FINDING = validate_verification_finding_code(
    "verification.trajectory.below_requirement",
)


def _bounded_summary(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "trajectory evaluation did not meet requirement"
    first = reasons[0].strip()
    if len(first) > 240:
        return first[:237] + "..."
    return first


@dataclass(frozen=True, slots=True)
class TrajectoryVerificationStage(Generic[T]):
    """Probabilistic trajectory verification over canonical execution lineage."""

    evaluator: TrajectoryEvaluator
    agent_id_provider: TrajectoryAgentIdProvider[T]
    config: TrajectoryVerificationStageConfig = TrajectoryVerificationStageConfig()

    def __post_init__(self) -> None:
        if type(self.config) is not TrajectoryVerificationStageConfig:
            raise TypeError(
                "TrajectoryVerificationStage.config must be TrajectoryVerificationStageConfig",
            )

    @property
    def kind(self) -> VerificationStageKind:
        return TRAJECTORY_VERIFICATION_STAGE_KIND

    @property
    def execution_class(self) -> VerificationStageExecutionClass:
        return VerificationStageExecutionClass.PROBABILISTIC

    async def verify(
        self,
        candidate: CandidateDecision[T],
    ) -> VerificationStageRecord:
        if not self.evaluator.is_available():
            raise VerificationStageUnavailableError(
                "trajectory evaluator infrastructure is unavailable",
            )
        proposal_ref = candidate_decision_ref(candidate)
        resolved_agent_id = self.agent_id_provider.resolve(candidate)
        if resolved_agent_id is None:
            finding = verification_finding(
                code=_AGENT_ID_MISSING_FINDING,
                message="required trajectory agent identity is missing",
            )
            challenge = verification_challenge(
                proposal_ref=proposal_ref,
                stage=TRAJECTORY_VERIFICATION_STAGE_KIND,
                requirement_code=_AGENT_ID_MISSING_REQUIREMENT,
                finding=finding,
            )
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=TRAJECTORY_VERIFICATION_STAGE_KIND,
                outcome=VerificationStageOutcome.CHALLENGED,
                challenge=challenge,
            )
        agent_id = validate_trajectory_agent_id(resolved_agent_id)
        trajectory_input = EvalTrajectoryInput(
            run_id=str(candidate.identity.execution.run_id),
            tenant_id=candidate.identity.tenant_id,
            min_score=self.config.min_score,
            agent_id=agent_id,
            record_observation=False,
        )
        trajectory_output = self.evaluator.evaluate(trajectory_input)
        if trajectory_output.passed:
            return verification_stage_record(
                proposal_ref=proposal_ref,
                stage=TRAJECTORY_VERIFICATION_STAGE_KIND,
                outcome=VerificationStageOutcome.PASSED,
            )
        message = _bounded_summary(tuple(trajectory_output.reasons))
        finding = verification_finding(
            code=_BELOW_FINDING,
            message=message,
        )
        challenge = verification_challenge(
            proposal_ref=proposal_ref,
            stage=TRAJECTORY_VERIFICATION_STAGE_KIND,
            requirement_code=_BELOW_REQUIREMENT,
            finding=finding,
        )
        return verification_stage_record(
            proposal_ref=proposal_ref,
            stage=TRAJECTORY_VERIFICATION_STAGE_KIND,
            outcome=VerificationStageOutcome.CHALLENGED,
            challenge=challenge,
        )
