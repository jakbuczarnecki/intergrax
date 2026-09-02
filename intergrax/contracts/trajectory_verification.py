# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trajectory verification contracts (DS-VER-STAGE-TRAJ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput, EvalTrajectoryOutput

TrajectoryAgentId = NewType("TrajectoryAgentId", str)

T = TypeVar("T")


def validate_trajectory_agent_id(value: object) -> TrajectoryAgentId:
    if type(value) is not str:
        raise TypeError(
            f"TrajectoryAgentId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "TrajectoryAgentId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "TrajectoryAgentId must not contain leading or trailing whitespace",
        )
    return TrajectoryAgentId(value)


@runtime_checkable
class TrajectoryAgentIdProvider(Protocol[T]):
    """Resolve explicit agent identity for trajectory evaluation."""

    def resolve(self, candidate: CandidateDecision[T]) -> TrajectoryAgentId | None:
        """Return configured agent identity bound to one candidate."""
        ...


@runtime_checkable
class TrajectoryEvaluator(Protocol):
    """Neutral trajectory evaluation capability over Tier-0 eval contracts."""

    def is_available(self) -> bool:
        """Return whether trajectory evaluator infrastructure is available."""
        ...

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        """Evaluate one execution trajectory."""
        ...


@dataclass(frozen=True, slots=True)
class TrajectoryVerificationStageConfig:
    """Immutable trajectory stage threshold configuration."""

    min_score: float = 0.75

    def __post_init__(self) -> None:
        if type(self.min_score) is not float or isinstance(self.min_score, bool):
            raise TypeError("TrajectoryVerificationStageConfig.min_score must be float")
        if self.min_score < 0.0 or self.min_score > 1.0:
            raise ValueError(
                "TrajectoryVerificationStageConfig.min_score must be in [0.0, 1.0]",
            )


def trajectory_verification_stage_config(
    *,
    min_score: float = 0.75,
) -> TrajectoryVerificationStageConfig:
    """Build normalized immutable trajectory stage configuration."""
    return TrajectoryVerificationStageConfig(min_score=min_score)
