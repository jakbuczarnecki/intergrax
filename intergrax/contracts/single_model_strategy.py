# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single Model deliberation strategy foundation (DS-DELIB-02).

Semantic configuration for one-source candidate production. The strategy
describes provider-neutral inference selection and deliberation inputs; canonical
Execution hosts inference work and returns typed output for CandidateDecision
assembly. Single Model does not finalize, authorize, or perform technical retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionArtifactKind,
    DecisionVersionLineage,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_strategy import (
    DecisionStrategyKind,
    DecisionStrategyRegistration,
    DecisionStrategyRegistry,
    register_decision_strategy,
    validate_decision_strategy_kind,
)
from intergrax.llm.messages import ChatMessage

_SINGLE_MODEL_KIND = validate_decision_strategy_kind("single_model")

T = TypeVar("T")


def single_model_strategy_kind() -> DecisionStrategyKind:
    """Canonical deliberation strategy identity for Single Model."""
    return _SINGLE_MODEL_KIND


@dataclass(frozen=True, slots=True)
class SingleModelInferenceConfiguration:
    """Provider-neutral inference profile reference resolved by Execution host."""

    llm_profile_id: str

    def __post_init__(self) -> None:
        if type(self.llm_profile_id) is not str:
            raise TypeError(
                "SingleModelInferenceConfiguration.llm_profile_id must be str",
            )
        if not self.llm_profile_id or not self.llm_profile_id.strip():
            raise ValueError(
                "SingleModelInferenceConfiguration.llm_profile_id must be "
                "non-empty and not whitespace-only",
            )
        if self.llm_profile_id != self.llm_profile_id.strip():
            raise ValueError(
                "SingleModelInferenceConfiguration.llm_profile_id must not "
                "contain leading or trailing whitespace",
            )


@dataclass(frozen=True, slots=True)
class SingleModelDeliberationInput(Generic[T]):
    """Typed deliberation input describing one inference work unit."""

    messages: tuple[ChatMessage, ...]
    output_type: type[T]
    artifact_kind: DecisionArtifactKind

    def __post_init__(self) -> None:
        if type(self.messages) is not tuple:
            raise TypeError("SingleModelDeliberationInput.messages must be tuple")
        if type(self.output_type) is not type:
            raise TypeError(
                "SingleModelDeliberationInput.output_type must be type",
            )
        validate_decision_artifact_kind(self.artifact_kind)


@dataclass(frozen=True, slots=True)
class SingleModelStrategy:
    """Single-source deliberation strategy — one inference producer per candidate."""

    inference: SingleModelInferenceConfiguration
    kind: DecisionStrategyKind = field(default=_SINGLE_MODEL_KIND)

    def __post_init__(self) -> None:
        validated_kind = validate_decision_strategy_kind(self.kind)
        if validated_kind != _SINGLE_MODEL_KIND:
            raise ValueError(
                "SingleModelStrategy.kind must be single_model "
                f"got {validated_kind!r}",
            )


def single_model_strategy_registration(
    inference: SingleModelInferenceConfiguration,
) -> DecisionStrategyRegistration:
    """Build one explicit Single Model registration for host/bootstrap wiring."""
    strategy = SingleModelStrategy(inference=inference)
    return DecisionStrategyRegistration(kind=strategy.kind, strategy=strategy)


def register_single_model_strategy(
    registry: DecisionStrategyRegistry,
    inference: SingleModelInferenceConfiguration,
) -> DecisionStrategyRegistry:
    """Register Single Model on ``registry``; return a new immutable registry."""
    return register_decision_strategy(
        registry,
        single_model_strategy_registration(inference),
    )


def single_model_candidate_decision(
    *,
    identity: DecisionIdentity,
    artifact_kind: DecisionArtifactKind,
    payload: T,
    lineage: DecisionVersionLineage | None = None,
) -> CandidateDecision[T]:
    """Assemble a typed candidate from Single Model inference output."""
    resolved_lineage = lineage or decision_version_lineage(
        current=decision_lineage_ref(identity.version),
    )
    artifact = DecisionArtifact(kind=artifact_kind, content=payload)
    return CandidateDecision(
        identity=identity,
        artifact=artifact,
        lineage=resolved_lineage,
    )
