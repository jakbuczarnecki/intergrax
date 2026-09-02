# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semantic verification contracts (DS-VER-STAGE-SEM)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_record import CandidateDecision
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileId,
    validate_inference_profile_id,
)
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput

SemanticRubricId = NewType("SemanticRubricId", str)
SemanticRubricProvenanceRef = NewType("SemanticRubricProvenanceRef", str)

T = TypeVar("T")


class VerifierIndependenceMode(str, Enum):
    """Whether semantic verification uses an independent verifier profile."""

    INDEPENDENT = "independent"
    SHARED_PROFILE = "shared_profile"


class SemanticRubricNotFoundError(LookupError):
    """Raised when a configured rubric reference cannot be resolved."""


def validate_semantic_rubric_id(value: object) -> SemanticRubricId:
    if type(value) is not str:
        raise TypeError(
            f"SemanticRubricId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "SemanticRubricId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "SemanticRubricId must not contain leading or trailing whitespace",
        )
    return SemanticRubricId(value)


def validate_semantic_rubric_version(value: object) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(
            f"SemanticRubric version must be int, got {type(value).__name__}",
        )
    if value < 1:
        raise ValueError("SemanticRubric version must be a positive int >= 1")
    return value


def validate_semantic_rubric_provenance_ref(
    value: object,
) -> SemanticRubricProvenanceRef:
    if type(value) is not str:
        raise TypeError(
            "SemanticRubricProvenanceRef must be str, "
            f"got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "SemanticRubricProvenanceRef must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "SemanticRubricProvenanceRef must not contain leading or trailing whitespace",
        )
    return SemanticRubricProvenanceRef(value)


@dataclass(frozen=True, slots=True)
class SemanticRubricRef:
    """Versioned rubric reference supplied by stage configuration."""

    rubric_id: SemanticRubricId
    version: int

    def __post_init__(self) -> None:
        validate_semantic_rubric_id(self.rubric_id)
        validate_semantic_rubric_version(self.version)


def semantic_rubric_ref(
    *,
    rubric_id: str | SemanticRubricId,
    version: int,
) -> SemanticRubricRef:
    """Build one versioned semantic rubric reference."""
    resolved_id = (
        rubric_id
        if type(rubric_id) is SemanticRubricId
        else validate_semantic_rubric_id(rubric_id)
    )
    return SemanticRubricRef(
        rubric_id=resolved_id,
        version=validate_semantic_rubric_version(version),
    )


@dataclass(frozen=True, slots=True)
class ResolvedSemanticRubric:
    """Immutable resolved rubric artifact with provenance."""

    ref: SemanticRubricRef
    criteria: tuple[str, ...]
    min_score: float
    provenance_ref: SemanticRubricProvenanceRef
    reference_context: str | None = None

    def __post_init__(self) -> None:
        if type(self.ref) is not SemanticRubricRef:
            raise TypeError("ResolvedSemanticRubric.ref must be SemanticRubricRef")
        if type(self.criteria) is not tuple:
            raise TypeError("ResolvedSemanticRubric.criteria must be tuple")
        for item in self.criteria:
            if type(item) is not str or not item.strip():
                raise ValueError(
                    "ResolvedSemanticRubric.criteria items must be non-empty str",
                )
        if type(self.min_score) is not float or isinstance(self.min_score, bool):
            raise TypeError("ResolvedSemanticRubric.min_score must be float")
        if self.min_score < 0.0 or self.min_score > 1.0:
            raise ValueError("ResolvedSemanticRubric.min_score must be in [0.0, 1.0]")
        validate_semantic_rubric_provenance_ref(self.provenance_ref)
        if self.reference_context is not None and type(self.reference_context) is not str:
            raise TypeError(
                "ResolvedSemanticRubric.reference_context must be str or None",
            )


def resolved_semantic_rubric(
    *,
    ref: SemanticRubricRef,
    criteria: tuple[str, ...],
    min_score: float,
    provenance_ref: str | SemanticRubricProvenanceRef,
    reference_context: str | None = None,
) -> ResolvedSemanticRubric:
    """Build one resolved semantic rubric artifact."""
    resolved_provenance = (
        provenance_ref
        if type(provenance_ref) is SemanticRubricProvenanceRef
        else validate_semantic_rubric_provenance_ref(provenance_ref)
    )
    return ResolvedSemanticRubric(
        ref=ref,
        criteria=criteria,
        min_score=min_score,
        provenance_ref=resolved_provenance,
        reference_context=reference_context,
    )


@runtime_checkable
class SemanticRubricResolver(Protocol):
    """Resolve configured rubric references to immutable rubric artifacts."""

    def is_available(self) -> bool:
        """Return whether resolver infrastructure is available."""
        ...

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        """Resolve one rubric reference or raise SemanticRubricNotFoundError."""
        ...


@runtime_checkable
class SemanticContentProvider(Protocol[T]):
    """Extract candidate semantic output text for judge evaluation."""

    def extract(self, candidate: CandidateDecision[T]) -> str:
        """Return semantic output text from one candidate artifact."""
        ...


@runtime_checkable
class SemanticJudge(Protocol):
    """Neutral semantic judge capability over Tier-0 eval contracts."""

    def is_available(self) -> bool:
        """Return whether judge infrastructure is available."""
        ...

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        """Evaluate candidate output against structured judge input."""
        ...


@dataclass(frozen=True, slots=True)
class SemanticVerificationIndependenceConfig:
    """Producer/verifier profile separation declaration for semantic verification."""

    mode: VerifierIndependenceMode
    producer_profile_id: InferenceProfileId
    verifier_profile_id: InferenceProfileId

    def __post_init__(self) -> None:
        if type(self.mode) is not VerifierIndependenceMode:
            raise TypeError(
                "SemanticVerificationIndependenceConfig.mode must be "
                "VerifierIndependenceMode",
            )
        validate_inference_profile_id(self.producer_profile_id)
        validate_inference_profile_id(self.verifier_profile_id)


def semantic_verification_independence_config(
    *,
    mode: VerifierIndependenceMode,
    producer_profile_id: str | InferenceProfileId,
    verifier_profile_id: str | InferenceProfileId,
) -> SemanticVerificationIndependenceConfig:
    """Build one semantic verification independence configuration."""
    resolved_producer = (
        producer_profile_id
        if type(producer_profile_id) is InferenceProfileId
        else validate_inference_profile_id(producer_profile_id)
    )
    resolved_verifier = (
        verifier_profile_id
        if type(verifier_profile_id) is InferenceProfileId
        else validate_inference_profile_id(verifier_profile_id)
    )
    return SemanticVerificationIndependenceConfig(
        mode=mode,
        producer_profile_id=resolved_producer,
        verifier_profile_id=resolved_verifier,
    )
