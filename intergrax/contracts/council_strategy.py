# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Council deliberation strategy foundation (DS-COUNCIL).

Multi-participant deliberation configuration producing candidate decisions
through parallel independent proposals, structured disagreement, and bounded
synthesis. Council does not finalize, authorize, verify, or own governance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.decision_context_visibility import (
    DeliberationContextId,
    ParticipantContextVisibilityConfiguration,
    validate_deliberation_context_id,
)
from intergrax.contracts.decision_disagreement import DecisionDisagreementArtifact
from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_participants import (
    ParticipantConfiguration,
    ParticipantId,
    ParticipantRoleId,
    validate_participant_id,
    validate_participant_role_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifactKind,
    DecisionProposalRef,
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
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileId,
    validate_inference_profile_id,
)

_COUNCIL_KIND = validate_decision_strategy_kind("council")

T = TypeVar("T")


def council_strategy_kind() -> DecisionStrategyKind:
    """Canonical deliberation strategy identity for Council."""
    return _COUNCIL_KIND


class CouncilResolutionDisposition(str, Enum):
    """Top-level Council deliberation outcome — not Decision lifecycle resolution."""

    SYNTHESIZED = "synthesized"
    DEADLOCK = "deadlock"


class CouncilSynthesisDisposition(str, Enum):
    """Per-round synthesis attempt outcome."""

    RESOLVED = "resolved"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


class CouncilDeadlockReasonCode(str, Enum):
    """Typed Council deadlock reason for lifecycle routing."""

    PERSISTENT_DISAGREEMENT = "persistent_disagreement"
    INSUFFICIENT_PROPOSALS = "insufficient_proposals"
    EXECUTION_BUDGET_EXHAUSTED = "execution_budget_exhausted"
    PARTICIPANT_FAILURE = "participant_failure"


@dataclass(frozen=True, slots=True)
class CouncilRoundPolicy:
    """Immutable semantic deliberation round budget."""

    max_rounds: int

    def __post_init__(self) -> None:
        if type(self.max_rounds) is not int or isinstance(self.max_rounds, bool):
            raise TypeError("CouncilRoundPolicy.max_rounds must be int")
        if self.max_rounds < 1:
            raise ValueError("CouncilRoundPolicy.max_rounds must be >= 1")


@dataclass(frozen=True, slots=True)
class CouncilParticipantFailurePolicy:
    """Fail-closed policy when participants do not produce enough proposals."""

    minimum_successful_participants: int

    def __post_init__(self) -> None:
        if (
            type(self.minimum_successful_participants) is not int
            or isinstance(self.minimum_successful_participants, bool)
        ):
            raise TypeError(
                "CouncilParticipantFailurePolicy.minimum_successful_participants "
                "must be int",
            )
        if self.minimum_successful_participants < 2:
            raise ValueError(
                "CouncilParticipantFailurePolicy.minimum_successful_participants "
                "must be >= 2",
            )


@dataclass(frozen=True, slots=True)
class CouncilSynthesisConfiguration:
    """Trusted synthesis instruction and participant failure semantics."""

    synthesis_instruction: str
    failure_policy: CouncilParticipantFailurePolicy

    def __post_init__(self) -> None:
        if type(self.synthesis_instruction) is not str:
            raise TypeError(
                "CouncilSynthesisConfiguration.synthesis_instruction must be str",
            )
        if not self.synthesis_instruction or not self.synthesis_instruction.strip():
            raise ValueError(
                "CouncilSynthesisConfiguration.synthesis_instruction "
                "must be non-empty",
            )
        if self.synthesis_instruction != self.synthesis_instruction.strip():
            raise ValueError(
                "CouncilSynthesisConfiguration.synthesis_instruction "
                "must not contain leading or trailing whitespace",
            )
        if type(self.failure_policy) is not CouncilParticipantFailurePolicy:
            raise TypeError(
                "CouncilSynthesisConfiguration.failure_policy must be "
                "CouncilParticipantFailurePolicy",
            )


def _validate_canonical_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class CouncilContextSurface:
    """One logical deliberation context channel with materializable messages."""

    context_id: DeliberationContextId
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        validate_deliberation_context_id(self.context_id)
        if type(self.messages) is not tuple:
            raise TypeError("CouncilContextSurface.messages must be tuple")
        if len(self.messages) == 0:
            raise ValueError("CouncilContextSurface.messages must not be empty")


def council_context_surface(
    *,
    context_id: object,
    messages: tuple[ChatMessage, ...],
) -> CouncilContextSurface:
    """Build one context surface with validated identifiers."""
    return CouncilContextSurface(
        context_id=validate_deliberation_context_id(context_id),
        messages=messages,
    )


def _canonicalize_context_surfaces(
    surfaces: tuple[CouncilContextSurface, ...],
) -> tuple[CouncilContextSurface, ...]:
    normalized: list[CouncilContextSurface] = []
    seen: set[DeliberationContextId] = set()
    for surface in surfaces:
        if type(surface) is not CouncilContextSurface:
            raise TypeError("context_surfaces must contain CouncilContextSurface")
        validated = council_context_surface(
            context_id=surface.context_id,
            messages=surface.messages,
        )
        if validated.context_id in seen:
            raise ValueError(
                "context_surfaces must not contain duplicate context_id: "
                f"{validated.context_id!r}",
            )
        seen.add(validated.context_id)
        normalized.append(validated)
    return tuple(sorted(normalized, key=lambda item: str(item.context_id)))


def _require_canonical_context_surfaces(
    surfaces: tuple[CouncilContextSurface, ...],
) -> None:
    canonical = _canonicalize_context_surfaces(surfaces)
    if surfaces != canonical:
        raise ValueError(
            "context_surfaces must be in canonical order without duplicates",
        )


@dataclass(frozen=True, slots=True)
class CouncilDeliberationInput(Generic[T]):
    """Provider-neutral Council deliberation input."""

    identity: DecisionIdentity
    task_messages: tuple[ChatMessage, ...]
    context_surfaces: tuple[CouncilContextSurface, ...]
    output_type: type[T]
    artifact_kind: DecisionArtifactKind

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError("CouncilDeliberationInput.identity must be DecisionIdentity")
        if type(self.task_messages) is not tuple:
            raise TypeError("CouncilDeliberationInput.task_messages must be tuple")
        if len(self.task_messages) == 0:
            raise ValueError("CouncilDeliberationInput.task_messages must not be empty")
        _require_canonical_context_surfaces(self.context_surfaces)
        if type(self.output_type) is not type:
            raise TypeError("CouncilDeliberationInput.output_type must be type")
        validate_decision_artifact_kind(self.artifact_kind)


@dataclass(frozen=True, slots=True)
class CouncilStrategy:
    """Multi-participant deliberation strategy configuration."""

    participants: ParticipantConfiguration
    visibility: ParticipantContextVisibilityConfiguration
    round_policy: CouncilRoundPolicy
    synthesis: CouncilSynthesisConfiguration
    kind: DecisionStrategyKind = field(default=_COUNCIL_KIND)

    def __post_init__(self) -> None:
        validated_kind = validate_decision_strategy_kind(self.kind)
        if validated_kind != _COUNCIL_KIND:
            raise ValueError(
                "CouncilStrategy.kind must be council "
                f"got {validated_kind!r}",
            )
        if type(self.participants) is not ParticipantConfiguration:
            raise TypeError("CouncilStrategy.participants must be ParticipantConfiguration")
        if len(self.participants.participants) < 2:
            raise ValueError(
                "CouncilStrategy requires at least two participants",
            )
        if type(self.visibility) is not ParticipantContextVisibilityConfiguration:
            raise TypeError(
                "CouncilStrategy.visibility must be ParticipantContextVisibilityConfiguration",
            )
        if type(self.round_policy) is not CouncilRoundPolicy:
            raise TypeError("CouncilStrategy.round_policy must be CouncilRoundPolicy")
        if type(self.synthesis) is not CouncilSynthesisConfiguration:
            raise TypeError("CouncilStrategy.synthesis must be CouncilSynthesisConfiguration")
        participant_count = len(self.participants.participants)
        minimum = self.synthesis.failure_policy.minimum_successful_participants
        if minimum > participant_count:
            raise ValueError(
                "minimum_successful_participants must be <= configured participant count",
            )


@dataclass(frozen=True, slots=True)
class CouncilParticipantProposal(Generic[T]):
    """Typed participant proposal bound to decision identity and lineage."""

    participant_id: ParticipantId
    role_id: ParticipantRoleId
    inference_profile_id: InferenceProfileId
    proposal_ref: DecisionProposalRef
    candidate: CandidateDecision[T]
    rationale_summary: str

    def __post_init__(self) -> None:
        validate_participant_id(self.participant_id)
        validate_participant_role_id(self.role_id)
        validate_inference_profile_id(self.inference_profile_id)
        if type(self.proposal_ref) is not DecisionProposalRef:
            raise TypeError(
                "CouncilParticipantProposal.proposal_ref must be DecisionProposalRef",
            )
        if type(self.candidate) is not CandidateDecision:
            raise TypeError(
                "CouncilParticipantProposal.candidate must be CandidateDecision",
            )
        _validate_canonical_string(
            self.rationale_summary,
            "CouncilParticipantProposal.rationale_summary",
        )


@dataclass(frozen=True, slots=True)
class CouncilRoundState:
    """Semantic artifacts retained for one deliberation round."""

    round_number: int
    proposal_refs: tuple[DecisionProposalRef, ...]
    disagreement: DecisionDisagreementArtifact | None

    def __post_init__(self) -> None:
        if type(self.round_number) is not int or isinstance(self.round_number, bool):
            raise TypeError("CouncilRoundState.round_number must be int")
        if self.round_number < 1:
            raise ValueError("CouncilRoundState.round_number must be >= 1")
        if type(self.proposal_refs) is not tuple:
            raise TypeError("CouncilRoundState.proposal_refs must be tuple")
        for ref in self.proposal_refs:
            if type(ref) is not DecisionProposalRef:
                raise TypeError(
                    "CouncilRoundState.proposal_refs must contain DecisionProposalRef",
                )


@dataclass(frozen=True, slots=True)
class CouncilSynthesisAttempt(Generic[T]):
    """One synthesis attempt — candidate or explicit unresolved conflict."""

    disposition: CouncilSynthesisDisposition
    candidate: CandidateDecision[T] | None = None

    def __post_init__(self) -> None:
        if self.disposition == CouncilSynthesisDisposition.RESOLVED:
            if self.candidate is None:
                raise ValueError(
                    "CouncilSynthesisAttempt with RESOLVED requires candidate",
                )
        elif self.disposition == CouncilSynthesisDisposition.UNRESOLVED_CONFLICT:
            if self.candidate is not None:
                raise ValueError(
                    "CouncilSynthesisAttempt with UNRESOLVED_CONFLICT "
                    "must not include candidate",
                )
        else:
            raise ValueError(f"unknown CouncilSynthesisDisposition: {self.disposition!r}")


class CouncilDisagreementAnalyzer(Protocol[T]):
    """Provider-neutral disagreement analysis port."""

    def analyze(
        self,
        *,
        proposals: tuple[CouncilParticipantProposal[T], ...],
    ) -> DecisionDisagreementArtifact:
        """Produce structured disagreement from independent proposals."""
        ...


class CouncilSynthesizer(Protocol[T]):
    """Provider-neutral synthesis port — produces non-authoritative candidates."""

    def synthesize(
        self,
        *,
        proposals: tuple[CouncilParticipantProposal[T], ...],
        disagreement: DecisionDisagreementArtifact,
        round_state: CouncilRoundState,
        synthesis_instruction: str,
    ) -> CouncilSynthesisAttempt[T]:
        """Attempt synthesis from proposals and structured disagreement."""
        ...


@dataclass(frozen=True, slots=True)
class CouncilDeliberationResult(Generic[T]):
    """Typed top-level Council outcome — exactly one valid state."""

    disposition: CouncilResolutionDisposition
    proposal_refs: tuple[DecisionProposalRef, ...]
    disagreement: DecisionDisagreementArtifact | None
    rounds_used: int
    candidate: CandidateDecision[T] | None = None
    deadlock_reason: CouncilDeadlockReasonCode | None = None

    def __post_init__(self) -> None:
        if type(self.rounds_used) is not int or isinstance(self.rounds_used, bool):
            raise TypeError("CouncilDeliberationResult.rounds_used must be int")
        if self.rounds_used < 1:
            raise ValueError("CouncilDeliberationResult.rounds_used must be >= 1")
        if type(self.proposal_refs) is not tuple:
            raise TypeError("CouncilDeliberationResult.proposal_refs must be tuple")
        for ref in self.proposal_refs:
            if type(ref) is not DecisionProposalRef:
                raise TypeError(
                    "CouncilDeliberationResult.proposal_refs must contain "
                    "DecisionProposalRef",
                )
        if self.disposition == CouncilResolutionDisposition.SYNTHESIZED:
            if self.candidate is None:
                raise ValueError(
                    "CouncilDeliberationResult SYNTHESIZED requires candidate",
                )
            if self.deadlock_reason is not None:
                raise ValueError(
                    "CouncilDeliberationResult SYNTHESIZED must not include "
                    "deadlock_reason",
                )
            if self.disagreement is None:
                raise ValueError(
                    "CouncilDeliberationResult SYNTHESIZED requires disagreement",
                )
        elif self.disposition == CouncilResolutionDisposition.DEADLOCK:
            if self.candidate is not None:
                raise ValueError(
                    "CouncilDeliberationResult DEADLOCK must not include candidate",
                )
            if self.deadlock_reason is None:
                raise ValueError(
                    "CouncilDeliberationResult DEADLOCK requires deadlock_reason",
                )
            if self.disagreement is None and self.deadlock_reason not in (
                CouncilDeadlockReasonCode.PARTICIPANT_FAILURE,
                CouncilDeadlockReasonCode.INSUFFICIENT_PROPOSALS,
                CouncilDeadlockReasonCode.EXECUTION_BUDGET_EXHAUSTED,
            ):
                raise ValueError(
                    "CouncilDeliberationResult DEADLOCK requires disagreement "
                    "except for participant failure reasons",
                )
        else:
            raise ValueError(f"unknown CouncilResolutionDisposition: {self.disposition!r}")


def council_round_policy(*, max_rounds: int) -> CouncilRoundPolicy:
    """Build one validated round policy."""
    return CouncilRoundPolicy(max_rounds=max_rounds)


def council_participant_failure_policy(
    *,
    minimum_successful_participants: int,
) -> CouncilParticipantFailurePolicy:
    """Build one validated participant failure policy."""
    return CouncilParticipantFailurePolicy(
        minimum_successful_participants=minimum_successful_participants,
    )


def council_synthesis_configuration(
    *,
    synthesis_instruction: str,
    failure_policy: CouncilParticipantFailurePolicy,
) -> CouncilSynthesisConfiguration:
    """Build one validated synthesis configuration."""
    return CouncilSynthesisConfiguration(
        synthesis_instruction=synthesis_instruction,
        failure_policy=failure_policy,
    )


def council_strategy_registration(
    *,
    participants: ParticipantConfiguration,
    visibility: ParticipantContextVisibilityConfiguration,
    round_policy: CouncilRoundPolicy,
    synthesis: CouncilSynthesisConfiguration,
) -> DecisionStrategyRegistration:
    """Build one explicit Council registration for host/bootstrap wiring."""
    strategy = CouncilStrategy(
        participants=participants,
        visibility=visibility,
        round_policy=round_policy,
        synthesis=synthesis,
    )
    return DecisionStrategyRegistration(kind=strategy.kind, strategy=strategy)


def register_council_strategy(
    registry: DecisionStrategyRegistry,
    *,
    participants: ParticipantConfiguration,
    visibility: ParticipantContextVisibilityConfiguration,
    round_policy: CouncilRoundPolicy,
    synthesis: CouncilSynthesisConfiguration,
) -> DecisionStrategyRegistry:
    """Register Council on ``registry``; return a new immutable registry."""
    return register_decision_strategy(
        registry,
        council_strategy_registration(
            participants=participants,
            visibility=visibility,
            round_policy=round_policy,
            synthesis=synthesis,
        ),
    )


def council_deliberation_result_synthesized(
    *,
    candidate: CandidateDecision[T],
    proposal_refs: tuple[DecisionProposalRef, ...],
    disagreement: DecisionDisagreementArtifact,
    rounds_used: int,
) -> CouncilDeliberationResult[T]:
    """Build one validated synthesized Council result."""
    return CouncilDeliberationResult(
        disposition=CouncilResolutionDisposition.SYNTHESIZED,
        proposal_refs=proposal_refs,
        disagreement=disagreement,
        rounds_used=rounds_used,
        candidate=candidate,
        deadlock_reason=None,
    )


def council_deliberation_result_deadlock[T](
    *,
    result_payload_type: type[T],
    proposal_refs: tuple[DecisionProposalRef, ...],
    disagreement: DecisionDisagreementArtifact | None,
    rounds_used: int,
    deadlock_reason: CouncilDeadlockReasonCode,
) -> CouncilDeliberationResult[T]:
    """Build one validated Council deadlock result."""
    if type(result_payload_type) is not type:
        raise TypeError("result_payload_type must be type")
    return CouncilDeliberationResult(
        disposition=CouncilResolutionDisposition.DEADLOCK,
        proposal_refs=proposal_refs,
        disagreement=disagreement,
        rounds_used=rounds_used,
        candidate=None,
        deadlock_reason=deadlock_reason,
    )
