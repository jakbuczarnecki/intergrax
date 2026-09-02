# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision Verification observability via canonical RuntimeEvent spine (DS-VER-PIPE-07)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol

from pydantic import model_validator

from intergrax.contracts.decision_record import CandidateDecision, candidate_decision_ref
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
    VerificationStageOutcome,
    VerificationStageRecord,
)
from intergrax.contracts.decision_verification_stage import (
    T,
    VerificationStageRegistration,
)
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_kind_registry import register_event_kind
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.signals import emit_domain_signal

DECISION_VERIFICATION_SIGNAL_PAYLOAD_SCHEMA_ID = (
    "intergrax.decision.verification.signal.v1"
)

DECISION_VERIFICATION_STARTED_EVENT_KIND = "intergrax.decision.verification.started"
DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND = (
    "intergrax.decision.verification.stage_completed"
)
DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND = (
    "intergrax.decision.verification.stage_unavailable"
)
DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND = (
    "intergrax.decision.verification.probabilistic_skipped"
)
DECISION_VERIFICATION_COMPLETED_EVENT_KIND = (
    "intergrax.decision.verification.completed"
)

_DECISION_VERIFICATION_EVENT_KINDS: tuple[str, ...] = (
    DECISION_VERIFICATION_STARTED_EVENT_KIND,
    DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
    DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
    DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND,
    DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
)


class DecisionVerificationSignalPhase(StrEnum):
    """Semantic phase carried in Decision Verification domain signals."""

    STARTED = "started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_UNAVAILABLE = "stage_unavailable"
    PROBABILISTIC_SKIPPED = "probabilistic_skipped"
    COMPLETED = "completed"


class DecisionVerificationUnavailableReason(StrEnum):
    """Machine-readable unavailable reason category."""

    REQUIRED_UNAVAILABLE = "required_unavailable"


class DecisionVerificationSkipReason(StrEnum):
    """Machine-readable probabilistic short-circuit reason."""

    DETERMINISTIC_CHALLENGE = "deterministic_challenge"


def _validate_positive_int(value: int, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{label} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _validate_decision_version_value(value: int) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"decision_version must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError("decision_version must be >= 1")
    return value


def _validate_disposition_value(value: str) -> str:
    if value not in {item.value for item in VerificationDisposition}:
        raise ValueError(f"overall_disposition must be a VerificationDisposition value, got {value!r}")
    return value


def _validate_stage_outcome_value(value: str) -> str:
    if value not in {item.value for item in VerificationStageOutcome}:
        raise ValueError(f"stage_outcome must be a VerificationStageOutcome value, got {value!r}")
    return value


class DecisionVerificationSignalPayloadV1(RuntimeEventPayload):
    """Typed redaction-safe payload for Decision Verification domain signals."""

    schema_id = DECISION_VERIFICATION_SIGNAL_PAYLOAD_SCHEMA_ID

    phase: str
    decision_id: str
    decision_version: int
    branch_id: str
    stage_kind: str | None = None
    execution_class: str | None = None
    required: bool | None = None
    stage_outcome: str | None = None
    overall_disposition: str | None = None
    requirement_code: str | None = None
    finding_code: str | None = None
    executed_stage_count: int | None = None
    challenged_stage_count: int | None = None
    skipped_stage_count: int | None = None
    stage_count: int | None = None
    unavailable_reason_category: str | None = None
    skip_reason: str | None = None

    @model_validator(mode="after")
    def _validate_semantic_shape(self) -> DecisionVerificationSignalPayloadV1:
        _validate_decision_version_value(self.decision_version)
        if self.phase not in {item.value for item in DecisionVerificationSignalPhase}:
            raise ValueError(f"phase must be a DecisionVerificationSignalPhase value, got {self.phase!r}")
        if self.stage_outcome is not None:
            _validate_stage_outcome_value(self.stage_outcome)
        if self.overall_disposition is not None:
            _validate_disposition_value(self.overall_disposition)
        if self.executed_stage_count is not None:
            _validate_positive_int(self.executed_stage_count, "executed_stage_count")
        if self.challenged_stage_count is not None:
            _validate_positive_int(self.challenged_stage_count, "challenged_stage_count")
        if self.skipped_stage_count is not None:
            _validate_positive_int(self.skipped_stage_count, "skipped_stage_count")
        if self.stage_count is not None:
            _validate_positive_int(self.stage_count, "stage_count")
            if self.stage_count < 1:
                raise ValueError("stage_count must be >= 1")
        phase = DecisionVerificationSignalPhase(self.phase)
        if phase is DecisionVerificationSignalPhase.STARTED:
            if self.stage_kind is not None or self.stage_outcome is not None:
                raise ValueError("started phase must not include stage outcome fields")
            if self.overall_disposition is not None:
                raise ValueError("started phase must not include overall_disposition")
            if self.stage_count is None or self.stage_count < 1:
                raise ValueError("started phase requires stage_count >= 1")
        if phase is DecisionVerificationSignalPhase.STAGE_COMPLETED:
            if self.stage_kind is None or self.stage_outcome is None:
                raise ValueError("stage_completed phase requires stage_kind and stage_outcome")
            if self.required is None or self.execution_class is None:
                raise ValueError("stage_completed phase requires required and execution_class")
        if phase is DecisionVerificationSignalPhase.STAGE_UNAVAILABLE:
            if self.stage_kind is None or self.required is None:
                raise ValueError("stage_unavailable phase requires stage_kind and required")
            if self.required and self.unavailable_reason_category != (
                DecisionVerificationUnavailableReason.REQUIRED_UNAVAILABLE.value
            ):
                raise ValueError(
                    "required stage_unavailable must set unavailable_reason_category to "
                    "required_unavailable",
                )
        if phase is DecisionVerificationSignalPhase.PROBABILISTIC_SKIPPED:
            if self.skipped_stage_count is None or self.skipped_stage_count < 1:
                raise ValueError("probabilistic_skipped phase requires skipped_stage_count >= 1")
            if self.skip_reason != DecisionVerificationSkipReason.DETERMINISTIC_CHALLENGE.value:
                raise ValueError(
                    "probabilistic_skipped phase requires skip_reason deterministic_challenge",
                )
        if phase is DecisionVerificationSignalPhase.COMPLETED:
            if self.overall_disposition is None:
                raise ValueError("completed phase requires overall_disposition")
            if self.executed_stage_count is None or self.challenged_stage_count is None:
                raise ValueError("completed phase requires executed and challenged stage counts")
        return self

    def redact(self) -> DecisionVerificationSignalPayloadV1:
        """Return a production-safe copy; fields are identifiers and codes only."""
        return self


def register_decision_verification_domain_signals() -> None:
    """Register Decision Verification payload schema and domain event kinds (idempotent)."""
    register_payload_schema(DecisionVerificationSignalPayloadV1, extension=True)
    for kind in _DECISION_VERIFICATION_EVENT_KINDS:
        register_event_kind(kind, DECISION_VERIFICATION_SIGNAL_PAYLOAD_SCHEMA_ID)


def _identity_fields_from_candidate(
    candidate: CandidateDecision[T],
) -> tuple[str, int, str]:
    proposal_ref = candidate_decision_ref(candidate)
    return (
        str(proposal_ref.identity.decision_id),
        proposal_ref.identity.version.value,
        str(proposal_ref.lineage_ref.branch_id),
    )


def _validate_emit_context_lineage(
    candidate: CandidateDecision[T],
    ctx: EmitContext,
) -> None:
    execution = candidate.identity.execution
    if execution.task_id != ctx.task_id:
        raise ValueError("candidate execution task_id must match EmitContext.task_id")
    if execution.run_id != ctx.run_id:
        raise ValueError("candidate execution run_id must match EmitContext.run_id")
    if execution.attempt_id != ctx.attempt_id:
        raise ValueError("candidate execution attempt_id must match EmitContext.attempt_id")
    if execution.execution_id is not None and execution.execution_id != ctx.execution_id:
        raise ValueError("candidate execution execution_id must match EmitContext.execution_id")
    if ctx.tenant_id is not None and candidate.identity.tenant_id != ctx.tenant_id:
        raise ValueError("candidate tenant_id must match EmitContext.tenant_id")


def _emit_decision_verification_signal(
    ctx: EmitContext,
    *,
    kind: str,
    payload: DecisionVerificationSignalPayloadV1,
) -> RuntimeEvent:
    register_decision_verification_domain_signals()
    return emit_domain_signal(ctx, kind=kind, payload=payload)


class VerificationObserver(Protocol, Generic[T]):
    """Optional observability seam for Decision Verification pipeline runs."""

    def verification_started(
        self,
        candidate: CandidateDecision[T],
        *,
        stage_count: int,
    ) -> None: ...

    def stage_completed(
        self,
        candidate: CandidateDecision[T],
        registration: VerificationStageRegistration[T],
        record: VerificationStageRecord,
    ) -> None: ...

    def stage_unavailable(
        self,
        candidate: CandidateDecision[T],
        registration: VerificationStageRegistration[T],
        *,
        required: bool,
    ) -> None: ...

    def probabilistic_skipped(
        self,
        candidate: CandidateDecision[T],
        *,
        skipped_stage_count: int,
    ) -> None: ...

    def verification_completed(
        self,
        candidate: CandidateDecision[T],
        result: VerificationResult,
        *,
        executed_stage_count: int,
        challenged_stage_count: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeEventVerificationObserver(Generic[T]):
    """Emit Decision Verification signals through the canonical RuntimeEvent spine."""

    ctx: EmitContext

    def verification_started(
        self,
        candidate: CandidateDecision[T],
        *,
        stage_count: int,
    ) -> None:
        _validate_emit_context_lineage(candidate, self.ctx)
        decision_id, decision_version, branch_id = _identity_fields_from_candidate(candidate)
        payload = DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.STARTED.value,
            decision_id=decision_id,
            decision_version=decision_version,
            branch_id=branch_id,
            stage_count=_validate_positive_int(stage_count, "stage_count"),
        )
        _emit_decision_verification_signal(
            self.ctx,
            kind=DECISION_VERIFICATION_STARTED_EVENT_KIND,
            payload=payload,
        )

    def stage_completed(
        self,
        candidate: CandidateDecision[T],
        registration: VerificationStageRegistration[T],
        record: VerificationStageRecord,
    ) -> None:
        _validate_emit_context_lineage(candidate, self.ctx)
        decision_id, decision_version, branch_id = _identity_fields_from_candidate(candidate)
        requirement_code: str | None = None
        finding_code: str | None = None
        if record.challenge is not None:
            requirement_code = str(record.challenge.requirement_code)
            finding_code = str(record.challenge.finding.code)
        payload = DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.STAGE_COMPLETED.value,
            decision_id=decision_id,
            decision_version=decision_version,
            branch_id=branch_id,
            stage_kind=str(record.stage),
            execution_class=registration.stage.execution_class.value,
            required=registration.required,
            stage_outcome=record.outcome.value,
            requirement_code=requirement_code,
            finding_code=finding_code,
        )
        _emit_decision_verification_signal(
            self.ctx,
            kind=DECISION_VERIFICATION_STAGE_COMPLETED_EVENT_KIND,
            payload=payload,
        )

    def stage_unavailable(
        self,
        candidate: CandidateDecision[T],
        registration: VerificationStageRegistration[T],
        *,
        required: bool,
    ) -> None:
        _validate_emit_context_lineage(candidate, self.ctx)
        decision_id, decision_version, branch_id = _identity_fields_from_candidate(candidate)
        unavailable_reason_category: str | None = None
        if required:
            unavailable_reason_category = (
                DecisionVerificationUnavailableReason.REQUIRED_UNAVAILABLE.value
            )
        payload = DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.STAGE_UNAVAILABLE.value,
            decision_id=decision_id,
            decision_version=decision_version,
            branch_id=branch_id,
            stage_kind=str(registration.kind),
            execution_class=registration.stage.execution_class.value,
            required=required,
            unavailable_reason_category=unavailable_reason_category,
        )
        _emit_decision_verification_signal(
            self.ctx,
            kind=DECISION_VERIFICATION_STAGE_UNAVAILABLE_EVENT_KIND,
            payload=payload,
        )

    def probabilistic_skipped(
        self,
        candidate: CandidateDecision[T],
        *,
        skipped_stage_count: int,
    ) -> None:
        _validate_emit_context_lineage(candidate, self.ctx)
        decision_id, decision_version, branch_id = _identity_fields_from_candidate(candidate)
        payload = DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.PROBABILISTIC_SKIPPED.value,
            decision_id=decision_id,
            decision_version=decision_version,
            branch_id=branch_id,
            skipped_stage_count=_validate_positive_int(
                skipped_stage_count,
                "skipped_stage_count",
            ),
            skip_reason=DecisionVerificationSkipReason.DETERMINISTIC_CHALLENGE.value,
        )
        _emit_decision_verification_signal(
            self.ctx,
            kind=DECISION_VERIFICATION_PROBABILISTIC_SKIPPED_EVENT_KIND,
            payload=payload,
        )

    def verification_completed(
        self,
        candidate: CandidateDecision[T],
        result: VerificationResult,
        *,
        executed_stage_count: int,
        challenged_stage_count: int,
    ) -> None:
        _validate_emit_context_lineage(candidate, self.ctx)
        decision_id, decision_version, branch_id = _identity_fields_from_candidate(candidate)
        payload = DecisionVerificationSignalPayloadV1(
            phase=DecisionVerificationSignalPhase.COMPLETED.value,
            decision_id=decision_id,
            decision_version=decision_version,
            branch_id=branch_id,
            overall_disposition=result.disposition.value,
            executed_stage_count=_validate_positive_int(
                executed_stage_count,
                "executed_stage_count",
            ),
            challenged_stage_count=_validate_positive_int(
                challenged_stage_count,
                "challenged_stage_count",
            ),
        )
        _emit_decision_verification_signal(
            self.ctx,
            kind=DECISION_VERIFICATION_COMPLETED_EVENT_KIND,
            payload=payload,
        )
