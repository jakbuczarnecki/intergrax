"""Map VPI observations to neutral platform stage signals (P1B)."""

from __future__ import annotations

from intergrax.contracts.application_execution_stage_signal import (
    ApplicationExecutionStageSignal,
)
from intergrax.contracts.event_severity import EventSeverity

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ClarificationObservedPayload,
    FusionObservedPayload,
    IdentityEvaluationObservedPayload,
    IdentityHypothesesObservedPayload,
    ProductIdentificationEventKind,
    ProductIdentificationObservation,
    ProductIdentificationStage,
    QueryContextObservedPayload,
    RetrievalChannelObservedPayload,
    StageFailureObservedPayload,
    StageTimingObservedPayload,
    TerminalObservedPayload,
    VerificationObservedPayload,
)

VPI_APPLICATION_SLUG = "verified_product_identification"


def project_product_identification_observation(
    observation: ProductIdentificationObservation,
) -> ApplicationExecutionStageSignal:
    """Project one VPI observation into a platform-neutral stage signal."""
    severity = _severity_for_observation(observation)
    summary = _summary_for_observation(observation)
    outcome_status, diagnostic_code = _outcome_fields(observation)
    return ApplicationExecutionStageSignal(
        application_slug=VPI_APPLICATION_SLUG,
        scenario_execution_correlation_id=observation.run_id.value,
        sequence=observation.sequence,
        stage_id=observation.stage.value,
        event_category=observation.kind.value,
        severity=severity,
        summary=summary,
        outcome_status=outcome_status,
        diagnostic_code=diagnostic_code,
    )


def _severity_for_observation(observation: ProductIdentificationObservation) -> EventSeverity:
    if observation.kind is ProductIdentificationEventKind.STAGE_FAILURE:
        return EventSeverity.ERROR
    if observation.kind is ProductIdentificationEventKind.TERMINAL:
        return EventSeverity.INFO
    return EventSeverity.INFO


def _outcome_fields(
    observation: ProductIdentificationObservation,
) -> tuple[str | None, str | None]:
    payload = observation.payload
    if isinstance(payload, TerminalObservedPayload):
        return payload.outcome.value, payload.reason_code.value
    if isinstance(payload, VerificationObservedPayload):
        return payload.terminal_outcome.value, payload.reason_code.value
    if isinstance(payload, StageFailureObservedPayload):
        if payload.catalog_failure is not None:
            return "stage_failed", payload.catalog_failure.kind.value
        if payload.observation_sink_failed:
            return "observation_sink_failed", "observation_sink_failed"
    return None, None


def _summary_for_observation(observation: ProductIdentificationObservation) -> str:
    kind = observation.kind
    payload = observation.payload
    if kind is ProductIdentificationEventKind.QUERY_CONTEXT:
        return "query_context stage observed"
    if kind is ProductIdentificationEventKind.STAGE_TIMING:
        if isinstance(payload, StageTimingObservedPayload):
            return (
                f"stage_timing stage={payload.stage.value} "
                f"duration_ns={payload.duration_ns}"
            )
        return "stage_timing observed"
    if kind is ProductIdentificationEventKind.RETRIEVAL_CHANNEL:
        if isinstance(payload, RetrievalChannelObservedPayload):
            return (
                f"retrieval channel={payload.channel.value} "
                f"status={payload.status.value} candidates={payload.candidate_count}"
            )
        return "retrieval_channel observed"
    if kind is ProductIdentificationEventKind.FUSION:
        if isinstance(payload, FusionObservedPayload):
            return f"fusion merged_offers={payload.merged_offer_count}"
        return "fusion observed"
    if kind is ProductIdentificationEventKind.IDENTITY_HYPOTHESES:
        if isinstance(payload, IdentityHypothesesObservedPayload):
            return f"identity_hypotheses count={len(payload.hypotheses)}"
        return "identity_hypotheses observed"
    if kind is ProductIdentificationEventKind.IDENTITY_EVALUATION:
        if isinstance(payload, IdentityEvaluationObservedPayload):
            return f"identity_evaluation count={len(payload.evaluated)}"
        return "identity_evaluation observed"
    if kind is ProductIdentificationEventKind.VERIFICATION:
        if isinstance(payload, VerificationObservedPayload):
            return f"verification outcome={payload.terminal_outcome.value}"
        return "verification observed"
    if kind is ProductIdentificationEventKind.CLARIFICATION:
        if isinstance(payload, ClarificationObservedPayload):
            return f"clarification required={payload.clarification_required}"
        return "clarification observed"
    if kind is ProductIdentificationEventKind.TERMINAL:
        if isinstance(payload, TerminalObservedPayload):
            return (
                f"terminal outcome={payload.outcome.value} "
                f"reason={payload.reason_code.value}"
            )
        return "terminal observed"
    if kind is ProductIdentificationEventKind.STAGE_FAILURE:
        if isinstance(payload, StageFailureObservedPayload):
            if payload.observation_sink_failed:
                return "stage_failure observation_sink_failed"
            if payload.catalog_failure is not None:
                return f"stage_failure kind={payload.catalog_failure.kind.value}"
        return "stage_failure observed"
    if observation.stage is ProductIdentificationStage.TERMINAL:
        return "terminal stage observed"
    return f"{kind.value} observed"
