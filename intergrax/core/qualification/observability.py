# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider qualification execution observability via platform HOS contracts (PROVIDER-QUAL-7-R2)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.core.qualification.provider import (
    ProviderQualificationExecutor,
    ProviderQualificationRun,
    ProviderQualificationSubject,
)
from intergrax.core.qualification.status import QualificationStatus
from intergrax.core.qualification.validity import QualificationRunId
from intergrax.runtime.observability.export_attributes import ApplicationObservabilityAttributes
from intergrax.runtime.observability.export_boundary import (
    ExportStatus,
    ObservabilityExportEnvelope,
    PlatformObservabilityExportSource,
    envelope_from_platform_observability_source,
)
from intergrax.runtime.observability.export_policy import (
    ExportPolicyResult,
    ObservabilityExportPolicy,
    apply_observability_export_policy,
)
from intergrax.runtime.observability.problem_export import envelope_from_problem_signal
from intergrax.runtime.observability.problem_reporter import (
    ProblemReportContext,
    build_problem_signal,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE,
    PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_INTEGRATION,
    PROBLEM_SOURCE_LAYER_RUNTIME,
)

logger = logging.getLogger(__name__)

PLATFORM_PROVIDER_QUALIFICATION_EXECUTION_EVENT_SCHEMA = (
    "intergrax.provider_qualification_execution_event.v1"
)


class ProviderQualificationExecutionEventType(StrEnum):
    """Non-execution platform lifecycle events for qualification execution."""

    STARTED = "provider_qualification.execution.started"
    COMPLETED = "provider_qualification.execution.completed"
    RECOVERED = "provider_qualification.execution.recovered"


class ProviderQualificationInfrastructurePhase(StrEnum):
    """Infrastructure failure phases for qualification execution."""

    RESOLUTION = "resolution"
    MATERIALIZATION = "materialization"
    SUITE = "suite"
    PERSISTENCE = "persistence"


class ProviderQualificationExecutionObservabilityAttributes(ApplicationObservabilityAttributes):
    """Policy-safe qualification execution metadata for platform observability export."""

    namespace: str = "provider_qualification"
    qualification_run_id: str
    provider_id: str
    capability_id: str
    qualification_suite_id: str
    qualification_suite_version: str
    domain: str
    executor_kind: str = ""
    executor_id: str = ""
    executor_version: str = ""
    source_revision: str = ""
    outcome_status: str = ""
    recovery_kind: str = ""


@dataclass(frozen=True, slots=True)
class ProviderQualificationExecutionEvent:
    """Typed authoring envelope for non-execution qualification lifecycle facts."""

    event_id: str
    event_type: ProviderQualificationExecutionEventType
    occurred_at: datetime
    qualification_run_id: QualificationRunId
    subject: ProviderQualificationSubject
    executor: ProviderQualificationExecutor
    source_revision: str
    status: QualificationStatus | None = None
    recovery_kind: str = ""

    @property
    def correlation_id(self) -> str:
        return str(self.qualification_run_id)


def qualification_execution_event_id(
    *,
    qualification_run_id: QualificationRunId | str,
    event_type: ProviderQualificationExecutionEventType,
) -> str:
    """Deterministic non-execution event identity for one qualification lifecycle fact."""
    digest = hashlib.sha256(
        f"{event_type.value}:{qualification_run_id}".encode("utf-8"),
    ).hexdigest()[:32]
    return f"pqev_{digest}"


def qualification_infrastructure_problem_event_id(
    *,
    qualification_run_id: QualificationRunId | str,
    phase: ProviderQualificationInfrastructurePhase,
    error_code: str,
) -> str:
    """Deterministic problem/event identity per infrastructure phase and error code."""
    digest = hashlib.sha256(
        f"infrastructure:{phase.value}:{error_code}:{qualification_run_id}".encode("utf-8"),
    ).hexdigest()[:32]
    return f"pqev_{digest}"


def qualification_infrastructure_source_layer(
    phase: ProviderQualificationInfrastructurePhase,
) -> str:
    """Map infrastructure phase to canonical platform problem source layer."""
    if phase in (
        ProviderQualificationInfrastructurePhase.RESOLUTION,
        ProviderQualificationInfrastructurePhase.MATERIALIZATION,
    ):
        return PROBLEM_SOURCE_LAYER_INTEGRATION
    return PROBLEM_SOURCE_LAYER_RUNTIME


def safe_record_qualification_observability(
    callback: Callable[[], None],
    *,
    operation: str,
) -> None:
    """Invoke observability port callback; failures never alter qualification truth."""
    try:
        callback()
    except Exception:
        logger.warning(
            "qualification observability %s failed (best-effort; qualification truth unchanged)",
            operation,
            exc_info=True,
        )


def _attributes_from_event(
    event: ProviderQualificationExecutionEvent,
) -> ProviderQualificationExecutionObservabilityAttributes:
    status_value = event.status.value if event.status is not None else ""
    return ProviderQualificationExecutionObservabilityAttributes(
        qualification_run_id=str(event.qualification_run_id),
        provider_id=event.subject.provider_id,
        capability_id=event.subject.capability_id,
        qualification_suite_id=event.subject.qualification_suite_id,
        qualification_suite_version=event.subject.qualification_suite_version,
        domain=event.subject.domain,
        executor_kind=event.executor.executor_kind,
        executor_id=event.executor.executor_id,
        executor_version=event.executor.executor_version or "",
        source_revision=event.source_revision,
        outcome_status=status_value,
        recovery_kind=event.recovery_kind,
    )


def qualification_execution_to_platform_export_source(
    event: ProviderQualificationExecutionEvent,
) -> PlatformObservabilityExportSource:
    """Project qualification execution lifecycle to a non-execution platform export source."""
    return PlatformObservabilityExportSource(
        event_id=event.event_id,
        source_schema_id=PLATFORM_PROVIDER_QUALIFICATION_EXECUTION_EVENT_SCHEMA,
        event_type=event.event_type.value,
        occurred_at=event.occurred_at,
        correlation_id=event.correlation_id,
        application_attributes=_attributes_from_event(event),
    )


def envelope_from_qualification_execution_event(
    event: ProviderQualificationExecutionEvent,
) -> ObservabilityExportEnvelope:
    """Map a qualification lifecycle event to a platform observability export envelope."""
    envelope = envelope_from_platform_observability_source(
        qualification_execution_to_platform_export_source(event),
    )
    if event.status is QualificationStatus.REJECTED:
        return envelope.model_copy(update={"status": ExportStatus.FAILED})
    if event.status in (
        QualificationStatus.PRODUCTION_QUALIFIED,
        QualificationStatus.QUALIFIED,
    ):
        return envelope.model_copy(update={"status": ExportStatus.SUCCEEDED})
    return envelope


def build_qualification_infrastructure_problem_envelope(
    *,
    qualification_run_id: QualificationRunId,
    subject: ProviderQualificationSubject,
    executor: ProviderQualificationExecutor,
    source_revision: str,
    phase: ProviderQualificationInfrastructurePhase,
    error_type: str,
    error_code: str,
) -> ObservabilityExportEnvelope:
    """Build a platform problem export envelope for qualification infrastructure failure."""
    safe_error_type = (error_type or "Exception").strip() or "Exception"
    problem_kind = (
        PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE
        if phase is ProviderQualificationInfrastructurePhase.PERSISTENCE
        else PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE
    )
    problem_event_id = qualification_infrastructure_problem_event_id(
        qualification_run_id=qualification_run_id,
        phase=phase,
        error_code=error_code,
    )
    signal = build_problem_signal(
        context=ProblemReportContext(
            correlation_id=str(qualification_run_id),
            capability=subject.capability_id,
        ),
        problem_kind=problem_kind,
        severity=PROBLEM_SEVERITY_ERROR,
        error_code=error_code,
        source_layer=qualification_infrastructure_source_layer(phase),
        source_component=f"provider_qualification.{phase.value}",
        problem_id=problem_event_id,
        event_id=problem_event_id,
        application_attributes=ProviderQualificationExecutionObservabilityAttributes(
            qualification_run_id=str(qualification_run_id),
            provider_id=subject.provider_id,
            capability_id=subject.capability_id,
            qualification_suite_id=subject.qualification_suite_id,
            qualification_suite_version=subject.qualification_suite_version,
            domain=subject.domain,
            executor_kind=executor.executor_kind,
            executor_id=executor.executor_id,
            executor_version=executor.executor_version or "",
            source_revision=source_revision,
            outcome_status="infrastructure_failure",
        ),
    ).model_copy(update={"exception_type": safe_error_type})
    envelope = envelope_from_problem_signal(signal)
    return envelope.model_copy(
        update={
            "event_type": f"provider_qualification.infrastructure.{phase.value}",
            "problem_error_code": error_code,
        },
    )


@runtime_checkable
class ProviderQualificationExecutionObservabilityPort(Protocol):
    """Optional best-effort observability port; failures must not alter qualification truth."""

    def record_execution_started(
        self,
        *,
        qualification_run_id: QualificationRunId,
        subject: ProviderQualificationSubject,
        executor: ProviderQualificationExecutor,
        source_revision: str,
        occurred_at: datetime,
    ) -> None:
        """Record canonical qualification execution start."""

    def record_execution_completed(
        self,
        run: ProviderQualificationRun,
        *,
        occurred_at: datetime,
    ) -> None:
        """Record canonical qualification execution completion."""

    def record_execution_recovered(
        self,
        run: ProviderQualificationRun,
        *,
        recovery_kind: str,
        occurred_at: datetime,
    ) -> None:
        """Record idempotent recovery from persisted qualification evidence."""

    def record_infrastructure_failure(
        self,
        *,
        qualification_run_id: QualificationRunId,
        subject: ProviderQualificationSubject,
        executor: ProviderQualificationExecutor,
        source_revision: str,
        phase: ProviderQualificationInfrastructurePhase,
        error_type: str,
        error_code: str,
        occurred_at: datetime,
    ) -> None:
        """Record qualification infrastructure failure on the platform problem plane."""


class NoOpProviderQualificationExecutionObservability:
    """Default observability posture — records nothing."""

    def record_execution_started(
        self,
        *,
        qualification_run_id: QualificationRunId,
        subject: ProviderQualificationSubject,
        executor: ProviderQualificationExecutor,
        source_revision: str,
        occurred_at: datetime,
    ) -> None:
        return None

    def record_execution_completed(
        self,
        run: ProviderQualificationRun,
        *,
        occurred_at: datetime,
    ) -> None:
        return None

    def record_execution_recovered(
        self,
        run: ProviderQualificationRun,
        *,
        recovery_kind: str,
        occurred_at: datetime,
    ) -> None:
        return None

    def record_infrastructure_failure(
        self,
        *,
        qualification_run_id: QualificationRunId,
        subject: ProviderQualificationSubject,
        executor: ProviderQualificationExecutor,
        source_revision: str,
        phase: ProviderQualificationInfrastructurePhase,
        error_type: str,
        error_code: str,
        occurred_at: datetime,
    ) -> None:
        return None


@dataclass
class RecordingProviderQualificationExecutionObservability:
    """TEST / LOCAL PROOF collector — not enterprise production diagnostics.

    Retains policy-sanitized ``ObservabilityExportEnvelope`` instances in memory for
    unit tests and local proof runs. Production qualification hosts must inject the
    canonical platform observability adapter wired to HOS/export/diagnostics composition.
    """

    policy: ObservabilityExportPolicy = ObservabilityExportPolicy(enabled=True)
    envelopes: list[ObservabilityExportEnvelope] | None = None

    def __post_init__(self) -> None:
        if self.envelopes is None:
            self.envelopes = []

    def _record_envelope(self, envelope: ObservabilityExportEnvelope) -> ExportPolicyResult:
        result = apply_observability_export_policy(envelope, self.policy)
        if result.exported and result.envelope is not None:
            envelopes = self.envelopes
            if envelopes is None:
                envelopes = []
                self.envelopes = envelopes
            envelopes.append(result.envelope)
        return result

    def record_execution_started(
        self,
        *,
        qualification_run_id: QualificationRunId,
        subject: ProviderQualificationSubject,
        executor: ProviderQualificationExecutor,
        source_revision: str,
        occurred_at: datetime,
    ) -> None:
        event = ProviderQualificationExecutionEvent(
            event_id=qualification_execution_event_id(
                qualification_run_id=qualification_run_id,
                event_type=ProviderQualificationExecutionEventType.STARTED,
            ),
            event_type=ProviderQualificationExecutionEventType.STARTED,
            occurred_at=occurred_at,
            qualification_run_id=qualification_run_id,
            subject=subject,
            executor=executor,
            source_revision=source_revision,
        )
        self._record_envelope(envelope_from_qualification_execution_event(event))

    def record_execution_completed(
        self,
        run: ProviderQualificationRun,
        *,
        occurred_at: datetime,
    ) -> None:
        event = ProviderQualificationExecutionEvent(
            event_id=qualification_execution_event_id(
                qualification_run_id=run.qualification_run_id,
                event_type=ProviderQualificationExecutionEventType.COMPLETED,
            ),
            event_type=ProviderQualificationExecutionEventType.COMPLETED,
            occurred_at=occurred_at,
            qualification_run_id=run.qualification_run_id,
            subject=run.subject,
            executor=run.executor,
            source_revision=run.source_revision,
            status=run.status,
        )
        self._record_envelope(envelope_from_qualification_execution_event(event))

    def record_execution_recovered(
        self,
        run: ProviderQualificationRun,
        *,
        recovery_kind: str,
        occurred_at: datetime,
    ) -> None:
        event = ProviderQualificationExecutionEvent(
            event_id=qualification_execution_event_id(
                qualification_run_id=run.qualification_run_id,
                event_type=ProviderQualificationExecutionEventType.RECOVERED,
            ),
            event_type=ProviderQualificationExecutionEventType.RECOVERED,
            occurred_at=occurred_at,
            qualification_run_id=run.qualification_run_id,
            subject=run.subject,
            executor=run.executor,
            source_revision=run.source_revision,
            status=run.status,
            recovery_kind=recovery_kind,
        )
        self._record_envelope(envelope_from_qualification_execution_event(event))

    def record_infrastructure_failure(
        self,
        *,
        qualification_run_id: QualificationRunId,
        subject: ProviderQualificationSubject,
        executor: ProviderQualificationExecutor,
        source_revision: str,
        phase: ProviderQualificationInfrastructurePhase,
        error_type: str,
        error_code: str,
        occurred_at: datetime,
    ) -> None:
        _ = occurred_at
        envelope = build_qualification_infrastructure_problem_envelope(
            qualification_run_id=qualification_run_id,
            subject=subject,
            executor=executor,
            source_revision=source_revision,
            phase=phase,
            error_type=error_type,
            error_code=error_code,
        )
        self._record_envelope(envelope)


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "NoOpProviderQualificationExecutionObservability",
    "PLATFORM_PROVIDER_QUALIFICATION_EXECUTION_EVENT_SCHEMA",
    "ProviderQualificationExecutionEvent",
    "ProviderQualificationExecutionEventType",
    "ProviderQualificationExecutionObservabilityAttributes",
    "ProviderQualificationExecutionObservabilityPort",
    "ProviderQualificationInfrastructurePhase",
    "RecordingProviderQualificationExecutionObservability",
    "build_qualification_infrastructure_problem_envelope",
    "envelope_from_qualification_execution_event",
    "qualification_execution_event_id",
    "qualification_execution_to_platform_export_source",
    "qualification_infrastructure_problem_event_id",
    "qualification_infrastructure_source_layer",
    "safe_record_qualification_observability",
    "utc_now",
]
