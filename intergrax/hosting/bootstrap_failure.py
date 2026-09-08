# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pre-B5 hosted bootstrap failure producer (DG-001B3)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol, TypeVar
from uuid import uuid4

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting.contracts.identity import normalize_application_id
from intergrax.hosting.contracts.public_data import (
    validate_bounded_identifier,
    validate_instance_id,
)
from intergrax.hosting.process_bootstrap import (
    BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
    HostedProcessBootstrapFailureFacts,
    HostedProcessBootstrapPhase,
)

_LOGGER = logging.getLogger(__name__)

BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID = "intergrax.hosting.bootstrap_failure_record"
BOOTSTRAP_FAILURE_RECORD_SCHEMA_VERSION = "1.0.0"
ENVIRONMENT_GATE_EXIT_EXCEPTION_TYPE = "EnvironmentGateExit"
_UNSPECIFIED_PROCESS_ROLE = "unspecified"

_T = TypeVar("_T")


class BootstrapReadinessLevel(str, Enum):
    B0_PROCESS = "b0_process"
    B2_CONFIGURATION = "b2_configuration"
    B3_TENANT_BINDING = "b3_tenant_binding"
    B4_OBSERVABILITY = "b4_observability"
    B5_DIAGNOSTICS = "b5_diagnostics"
    B1_INSTANCE = "b1_instance"


class BootstrapSurfaceKind(str, Enum):
    WORKER_BACKGROUND = "worker_background"
    HOSTED_FOREGROUND = "hosted_foreground"
    SUPERVISOR_PRE_ENGINE = "supervisor_pre_engine"


class PromotionState(str, Enum):
    PENDING = "pending"
    PROMOTED = "promoted"
    OBSERVABILITY_ONLY = "observability_only"
    TERMINAL = "terminal"


def mint_bootstrap_attempt_id() -> str:
    """Mint one bootstrap_attempt_id per process entrypoint invocation."""
    return validate_bounded_identifier(
        f"bootstrap-attempt-{uuid4()}",
        field_name="bootstrap_attempt_id",
    )


def _mint_bootstrap_failure_record_id() -> str:
    return validate_bounded_identifier(
        f"bootstrap-failure-{uuid4()}",
        field_name="record_id",
    )


@dataclass(frozen=True, slots=True)
class BootstrapIdentitySnapshot:
    """Honest identity fields known at failure detection time."""

    bootstrap_attempt_id: str
    application_id: str | None = None
    process_role: str | None = None
    instance_id: str | None = None
    diagnostic_tenant_id: str | None = None
    process_os_id: int | None = None

    def __post_init__(self) -> None:
        validated_attempt_id = validate_bounded_identifier(
            self.bootstrap_attempt_id,
            field_name="bootstrap_attempt_id",
        )
        if validated_attempt_id != self.bootstrap_attempt_id:
            raise ValueError(
                "bootstrap_attempt_id must be canonical (already normalized before construction)"
            )
        if self.application_id is not None:
            normalized_application_id = normalize_application_id(self.application_id)
            if normalized_application_id != self.application_id:
                raise ValueError(
                    "application_id must be canonical (already normalized before construction)"
                )
        if self.process_role is not None:
            validated_process_role = validate_bounded_identifier(
                self.process_role,
                field_name="process_role",
            )
            if validated_process_role != self.process_role:
                raise ValueError(
                    "process_role must be canonical (already normalized before construction)"
                )
        if self.instance_id is not None:
            validated_instance_id = validate_instance_id(self.instance_id)
            if validated_instance_id != self.instance_id:
                raise ValueError(
                    "instance_id must be canonical (already normalized before construction)"
                )
        if self.diagnostic_tenant_id is not None:
            validated_tenant_id = validate_bounded_identifier(
                self.diagnostic_tenant_id,
                field_name="diagnostic_tenant_id",
            )
            if validated_tenant_id != self.diagnostic_tenant_id:
                raise ValueError(
                    "diagnostic_tenant_id must be canonical "
                    "(already normalized before construction)"
                )


@dataclass(frozen=True, slots=True)
class HostedBootstrapFailureRecord:
    """Typed, immutable pre-B5 bootstrap failure fact — not a Problem."""

    schema_id: str
    schema_version: str
    record_id: str
    bootstrap_attempt_id: str
    detected_at: datetime
    readiness_at_failure: BootstrapReadinessLevel
    stage: HostedProcessBootstrapPhase
    identity: BootstrapIdentitySnapshot
    failure_facts: HostedProcessBootstrapFailureFacts
    severity: EventSeverity
    surface_kind: BootstrapSurfaceKind
    promotion_state: PromotionState

    def __post_init__(self) -> None:
        if self.schema_id != BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID:
            raise ValueError("invalid bootstrap failure record schema_id")
        if self.schema_version != BOOTSTRAP_FAILURE_RECORD_SCHEMA_VERSION:
            raise ValueError("invalid bootstrap failure record schema_version")
        validated_record_id = validate_bounded_identifier(
            self.record_id,
            field_name="record_id",
        )
        if validated_record_id != self.record_id:
            raise ValueError("record_id must be canonical")
        if self.bootstrap_attempt_id != self.identity.bootstrap_attempt_id:
            raise ValueError("bootstrap_attempt_id must match identity snapshot")


class BootstrapFailureClassifier(Protocol):
    def classify(self, exc: BaseException) -> tuple[str, str]:
        """Return bounded (reason_code, exception_type)."""


@dataclass(frozen=True, slots=True)
class DefaultBootstrapFailureClassifier:
    def classify(self, exc: BaseException) -> tuple[str, str]:
        return (
            BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
            type(exc).__name__,
        )


class BootstrapFailureReporter(Protocol):
    def report(self, record: HostedBootstrapFailureRecord) -> None:
        """Report a bootstrap failure record to one channel."""


@dataclass(frozen=True, slots=True)
class LoggingBootstrapFailureReporter:
    """Structured local log reporter — always available."""

    def report(self, record: HostedBootstrapFailureRecord) -> None:
        _LOGGER.error(
            "hosted bootstrap failure detected",
            extra={
                "record_id": record.record_id,
                "bootstrap_attempt_id": record.bootstrap_attempt_id,
                "readiness_at_failure": record.readiness_at_failure.value,
                "stage": record.stage.value,
                "surface_kind": record.surface_kind.value,
                "promotion_state": record.promotion_state.value,
                "reason_code": record.failure_facts.reason_code,
                "exception_type": record.failure_facts.exception_type,
                "application_id": record.identity.application_id,
                "process_role": record.identity.process_role,
                "instance_id": record.identity.instance_id,
                "diagnostic_tenant_id": record.identity.diagnostic_tenant_id,
            },
        )


class HostedBootstrapFailureProducer:
    """Builds HostedBootstrapFailureRecord facts and routes them to reporters."""

    def __init__(
        self,
        *,
        reporters: Sequence[BootstrapFailureReporter],
        classifier: BootstrapFailureClassifier | None = None,
    ) -> None:
        self._reporters = tuple(reporters)
        self._classifier = classifier or DefaultBootstrapFailureClassifier()

    def build_record(
        self,
        *,
        readiness_at_failure: BootstrapReadinessLevel,
        stage: HostedProcessBootstrapPhase,
        identity: BootstrapIdentitySnapshot,
        surface_kind: BootstrapSurfaceKind,
        exc: BaseException,
        promotion_state: PromotionState = PromotionState.PENDING,
        severity: EventSeverity = EventSeverity.ERROR,
        detected_at: datetime | None = None,
    ) -> HostedBootstrapFailureRecord:
        reason_code, exception_type = self._classifier.classify(exc)
        process_role = identity.process_role or _UNSPECIFIED_PROCESS_ROLE
        facts = HostedProcessBootstrapFailureFacts(
            phase=stage,
            reason_code=reason_code,
            exception_type=exception_type,
            process_role=validate_bounded_identifier(
                process_role,
                field_name="process_role",
            ),
        )
        return HostedBootstrapFailureRecord(
            schema_id=BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID,
            schema_version=BOOTSTRAP_FAILURE_RECORD_SCHEMA_VERSION,
            record_id=_mint_bootstrap_failure_record_id(),
            bootstrap_attempt_id=identity.bootstrap_attempt_id,
            detected_at=detected_at or datetime.now(UTC),
            readiness_at_failure=readiness_at_failure,
            stage=stage,
            identity=identity,
            failure_facts=facts,
            severity=severity,
            surface_kind=surface_kind,
            promotion_state=promotion_state,
        )

    def emit(self, record: HostedBootstrapFailureRecord) -> None:
        for reporter in self._reporters:
            try:
                reporter.report(record)
            except Exception as reporter_exc:
                _LOGGER.error(
                    "bootstrap failure reporter failed",
                    extra={
                        "record_id": record.record_id,
                        "bootstrap_attempt_id": record.bootstrap_attempt_id,
                        "reporter_type": type(reporter).__name__,
                        "reporter_exception_type": type(reporter_exc).__name__,
                    },
                )


def run_guarded_hosted_bootstrap_segment(
    *,
    producer: HostedBootstrapFailureProducer,
    readiness_at_failure: BootstrapReadinessLevel,
    stage: HostedProcessBootstrapPhase,
    identity: BootstrapIdentitySnapshot,
    surface_kind: BootstrapSurfaceKind,
    segment: Callable[[], _T],
    promotion_state: PromotionState = PromotionState.PENDING,
) -> _T:
    """Run a sync bootstrap segment and emit a failure record on application errors."""
    try:
        return segment()
    except Exception as exc:
        record = producer.build_record(
            readiness_at_failure=readiness_at_failure,
            stage=stage,
            identity=identity,
            surface_kind=surface_kind,
            exc=exc,
            promotion_state=promotion_state,
        )
        producer.emit(record)
        raise
