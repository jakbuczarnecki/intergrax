# © Artur Czarnecki. All rights reserved.

"""DG-001B3 — HostedBootstrapFailureRecord producer contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting.bootstrap_failure import (
    BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID,
    BOOTSTRAP_FAILURE_RECORD_SCHEMA_VERSION,
    BootstrapIdentitySnapshot,
    BootstrapReadinessLevel,
    BootstrapSurfaceKind,
    DefaultBootstrapFailureClassifier,
    HostedBootstrapFailureProducer,
    HostedBootstrapFailureRecord,
    PromotionState,
    mint_bootstrap_attempt_id,
    run_guarded_hosted_bootstrap_segment,
)
from intergrax.hosting.process_bootstrap import (
    BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
    HostedProcessBootstrapPhase,
)

pytestmark = pytest.mark.unit


@dataclass
class _RecordingReporter:
    records: list[HostedBootstrapFailureRecord]

    def report(self, record: HostedBootstrapFailureRecord) -> None:
        self.records.append(record)


@dataclass
class _FailingReporter:
    def report(self, record: HostedBootstrapFailureRecord) -> None:
        raise RuntimeError("reporter failed")


def _identity(
    *,
    bootstrap_attempt_id: str | None = None,
    application_id: str = "local_workspace",
    process_role: str = "background_worker",
    instance_id: str | None = None,
    diagnostic_tenant_id: str | None = None,
) -> BootstrapIdentitySnapshot:
    return BootstrapIdentitySnapshot(
        bootstrap_attempt_id=bootstrap_attempt_id or mint_bootstrap_attempt_id(),
        application_id=application_id,
        process_role=process_role,
        instance_id=instance_id,
        diagnostic_tenant_id=diagnostic_tenant_id,
    )


def _producer(*reporters: _RecordingReporter | _FailingReporter) -> HostedBootstrapFailureProducer:
    return HostedBootstrapFailureProducer(reporters=list(reporters))


def test_build_record_populates_required_fields_without_fabricated_identity() -> None:
    attempt_id = mint_bootstrap_attempt_id()
    identity = _identity(bootstrap_attempt_id=attempt_id)
    producer = _producer(_RecordingReporter(records=[]))
    record = producer.build_record(
        readiness_at_failure=BootstrapReadinessLevel.B3_TENANT_BINDING,
        stage=HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION,
        identity=identity,
        surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
        exc=RuntimeError("secret dependency token"),
    )
    assert record.schema_id == BOOTSTRAP_FAILURE_RECORD_SCHEMA_ID
    assert record.schema_version == BOOTSTRAP_FAILURE_RECORD_SCHEMA_VERSION
    assert record.bootstrap_attempt_id == attempt_id
    assert record.readiness_at_failure is BootstrapReadinessLevel.B3_TENANT_BINDING
    assert record.stage is HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION
    assert record.surface_kind is BootstrapSurfaceKind.WORKER_BACKGROUND
    assert record.severity is EventSeverity.ERROR
    assert record.promotion_state is PromotionState.PENDING
    assert record.identity.instance_id is None
    assert record.identity.diagnostic_tenant_id is None
    assert (
        record.failure_facts.reason_code == BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE
    )
    assert record.failure_facts.exception_type == "RuntimeError"
    assert "secret dependency token" not in str(record)


def test_identity_snapshot_rejects_fabricated_instance_id() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        BootstrapIdentitySnapshot(
            bootstrap_attempt_id=mint_bootstrap_attempt_id(),
            instance_id="",
        )


def test_producer_invokes_all_reporters() -> None:
    first = _RecordingReporter(records=[])
    second = _RecordingReporter(records=[])
    producer = _producer(first, second)
    identity = _identity()
    record = producer.build_record(
        readiness_at_failure=BootstrapReadinessLevel.B2_CONFIGURATION,
        stage=HostedProcessBootstrapPhase.COMPOSITION,
        identity=identity,
        surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
        exc=TypeError("missing dependency"),
    )
    producer.emit(record)
    assert first.records == [record]
    assert second.records == [record]


def test_reporter_failure_does_not_mask_primary_failure() -> None:
    reporter = _RecordingReporter(records=[])
    producer = _producer(reporter, _FailingReporter())
    identity = _identity()
    with pytest.raises(ValueError, match="primary failure"):
        run_guarded_hosted_bootstrap_segment(
            producer=producer,
            readiness_at_failure=BootstrapReadinessLevel.B3_TENANT_BINDING,
            stage=HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION,
            identity=identity,
            surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
            segment=lambda: (_ for _ in ()).throw(ValueError("primary failure")),
        )
    assert len(reporter.records) == 1
    assert reporter.records[0].failure_facts.exception_type == "ValueError"


def test_guarded_segment_preserves_success_result() -> None:
    reporter = _RecordingReporter(records=[])
    producer = _producer(reporter)
    identity = _identity()
    result = run_guarded_hosted_bootstrap_segment(
        producer=producer,
        readiness_at_failure=BootstrapReadinessLevel.B3_TENANT_BINDING,
        stage=HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION,
        identity=identity,
        surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
        segment=lambda: {"ready": True},
    )
    assert result == {"ready": True}
    assert reporter.records == []


def test_identity_rules_no_instance_id_before_b1() -> None:
    attempt_id = mint_bootstrap_attempt_id()
    identity = BootstrapIdentitySnapshot(
        bootstrap_attempt_id=attempt_id,
        application_id="local_workspace",
        process_role="background_worker",
    )
    producer = _producer(_RecordingReporter(records=[]))
    record = producer.build_record(
        readiness_at_failure=BootstrapReadinessLevel.B2_CONFIGURATION,
        stage=HostedProcessBootstrapPhase.COMPOSITION,
        identity=identity,
        surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
        exc=RuntimeError("boom"),
        promotion_state=PromotionState.TERMINAL,
    )
    assert record.identity.instance_id is None
    assert record.readiness_at_failure is BootstrapReadinessLevel.B2_CONFIGURATION


def test_default_classifier_is_deterministic() -> None:
    classifier = DefaultBootstrapFailureClassifier()
    reason_code, exception_type = classifier.classify(RuntimeError("x"))
    assert reason_code == BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE
    assert exception_type == "RuntimeError"
