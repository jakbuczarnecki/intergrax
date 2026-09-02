# © Artur Czarnecki. All rights reserved.

"""Observability isolation and platform signal tests (PROVIDER-QUAL-7-R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.repository_qualification_suite import (
    CW_REPOSITORY_SUITE_VERSION,
    CW_SQLITE_REPOSITORY_SUITE_ID,
    COLLABORATIVE_WORK_DOMAIN,
    COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
)
from intergrax.core.qualification import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
    ProviderQualificationSuiteIdentity,
    ProviderQualificationSuiteOutcome,
    QualificationEvidence,
    QualificationRunId,
    QualificationStatus,
    execute_provider_qualification,
)
from intergrax.core.qualification.execution import (
    ProviderQualificationExecutionConflictError,
    ProviderQualificationExecutionDependencies,
    ProviderQualificationExecutionRequest,
    ProviderQualificationMaterializationError,
    ProviderQualificationPersistenceExecutionError,
    ProviderQualificationResolutionError,
    ProviderQualificationSuiteInfrastructureError,
)
from intergrax.core.qualification.observability import (
    ProviderQualificationExecutionEventType,
    ProviderQualificationInfrastructurePhase,
    RecordingProviderQualificationExecutionObservability,
    build_qualification_infrastructure_problem_envelope,
    qualification_infrastructure_problem_event_id,
    qualification_infrastructure_source_layer,
)
from intergrax.core.qualification.persistence import (
    DocumentStoreProviderQualificationPersistence,
    ProviderQualificationPersistenceConflictError,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.observability.export_boundary import ExportRecordKind, ExportStatus
from intergrax.runtime.observability.problem_signal import (
    PLATFORM_PROBLEM_SIGNAL_SCHEMA,
    PROBLEM_SOURCE_LAYER_INTEGRATION,
    PROBLEM_SOURCE_LAYER_RUNTIME,
)

pytestmark = pytest.mark.unit

_EXECUTOR = ProviderQualificationExecutor(
    executor_kind="unit_test_runner",
    executor_id="qual-obs-test",
    executor_version="2026.09.02",
)
_SOURCE_REVISION = "unit-test-revision"
_FIXED_RUN_ID = QualificationRunId("qual_run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")


def _sqlite_subject() -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id="sqlite",
        provider_version="lab",
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        domain=COLLABORATIVE_WORK_DOMAIN,
        intergrax_revision=_SOURCE_REVISION,
        qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
        environment_id="unit-test",
        adapter_identity="intergrax.integrations.providers.relational_store.sqlite",
    )


def _sqlite_profile(tmp_path: Path) -> IntegrationProfile:
    return IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path / "sqlite-data")}},
    )


@pytest.fixture(autouse=True)
def _sqlite_catalog() -> None:
    clear_catalog()
    register_sqlite_integration()
    yield
    clear_catalog()


@dataclass(frozen=True, slots=True)
class _FakeHandle:
    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _FakeSuite:
    identity: ProviderQualificationSuiteIdentity
    outcome: ProviderQualificationSuiteOutcome
    execute_raises: Exception | None = None

    def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
        if self.execute_raises is not None:
            raise self.execute_raises
        return self.outcome


@dataclass(frozen=True, slots=True)
class _FakeBinding:
    suite: _FakeSuite
    materialize_raises: Exception | None = None

    def validate_resolved_provider(self, subject: object, *, resolved_provider_id: str) -> None:
        return None

    def materialize(
        self,
        profile: object,
        *,
        resolved_provider_id: str,
    ) -> tuple[object, _FakeHandle]:
        if self.materialize_raises is not None:
            raise self.materialize_raises
        return "capability", _FakeHandle()


def _suite_outcome(*, status: QualificationStatus = QualificationStatus.QUALIFIED) -> ProviderQualificationSuiteOutcome:
    return ProviderQualificationSuiteOutcome(
        status=status,
        result_summary=ProviderQualificationResultSummary(passed=1, failed=0, skipped=0),
        evidence=(
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="suite.passed",
            ),
        ),
        environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
        ),
        limitations=(),
    )


def _suite_identity() -> ProviderQualificationSuiteIdentity:
    return ProviderQualificationSuiteIdentity(
        domain=COLLABORATIVE_WORK_DOMAIN,
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
    )


def _request(tmp_path: Path) -> ProviderQualificationExecutionRequest:
    return ProviderQualificationExecutionRequest(
        subject=_sqlite_subject(),
        executor=_EXECUTOR,
        source_revision=_SOURCE_REVISION,
        integration_profile=_sqlite_profile(tmp_path),
        qualification_run_id=_FIXED_RUN_ID,
    )


def _deps(
    binding: _FakeBinding,
    *,
    observability: object,
    store: InMemoryDocumentStore | None = None,
) -> ProviderQualificationExecutionDependencies:
    return ProviderQualificationExecutionDependencies(
        persistence=DocumentStoreProviderQualificationPersistence(store or InMemoryDocumentStore()),
        domain_binding=binding,
        observability=observability,
    )


class _RaisingObservability:
    """Intentionally raises on every observability port method."""

    def record_execution_started(self, **kwargs: object) -> None:
        raise RuntimeError("observability started failed")

    def record_execution_completed(self, run: object, *, occurred_at: object) -> None:
        raise RuntimeError("observability completed failed")

    def record_execution_recovered(
        self,
        run: object,
        *,
        recovery_kind: str,
        occurred_at: object,
    ) -> None:
        raise RuntimeError("observability recovered failed")

    def record_infrastructure_failure(self, **kwargs: object) -> None:
        raise RuntimeError("observability infrastructure failed")


def test_started_observer_failure_does_not_block_execution(tmp_path: Path) -> None:
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))
    run = execute_provider_qualification(
        _request(tmp_path),
        _deps(binding, observability=_RaisingObservability()),
    )
    assert run.status is QualificationStatus.QUALIFIED


def test_completed_observer_failure_returns_persisted_run(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))

    class _CompletedFailingObservability(_RaisingObservability):
        def record_execution_started(self, **kwargs: object) -> None:
            return None

    run = execute_provider_qualification(
        _request(tmp_path),
        _deps(binding, observability=_CompletedFailingObservability(), store=store),
    )
    assert run.status is QualificationStatus.QUALIFIED
    loaded = DocumentStoreProviderQualificationPersistence(store).get_by_qualification_run_id(
        run.qualification_run_id,
    )
    assert loaded == run


def test_recovered_observer_failure_returns_existing_without_rerun(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))
    observability = _RaisingObservability()
    deps = _deps(binding, observability=observability, store=store)
    first = execute_provider_qualification(_request(tmp_path), deps)
    second = execute_provider_qualification(_request(tmp_path), deps)
    assert first == second


def test_infrastructure_failure_observer_failure_preserves_original_error(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()),
        materialize_raises=ProviderQualificationMaterializationError("materialization failed"),
    )
    with pytest.raises(ProviderQualificationMaterializationError, match="materialization failed"):
        execute_provider_qualification(
            _request(tmp_path),
            _deps(binding, observability=_RaisingObservability()),
        )


def test_observer_failure_does_not_mask_conflict_error(tmp_path: Path) -> None:
    subject = _sqlite_subject()
    stored = ProviderQualificationRun(
        qualification_run_id=_FIXED_RUN_ID,
        subject=subject,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        executed_at=datetime(2026, 9, 2, tzinfo=UTC),
        executor=_EXECUTOR,
        result_summary=ProviderQualificationResultSummary(passed=1, failed=0, skipped=0),
        evidence=(),
        reproducibility=None,
        limitations=(),
        source_revision=_SOURCE_REVISION,
        environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
        ),
    )

    class _ConflictPersistence:
        def __init__(self) -> None:
            self._lookup_count = 0

        def persist(self, run: ProviderQualificationRun) -> ProviderQualificationRun:
            raise ProviderQualificationPersistenceConflictError("conflict")

        def get_by_qualification_run_id(
            self,
            qualification_run_id: QualificationRunId | str,
        ) -> ProviderQualificationRun | None:
            if str(qualification_run_id) != str(_FIXED_RUN_ID):
                return None
            self._lookup_count += 1
            if self._lookup_count >= 2:
                return stored
            return None

    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=_suite_identity(),
            outcome=_suite_outcome(status=QualificationStatus.REJECTED),
        ),
    )
    with pytest.raises(ProviderQualificationExecutionConflictError):
        execute_provider_qualification(
            _request(tmp_path),
            ProviderQualificationExecutionDependencies(
                persistence=_ConflictPersistence(),
                domain_binding=binding,
                observability=_RaisingObservability(),
            ),
        )


def test_infrastructure_problem_event_ids_are_unique_per_phase() -> None:
    resolution_id = qualification_infrastructure_problem_event_id(
        qualification_run_id=_FIXED_RUN_ID,
        phase=ProviderQualificationInfrastructurePhase.RESOLUTION,
        error_code="provider_resolution_failed",
    )
    materialization_id = qualification_infrastructure_problem_event_id(
        qualification_run_id=_FIXED_RUN_ID,
        phase=ProviderQualificationInfrastructurePhase.MATERIALIZATION,
        error_code="provider_materialization_failed",
    )
    assert resolution_id != materialization_id


def test_infrastructure_problem_envelope_uses_canonical_schema_id() -> None:
    envelope = build_qualification_infrastructure_problem_envelope(
        qualification_run_id=_FIXED_RUN_ID,
        subject=_sqlite_subject(),
        executor=_EXECUTOR,
        source_revision=_SOURCE_REVISION,
        phase=ProviderQualificationInfrastructurePhase.RESOLUTION,
        error_type="ProviderQualificationResolutionError",
        error_code="provider_resolution_failed",
    )
    assert envelope.schema_id == PLATFORM_PROBLEM_SIGNAL_SCHEMA
    assert envelope.source_schema_id == PLATFORM_PROBLEM_SIGNAL_SCHEMA
    assert envelope.record_kind is ExportRecordKind.PROBLEM_SIGNAL


def test_infrastructure_source_layer_mapping() -> None:
    assert (
        qualification_infrastructure_source_layer(
            ProviderQualificationInfrastructurePhase.RESOLUTION,
        )
        is PROBLEM_SOURCE_LAYER_INTEGRATION
    )
    assert (
        qualification_infrastructure_source_layer(
            ProviderQualificationInfrastructurePhase.PERSISTENCE,
        )
        is PROBLEM_SOURCE_LAYER_RUNTIME
    )


def test_platform_problem_signal_compatible_envelope_for_host_diagnostics() -> None:
    """Classification B: core emits canonical PROBLEM_SIGNAL; host wires central diagnostics."""
    envelope = build_qualification_infrastructure_problem_envelope(
        qualification_run_id=_FIXED_RUN_ID,
        subject=_sqlite_subject(),
        executor=_EXECUTOR,
        source_revision=_SOURCE_REVISION,
        phase=ProviderQualificationInfrastructurePhase.SUITE,
        error_type="ProviderQualificationSuiteInfrastructureError",
        error_code="suite_infrastructure_failed",
    )
    assert envelope.problem_kind
    assert envelope.problem_error_code == "suite_infrastructure_failed"
    assert envelope.correlation_id == str(_FIXED_RUN_ID)


def test_observability_emits_started_and_completed(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))
    execute_provider_qualification(_request(tmp_path), _deps(binding, observability=observability))
    types = {envelope.event_type for envelope in observability.envelopes}
    assert ProviderQualificationExecutionEventType.STARTED.value in types
    assert ProviderQualificationExecutionEventType.COMPLETED.value in types


def test_successful_qualification_completed_status_succeeded(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))
    execute_provider_qualification(_request(tmp_path), _deps(binding, observability=observability))
    completed = [
        envelope
        for envelope in observability.envelopes
        if envelope.event_type == ProviderQualificationExecutionEventType.COMPLETED.value
    ]
    assert len(completed) == 1
    assert completed[0].status is ExportStatus.SUCCEEDED


def test_semantic_rejection_completed_failed_without_problem_signal(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=_suite_identity(),
            outcome=_suite_outcome(status=QualificationStatus.REJECTED),
        ),
    )
    execute_provider_qualification(_request(tmp_path), _deps(binding, observability=observability))
    completed = [
        envelope
        for envelope in observability.envelopes
        if envelope.event_type == ProviderQualificationExecutionEventType.COMPLETED.value
    ]
    problem_signals = [
        envelope
        for envelope in observability.envelopes
        if envelope.record_kind is ExportRecordKind.PROBLEM_SIGNAL
    ]
    assert len(completed) == 1
    assert completed[0].status is ExportStatus.FAILED
    assert problem_signals == []


@pytest.mark.parametrize(
    ("phase", "patch_target", "error_type", "error_code"),
    [
        (
            ProviderQualificationInfrastructurePhase.RESOLUTION,
            "intergrax.core.qualification.execution.resolve_integration_provider_id",
            ProviderQualificationResolutionError,
            "resolution",
        ),
        (
            ProviderQualificationInfrastructurePhase.MATERIALIZATION,
            None,
            ProviderQualificationMaterializationError,
            "materialization",
        ),
        (
            ProviderQualificationInfrastructurePhase.SUITE,
            None,
            ProviderQualificationSuiteInfrastructureError,
            "suite",
        ),
    ],
)
def test_infrastructure_phases_emit_problem_signal(
    tmp_path: Path,
    phase: ProviderQualificationInfrastructurePhase,
    patch_target: str | None,
    error_type: type[Exception],
    error_code: str,
) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    if phase is ProviderQualificationInfrastructurePhase.MATERIALIZATION:
        binding = _FakeBinding(
            suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()),
            materialize_raises=error_type("failed"),
        )
        with pytest.raises(error_type):
            execute_provider_qualification(
                _request(tmp_path),
                _deps(binding, observability=observability),
            )
    elif phase is ProviderQualificationInfrastructurePhase.SUITE:
        binding = _FakeBinding(
            suite=_FakeSuite(
                identity=_suite_identity(),
                outcome=_suite_outcome(),
                execute_raises=RuntimeError("suite failed"),
            ),
        )
        with pytest.raises(ProviderQualificationSuiteInfrastructureError):
            execute_provider_qualification(
                _request(tmp_path),
                _deps(binding, observability=observability),
            )
    else:
        binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))
        with patch(patch_target, side_effect=error_type("failed")):
            with pytest.raises(error_type):
                execute_provider_qualification(
                    _request(tmp_path),
                    _deps(binding, observability=observability),
                )

    assert any(
        envelope.record_kind is ExportRecordKind.PROBLEM_SIGNAL
        and error_code in envelope.event_type
        for envelope in observability.envelopes
    )


def test_persistence_failure_emits_problem_signal(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))

    class _FailingPersistence:
        def persist(self, run: ProviderQualificationRun) -> ProviderQualificationRun:
            raise RuntimeError("persist failed")

        def get_by_qualification_run_id(
            self,
            qualification_run_id: QualificationRunId | str,
        ) -> ProviderQualificationRun | None:
            return None

    with pytest.raises(ProviderQualificationPersistenceExecutionError):
        execute_provider_qualification(
            _request(tmp_path),
            ProviderQualificationExecutionDependencies(
                persistence=_FailingPersistence(),
                domain_binding=binding,
                observability=observability,
            ),
        )
    assert any(
        envelope.record_kind is ExportRecordKind.PROBLEM_SIGNAL
        and "persistence" in envelope.event_type
        for envelope in observability.envelopes
    )


def test_recovered_without_false_completed_on_idempotent_return(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(suite=_FakeSuite(identity=_suite_identity(), outcome=_suite_outcome()))
    deps = _deps(binding, observability=observability, store=store)
    execute_provider_qualification(_request(tmp_path), deps)
    execute_provider_qualification(_request(tmp_path), deps)
    recovered = [
        envelope
        for envelope in observability.envelopes
        if envelope.event_type == ProviderQualificationExecutionEventType.RECOVERED.value
    ]
    completed = [
        envelope
        for envelope in observability.envelopes
        if envelope.event_type == ProviderQualificationExecutionEventType.COMPLETED.value
    ]
    assert len(recovered) == 1
    assert len(completed) == 1
