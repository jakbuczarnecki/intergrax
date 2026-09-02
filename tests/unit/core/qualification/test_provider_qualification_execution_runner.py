# © Artur Czarnecki. All rights reserved.

"""Shared provider qualification execution runner tests (PROVIDER-QUAL-7)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.repository_qualification_suite import (
    CW_REPOSITORY_SUITE_VERSION,
    CW_SQLITE_REPOSITORY_SUITE_ID,
    COLLABORATIVE_WORK_DOMAIN,
    COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
    collaborative_work_sqlite_repository_qualification_binding,
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
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationStatus,
    ValidityEvaluationId,
    execute_provider_qualification,
)
from intergrax.core.qualification.execution import (
    ProviderQualificationExecutionConflictError,
    ProviderQualificationExecutionDependencies,
    ProviderQualificationExecutionRequest,
    ProviderQualificationMaterializationError,
    ProviderQualificationRequestIncompatibleError,
    ProviderQualificationResolutionError,
    ProviderQualificationRunIdentityError,
    ProviderQualificationSubjectMismatchError,
    ProviderQualificationSuiteIdentityMismatchError,
    ProviderQualificationSuiteInfrastructureError,
    causality_from_requalification_identity,
)
from intergrax.core.qualification.observability import (
    ProviderQualificationExecutionEventType,
    RecordingProviderQualificationExecutionObservability,
)
from intergrax.core.qualification.persistence import (
    DocumentStoreProviderQualificationPersistence,
    ProviderQualificationPersistenceConflictError,
)
from intergrax.core.qualification.requalification import (
    ProviderRequalificationDecision,
    prepare_provider_requalification_run_identity,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.observability.export_boundary import ExportRecordKind
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_EXECUTOR = ProviderQualificationExecutor(
    executor_kind="unit_test_runner",
    executor_id="qual-runner-test",
    executor_version="2026.09.02",
)
_SOURCE_REVISION = "unit-test-revision"
_FIXED_RUN_ID = QualificationRunId("qual_run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


def _sqlite_subject(
    *,
    provider_id: str = "sqlite",
    suite_id: str = CW_SQLITE_REPOSITORY_SUITE_ID,
) -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id=provider_id,
        provider_version="lab",
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        domain=COLLABORATIVE_WORK_DOMAIN,
        intergrax_revision=_SOURCE_REVISION,
        qualification_suite_id=suite_id,
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
    closed: bool = False

    def close(self) -> None:
        object.__setattr__(self, "closed", True)


@dataclass(frozen=True, slots=True)
class _FakeBinding:
    suite: _FakeSuite
    materialize_raises: Exception | None = None
    validate_raises: Exception | None = None
    capability: object = "fake-capability"

    def validate_resolved_provider(self, subject: object, *, resolved_provider_id: str) -> None:
        if self.validate_raises is not None:
            raise self.validate_raises

    def materialize(
        self,
        profile: object,
        *,
        resolved_provider_id: str,
    ) -> tuple[object, _FakeHandle]:
        if self.materialize_raises is not None:
            raise self.materialize_raises
        return self.capability, _FakeHandle()


@dataclass(frozen=True, slots=True)
class _FakeSuite:
    identity: ProviderQualificationSuiteIdentity
    outcome: ProviderQualificationSuiteOutcome
    execute_raises: Exception | None = None

    def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
        if self.execute_raises is not None:
            raise self.execute_raises
        return self.outcome


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


def _request(
    *,
    subject: ProviderQualificationSubject | None = None,
    profile: IntegrationProfile | None = None,
    run_id: QualificationRunId | None = _FIXED_RUN_ID,
    tmp_path: Path | None = None,
) -> ProviderQualificationExecutionRequest:
    resolved_profile = profile
    if resolved_profile is None:
        assert tmp_path is not None
        resolved_profile = _sqlite_profile(tmp_path)
    return ProviderQualificationExecutionRequest(
        subject=subject or _sqlite_subject(),
        executor=_EXECUTOR,
        source_revision=_SOURCE_REVISION,
        integration_profile=resolved_profile,
        qualification_run_id=run_id,
    )


def _dependencies(
    binding: _FakeBinding,
    store: InMemoryDocumentStore | None = None,
    *,
    observability: RecordingProviderQualificationExecutionObservability | None = None,
) -> ProviderQualificationExecutionDependencies:
    document_store = store or InMemoryDocumentStore()
    return ProviderQualificationExecutionDependencies(
        persistence=DocumentStoreProviderQualificationPersistence(document_store),
        domain_binding=binding,
        observability=observability or RecordingProviderQualificationExecutionObservability(),
    )


def test_valid_request_executes_registered_typed_suite(tmp_path: Path) -> None:
    outcome = _suite_outcome()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=outcome,
        ),
    )
    run = execute_provider_qualification(
        _request(tmp_path=tmp_path),
        _dependencies(binding),
    )
    assert run.qualification_run_id == _FIXED_RUN_ID
    assert run.status is QualificationStatus.QUALIFIED
    assert run.result_summary.passed == 1


def test_returned_run_uses_requested_qualification_run_id(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    run = execute_provider_qualification(
        _request(tmp_path=tmp_path, run_id=_FIXED_RUN_ID),
        _dependencies(binding),
    )
    assert run.qualification_run_id == _FIXED_RUN_ID


def test_requalification_prepared_run_id_is_preserved(tmp_path: Path) -> None:
    prior_run_id = QualificationRunId("qual_run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    decision = ProviderRequalificationDecision(
        qualification_run_id=prior_run_id,
        required=True,
        reason="stale",
        based_on_validity=QualificationEvidenceValidity.STALE,
        basis_validity_evaluation_id=ValidityEvaluationId(
            "valid_eval_cccccccccccccccccccccccccccccc",
        ),
        prior_run_remains_terminal=False,
        decided_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    identity = prepare_provider_requalification_run_identity(decision)
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    run = execute_provider_qualification(
        ProviderQualificationExecutionRequest(
            subject=_sqlite_subject(),
            executor=_EXECUTOR,
            source_revision=_SOURCE_REVISION,
            integration_profile=_sqlite_profile(tmp_path),
            requalification_identity=identity,
            causality=causality_from_requalification_identity(identity),
        ),
        _dependencies(binding),
    )
    assert run.qualification_run_id == identity.new_qualification_run_id


def test_runner_does_not_invoke_validity_evaluation_layer(tmp_path: Path) -> None:
    with patch(
        "intergrax.core.qualification.validity_evaluation.evaluate_provider_qualification_validity",
    ) as mocked:
        binding = _FakeBinding(
            suite=_FakeSuite(
                identity=ProviderQualificationSuiteIdentity(
                    domain=COLLABORATIVE_WORK_DOMAIN,
                    capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                    qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                    qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
                ),
                outcome=_suite_outcome(),
            ),
        )
        execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))
        mocked.assert_not_called()


def test_provider_resolution_failure_is_explicit(tmp_path: Path) -> None:
    empty_profile = IntegrationProfile()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    with pytest.raises(ProviderQualificationResolutionError):
        execute_provider_qualification(
            _request(profile=empty_profile, tmp_path=tmp_path),
            _dependencies(binding),
        )


def test_materialization_failure_is_explicit(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
        materialize_raises=RuntimeError("backend unavailable"),
    )
    with pytest.raises(ProviderQualificationMaterializationError):
        execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))


def test_suite_infrastructure_error_is_not_provider_rejection(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
            execute_raises=RuntimeError("suite host failed"),
        ),
    )
    with pytest.raises(ProviderQualificationSuiteInfrastructureError):
        execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))


def test_suite_semantic_failure_maps_to_rejected_status(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(status=QualificationStatus.REJECTED),
        ),
    )
    run = execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))
    assert run.status is QualificationStatus.REJECTED


def test_subject_provider_mismatch_fails_closed(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
        validate_raises=ProviderQualificationSubjectMismatchError("mismatch"),
    )
    with pytest.raises(ProviderQualificationSubjectMismatchError):
        execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))


def test_suite_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id="other.suite.v1",
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    with pytest.raises(ProviderQualificationSuiteIdentityMismatchError):
        execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))


def test_no_provider_fallback_on_resolution_failure(tmp_path: Path) -> None:
    profile = IntegrationProfile(
        relational_store=IntegrationBinding.from_slug("sqlite"),
        options={SQLITE: {"data_dir": str(tmp_path / "sqlite-data")}},
    )
    binding = collaborative_work_sqlite_repository_qualification_binding()
    with patch(
        "intergrax.core.qualification.execution.resolve_integration_provider_id",
        side_effect=ProviderQualificationResolutionError("unresolved"),
    ):
        with pytest.raises(ProviderQualificationResolutionError):
            execute_provider_qualification(
                _request(profile=profile, tmp_path=tmp_path),
                ProviderQualificationExecutionDependencies(
                    persistence=DocumentStoreProviderQualificationPersistence(
                        InMemoryDocumentStore(),
                    ),
                    domain_binding=binding,
                ),
            )


def test_provider_qualification_run_is_immutable_after_construction(tmp_path: Path) -> None:
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    run = execute_provider_qualification(_request(tmp_path=tmp_path), _dependencies(binding))
    with pytest.raises(AttributeError):
        run.status = QualificationStatus.REJECTED  # type: ignore[misc]


def test_idempotent_execution_returns_existing_persisted_run(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    deps = _dependencies(binding, store=store, observability=observability)
    first = execute_provider_qualification(_request(tmp_path=tmp_path), deps)
    second = execute_provider_qualification(_request(tmp_path=tmp_path), deps)
    assert first == second
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


def test_incompatible_existing_run_subject_fails_closed(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    subject = _sqlite_subject()
    existing = ProviderQualificationRun(
        qualification_run_id=_FIXED_RUN_ID,
        subject=subject,
        status=QualificationStatus.QUALIFIED,
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
    persistence.persist(existing)

    incompatible_subject = replace(subject, provider_version="different-version")
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(status=QualificationStatus.REJECTED),
        ),
    )
    with pytest.raises(ProviderQualificationRequestIncompatibleError):
        execute_provider_qualification(
            _request(subject=incompatible_subject, tmp_path=tmp_path),
            ProviderQualificationExecutionDependencies(
                persistence=persistence,
                domain_binding=binding,
            ),
        )


def test_persisted_conflict_with_different_fact_fails_closed(tmp_path: Path) -> None:
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
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(status=QualificationStatus.REJECTED),
        ),
    )
    observability = RecordingProviderQualificationExecutionObservability()
    with pytest.raises(ProviderQualificationExecutionConflictError):
        execute_provider_qualification(
            _request(tmp_path=tmp_path),
            ProviderQualificationExecutionDependencies(
                persistence=_ConflictPersistence(),
                domain_binding=binding,
                observability=observability,
            ),
        )
    assert any(
        envelope.record_kind is ExportRecordKind.PROBLEM_SIGNAL
        for envelope in observability.envelopes
    )


def test_observability_records_start_and_completion(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    execute_provider_qualification(
        _request(tmp_path=tmp_path),
        _dependencies(binding, observability=observability),
    )
    event_types = {envelope.event_type for envelope in observability.envelopes}
    assert ProviderQualificationExecutionEventType.STARTED.value in event_types
    assert ProviderQualificationExecutionEventType.COMPLETED.value in event_types


def test_observability_records_semantic_rejection_as_completed(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(status=QualificationStatus.REJECTED),
        ),
    )
    run = execute_provider_qualification(
        _request(tmp_path=tmp_path),
        _dependencies(binding, observability=observability),
    )
    assert run.status is QualificationStatus.REJECTED
    completed = [
        envelope
        for envelope in observability.envelopes
        if envelope.event_type == ProviderQualificationExecutionEventType.COMPLETED.value
    ]
    assert len(completed) == 1


def test_observability_records_resolution_failure(tmp_path: Path) -> None:
    observability = RecordingProviderQualificationExecutionObservability()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
        ),
    )
    with patch(
        "intergrax.core.qualification.execution.resolve_integration_provider_id",
        side_effect=ProviderQualificationResolutionError("unresolved"),
    ):
        with pytest.raises(ProviderQualificationResolutionError):
            execute_provider_qualification(
                _request(tmp_path=tmp_path),
                _dependencies(binding, observability=observability),
            )
    assert any(
        envelope.record_kind is ExportRecordKind.PROBLEM_SIGNAL
        and "resolution" in envelope.event_type
        for envelope in observability.envelopes
    )


def test_sqlite_provider_execution(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    binding = collaborative_work_sqlite_repository_qualification_binding()
    run = execute_provider_qualification(
        _request(tmp_path=tmp_path),
        ProviderQualificationExecutionDependencies(
            persistence=DocumentStoreProviderQualificationPersistence(store),
            domain_binding=binding,
        ),
    )
    assert run.status is QualificationStatus.QUALIFIED
    assert run.environment_metadata.real_backend is True
    assert run.environment_metadata.mocks is False
    assert run.environment_metadata.sqlite_substitution is False
    loaded = DocumentStoreProviderQualificationPersistence(store).get_by_qualification_run_id(
        run.qualification_run_id,
    )
    assert loaded == run


def test_synthetic_third_provider_requires_no_core_vendor_dispatch_changes(tmp_path: Path) -> None:
    @dataclass(frozen=True, slots=True)
    class _SyntheticSuite:
        @property
        def identity(self) -> ProviderQualificationSuiteIdentity:
            return ProviderQualificationSuiteIdentity(
                domain="synthetic_domain",
                capability_id="synthetic.capability.v1",
                qualification_suite_id="synthetic.suite.v1",
                qualification_suite_version="1.0.0",
            )

        def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
            assert capability == "synthetic-capability"
            return ProviderQualificationSuiteOutcome(
                status=QualificationStatus.NOT_QUALIFIED,
                result_summary=ProviderQualificationResultSummary(passed=0, failed=0, skipped=1),
                evidence=(),
                environment_metadata=ProviderQualificationEnvironmentMetadata(
                    real_backend=False,
                    mocks=True,
                    sqlite_substitution=False,
                ),
                limitations=("synthetic extension proof only",),
            )

    @dataclass(frozen=True, slots=True)
    class _SyntheticBinding:
        @property
        def suite(self) -> _SyntheticSuite:
            return _SyntheticSuite()

        def validate_resolved_provider(self, subject: object, *, resolved_provider_id: str) -> None:
            if not isinstance(subject, ProviderQualificationSubject):
                raise TypeError("subject must be ProviderQualificationSubject")
            if subject.provider_id != resolved_provider_id:
                raise ProviderQualificationSubjectMismatchError("provider mismatch")

        def materialize(
            self,
            profile: object,
            *,
            resolved_provider_id: str,
        ) -> tuple[object, _FakeHandle]:
            return "synthetic-capability", _FakeHandle()

    subject = ProviderQualificationSubject(
        provider_id="synthetic_vendor",
        provider_version="0.1.0",
        capability_id="synthetic.capability.v1",
        domain="synthetic_domain",
        intergrax_revision=_SOURCE_REVISION,
        qualification_suite_id="synthetic.suite.v1",
        qualification_suite_version="1.0.0",
        environment_id="unit-test",
    )
    profile = IntegrationProfile()
    with patch(
        "intergrax.core.qualification.execution.resolve_integration_provider_id",
        return_value="synthetic_vendor",
    ):
        run = execute_provider_qualification(
            ProviderQualificationExecutionRequest(
                subject=subject,
                executor=_EXECUTOR,
                source_revision=_SOURCE_REVISION,
                integration_profile=profile,
                qualification_run_id=_FIXED_RUN_ID,
            ),
            ProviderQualificationExecutionDependencies(
                persistence=DocumentStoreProviderQualificationPersistence(
                    InMemoryDocumentStore(),
                ),
                domain_binding=_SyntheticBinding(),
            ),
        )
    assert run.status is QualificationStatus.NOT_QUALIFIED


def test_requalification_run_id_mismatch_raises(tmp_path: Path) -> None:
    prior_run_id = QualificationRunId("qual_run_dddddddddddddddddddddddddddddddd")
    decision = ProviderRequalificationDecision(
        qualification_run_id=prior_run_id,
        required=True,
        reason="stale",
        based_on_validity=QualificationEvidenceValidity.STALE,
        basis_validity_evaluation_id=ValidityEvaluationId(
            "valid_eval_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        ),
        prior_run_remains_terminal=False,
        decided_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    identity = prepare_provider_requalification_run_identity(decision)
    with pytest.raises(ProviderQualificationRunIdentityError):
        execute_provider_qualification(
            ProviderQualificationExecutionRequest(
                subject=_sqlite_subject(),
                executor=_EXECUTOR,
                source_revision=_SOURCE_REVISION,
                integration_profile=_sqlite_profile(tmp_path),
                qualification_run_id=_FIXED_RUN_ID,
                requalification_identity=identity,
            ),
            _dependencies(
                _FakeBinding(
                    suite=_FakeSuite(
                        identity=ProviderQualificationSuiteIdentity(
                            domain=COLLABORATIVE_WORK_DOMAIN,
                            capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                            qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                            qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
                        ),
                        outcome=_suite_outcome(),
                    ),
                ),
            ),
        )


def test_materialization_handle_is_closed_on_suite_failure(tmp_path: Path) -> None:
    handle = _FakeHandle()
    binding = _FakeBinding(
        suite=_FakeSuite(
            identity=ProviderQualificationSuiteIdentity(
                domain=COLLABORATIVE_WORK_DOMAIN,
                capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
                qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
                qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
            ),
            outcome=_suite_outcome(),
            execute_raises=RuntimeError("suite failed"),
        ),
        capability="cap",
    )
    binding = replace(binding, capability="cap")

    @dataclass(frozen=True, slots=True)
    class _HandleBinding:
        suite: _FakeSuite = binding.suite
        _handle: _FakeHandle = handle

        def validate_resolved_provider(self, subject: object, *, resolved_provider_id: str) -> None:
            return None

        def materialize(
            self,
            profile: object,
            *,
            resolved_provider_id: str,
        ) -> tuple[object, _FakeHandle]:
            return "cap", self._handle

    with pytest.raises(ProviderQualificationSuiteInfrastructureError):
        execute_provider_qualification(
            _request(tmp_path=tmp_path),
            ProviderQualificationExecutionDependencies(
                persistence=DocumentStoreProviderQualificationPersistence(
                    InMemoryDocumentStore(),
                ),
                domain_binding=_HandleBinding(),
            ),
        )
    assert handle.closed is True
