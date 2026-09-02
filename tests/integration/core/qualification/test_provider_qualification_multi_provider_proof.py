# © Artur Czarnecki. All rights reserved.

"""Multi-provider real qualification proof (PROVIDER-QUAL-8)."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.collaborative_work.repository_qualification_suite import (
    CW_REPOSITORY_SUITE_VERSION,
    CW_SQLITE_REPOSITORY_SUITE_ID,
    COLLABORATIVE_WORK_DOMAIN,
    COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
    CollaborativeWorkRepositoryQualificationSuite,
    collaborative_work_postgresql_repository_qualification_suite,
    collaborative_work_sqlite_repository_qualification_binding,
    collaborative_work_sqlite_repository_qualification_suite,
)
from intergrax.core.qualification import (
    ProviderQualificationExecutor,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationStatus,
    QualificationValidityRecord,
    ValidityEvaluationId,
    execute_provider_qualification,
    new_qualification_run_id,
)
from intergrax.core.qualification.execution import (
    ProviderQualificationExecutionDependencies,
    ProviderQualificationExecutionRequest,
    ProviderQualificationMaterializationError,
    ProviderQualificationSubjectMismatchError,
    causality_from_requalification_identity,
)
from intergrax.core.qualification.persistence import DocumentStoreProviderQualificationPersistence
from intergrax.core.qualification.provider import ProviderQualificationSubject
from intergrax.core.qualification.requalification import (
    establish_provider_requalification_requirement,
    prepare_provider_requalification_run_identity,
)
from intergrax.core.qualification.validity_evaluation import (
    record_provider_qualification_validity_revocation,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.integration]

_EXECUTOR = ProviderQualificationExecutor(
    executor_kind="integration_test_runner",
    executor_id="provider-qual-8-multi",
    executor_version="2026.09.02",
)
_SOURCE_REVISION = "provider-qual-8-multi-provider-proof"
_STALE_EVAL_ID = ValidityEvaluationId("valid_eval_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
_REVOKED_EVAL_ID = ValidityEvaluationId("valid_eval_cccccccccccccccccccccccccccccccc")
_DECIDED_AT = datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)


def _sqlite_subject() -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id="sqlite",
        provider_version="execution-config-lab",
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        domain=COLLABORATIVE_WORK_DOMAIN,
        intergrax_revision=_SOURCE_REVISION,
        qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
        environment_id="local-sqlite-qual-host",
        adapter_identity="intergrax.integrations.providers.relational_store.sqlite",
    )


def _sqlite_profile(tmp_path: Path, *, isolate: str = "") -> IntegrationProfile:
    suffix = f"-{isolate}" if isolate else ""
    return IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path / f"sqlite-data{suffix}")}},
    )


def _sqlite_dependencies(
    store: InMemoryDocumentStore | None = None,
) -> ProviderQualificationExecutionDependencies:
    return ProviderQualificationExecutionDependencies(
        persistence=DocumentStoreProviderQualificationPersistence(
            store or InMemoryDocumentStore(),
        ),
        domain_binding=collaborative_work_sqlite_repository_qualification_binding(),
    )


def _execute_sqlite(
    tmp_path: Path,
    *,
    run_id: QualificationRunId | None = None,
    requalification_identity: object | None = None,
    causality: object | None = None,
    store: InMemoryDocumentStore | None = None,
    profile_isolate: str = "",
):
    request = ProviderQualificationExecutionRequest(
        subject=_sqlite_subject(),
        executor=_EXECUTOR,
        source_revision=_SOURCE_REVISION,
        integration_profile=_sqlite_profile(tmp_path, isolate=profile_isolate),
        qualification_run_id=run_id,
        requalification_identity=requalification_identity,  # type: ignore[arg-type]
        causality=causality,  # type: ignore[arg-type]
    )
    return execute_provider_qualification(request, _sqlite_dependencies(store=store))


@pytest.fixture(autouse=True)
def _sqlite_catalog() -> Generator[None, None, None]:
    clear_catalog()
    register_sqlite_integration()
    yield
    clear_catalog()


def test_sqlite_real_provider_qualification_execution_and_persistence(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    run_id = new_qualification_run_id()
    run = _execute_sqlite(tmp_path, run_id=run_id, store=store)

    assert run.qualification_run_id == run_id
    assert run.status is QualificationStatus.QUALIFIED
    assert run.subject.provider_id == "sqlite"
    assert run.environment_metadata.real_backend is True
    assert run.environment_metadata.mocks is False
    assert run.environment_metadata.sqlite_substitution is False

    loaded = DocumentStoreProviderQualificationPersistence(store).get_by_qualification_run_id(run_id)
    assert loaded == run
    assert loaded is not None
    assert loaded.evidence == run.evidence


def test_postgresql_and_sqlite_share_semantic_suite_implementation() -> None:
    pg_suite = collaborative_work_postgresql_repository_qualification_suite()
    sqlite_suite = collaborative_work_sqlite_repository_qualification_suite()
    assert isinstance(pg_suite, CollaborativeWorkRepositoryQualificationSuite)
    assert isinstance(sqlite_suite, CollaborativeWorkRepositoryQualificationSuite)
    assert type(pg_suite) is type(sqlite_suite)
    assert pg_suite.identity.qualification_suite_id != sqlite_suite.identity.qualification_suite_id
    assert pg_suite.identity.capability_id == sqlite_suite.identity.capability_id


def test_sqlite_requalification_composes_with_shared_runner(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    prior_run_id = new_qualification_run_id()
    prior_run = _execute_sqlite(tmp_path, run_id=prior_run_id, store=store)
    prior_snapshot = (
        prior_run.qualification_run_id,
        prior_run.status,
        prior_run.subject,
        prior_run.executed_at,
    )

    stale_record = QualificationValidityRecord(
        qualification_run_id=prior_run_id,
        validity_evaluation_id=_STALE_EVAL_ID,
        validity=QualificationEvidenceValidity.STALE,
        evaluated_at=_DECIDED_AT,
        reason="qualification_suite_version_changed",
    )
    decision = establish_provider_requalification_requirement(
        prior_run_id,
        (stale_record,),
        decided_at=_DECIDED_AT,
    )
    identity = prepare_provider_requalification_run_identity(decision)

    new_run = _execute_sqlite(
        tmp_path,
        requalification_identity=identity,
        causality=causality_from_requalification_identity(identity),
        store=store,
        profile_isolate="requal-stale",
    )

    assert new_run.qualification_run_id == identity.new_qualification_run_id
    assert new_run.qualification_run_id != prior_run_id
    reloaded_prior = persistence.get_by_qualification_run_id(prior_run_id)
    assert reloaded_prior is not None
    assert (
        reloaded_prior.qualification_run_id,
        reloaded_prior.status,
        reloaded_prior.subject,
        reloaded_prior.executed_at,
    ) == prior_snapshot


def test_revoked_prior_run_remains_terminal_new_qualification_succeeds(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    prior_run_id = new_qualification_run_id()
    prior_run = _execute_sqlite(tmp_path, run_id=prior_run_id, store=store)

    revoked_record = record_provider_qualification_validity_revocation(
        prior_run_id,
        reason="manual_revocation",
        evaluated_at=_DECIDED_AT,
        validity_evaluation_id=_REVOKED_EVAL_ID,
    )
    decision = establish_provider_requalification_requirement(
        prior_run_id,
        (revoked_record,),
        decided_at=_DECIDED_AT,
    )
    assert decision.prior_run_remains_terminal is True
    identity = prepare_provider_requalification_run_identity(decision)

    new_run = _execute_sqlite(
        tmp_path,
        requalification_identity=identity,
        causality=causality_from_requalification_identity(identity),
        store=store,
        profile_isolate="requal-revoked",
    )
    assert new_run.status is QualificationStatus.QUALIFIED
    reloaded_prior = persistence.get_by_qualification_run_id(prior_run_id)
    assert reloaded_prior == prior_run
    assert reloaded_prior is not None
    assert reloaded_prior.qualification_run_id == prior_run_id


def test_sqlite_idempotent_recovery_no_conflicting_fact(tmp_path: Path) -> None:
    store = InMemoryDocumentStore()
    deps = _sqlite_dependencies(store=store)
    run_id = new_qualification_run_id()
    request = ProviderQualificationExecutionRequest(
        subject=_sqlite_subject(),
        executor=_EXECUTOR,
        source_revision=_SOURCE_REVISION,
        integration_profile=_sqlite_profile(tmp_path),
        qualification_run_id=run_id,
    )
    first = execute_provider_qualification(request, deps)
    second = execute_provider_qualification(request, deps)
    assert first == second
    assert first.qualification_run_id == run_id


def test_wrong_provider_id_fails_closed(tmp_path: Path) -> None:
    subject = ProviderQualificationSubject(
        provider_id="postgresql",
        provider_version="execution-config",
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        domain=COLLABORATIVE_WORK_DOMAIN,
        intergrax_revision=_SOURCE_REVISION,
        qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
        environment_id="local-sqlite-qual-host",
        adapter_identity="intergrax.integrations.providers.relational_store.sqlite",
    )
    with pytest.raises(ProviderQualificationSubjectMismatchError):
        execute_provider_qualification(
            ProviderQualificationExecutionRequest(
                subject=subject,
                executor=_EXECUTOR,
                source_revision=_SOURCE_REVISION,
                integration_profile=_sqlite_profile(tmp_path),
                qualification_run_id=new_qualification_run_id(),
            ),
            _sqlite_dependencies(),
        )


def test_bad_config_materialization_failure_explicit(tmp_path: Path) -> None:
    blocked_path = tmp_path / "blocked.sqlite"
    blocked_path.write_text("not-a-directory", encoding="utf-8")
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(blocked_path)}},
    )
    with pytest.raises(ProviderQualificationMaterializationError):
        execute_provider_qualification(
            ProviderQualificationExecutionRequest(
                subject=_sqlite_subject(),
                executor=_EXECUTOR,
                source_revision=_SOURCE_REVISION,
                integration_profile=profile,
                qualification_run_id=new_qualification_run_id(),
            ),
            _sqlite_dependencies(),
        )
