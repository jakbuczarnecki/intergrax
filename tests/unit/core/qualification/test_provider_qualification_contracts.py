# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from intergrax.core.qualification import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
    QualificationEvidence,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationStatus,
    QualificationValidityInterpretation,
    QualificationValidityRecord,
    new_qualification_run_id,
    new_validity_evaluation_id,
    validate_qualification_run_id,
)
from intergrax.core.plugins.platform_qualification import PluginQualificationEvidenceKind

pytestmark = pytest.mark.unit

_EXECUTED_AT = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_T1 = datetime(2026, 8, 17, 13, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_T2 = datetime(2026, 8, 18, 9, 0, 0, tzinfo=timezone.utc)


def _subject(*, provider_id: str = "postgresql", provider_version: str = "16.6") -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id=provider_id,
        provider_version=provider_version,
        capability_id="collaborative_work.persistence.v1",
        domain="collaborative_work",
        intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        qualification_suite_id="cw.postgresql.repository.v1",
        qualification_suite_version="1.0.0",
        environment_id="local-docker-qual-host",
        adapter_identity="intergrax.integrations.providers.relational_store.postgresql",
    )


def _executor(*, executor_kind: str = "local_cli") -> ProviderQualificationExecutor:
    return ProviderQualificationExecutor(
        executor_kind=executor_kind,
        executor_id="qual-host-01",
        executor_version="2026.08.17",
    )


def _environment_metadata() -> ProviderQualificationEnvironmentMetadata:
    return ProviderQualificationEnvironmentMetadata(
        real_backend=True,
        mocks=False,
        sqlite_substitution=False,
        bounded_environment="docker-postgres-16",
    )


def _result_summary() -> ProviderQualificationResultSummary:
    return ProviderQualificationResultSummary(
        passed=42,
        failed=0,
        skipped=3,
        label="cw.postgresql.repository.v1",
    )


def _evidence() -> tuple[QualificationEvidence[ProviderQualificationEvidenceKind], ...]:
    return (
        QualificationEvidence(
            kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
            code="suite.passed",
            ref="tests/integration/cw/test_postgresql_repository.py",
        ),
        QualificationEvidence(
            kind=ProviderQualificationEvidenceKind.LIVE_BACKEND,
            code="backend.live",
            label="postgresql-16.6",
        ),
    )


def _run(
    *,
    run_id: QualificationRunId | None = None,
    status: QualificationStatus = QualificationStatus.PRODUCTION_QUALIFIED,
) -> ProviderQualificationRun:
    return ProviderQualificationRun(
        qualification_run_id=run_id or new_qualification_run_id(),
        subject=_subject(),
        status=status,
        executed_at=_EXECUTED_AT,
        executor=_executor(),
        result_summary=_result_summary(),
        evidence=_evidence(),
        reproducibility="uv run pytest tests/integration/cw/test_postgresql_repository.py",
        limitations=("capability=collaborative_work.persistence.v1",),
        source_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        environment_metadata=_environment_metadata(),
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("provider_id", ""),
        ("provider_version", "   "),
        ("capability_id", ""),
        ("domain", ""),
        ("intergrax_revision", ""),
        ("qualification_suite_id", ""),
        ("qualification_suite_version", ""),
        ("environment_id", ""),
    ],
)
def test_subject_rejects_empty_required_fields(field_name: str, value: str) -> None:
    payload = {
        "provider_id": "postgresql",
        "provider_version": "16.6",
        "capability_id": "collaborative_work.persistence.v1",
        "domain": "collaborative_work",
        "intergrax_revision": "rev",
        "qualification_suite_id": "suite",
        "qualification_suite_version": "1.0.0",
        "environment_id": "env",
    }
    payload[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        ProviderQualificationSubject(**payload)


@pytest.mark.parametrize(
    ("provider_id", "provider_version"),
    [
        ("postgresql", "16.6"),
        ("oracle", "23ai"),
        ("future-vendor-x", "preview"),
    ],
)
def test_vendor_neutral_provider_identity(provider_id: str, provider_version: str) -> None:
    subject = _subject(provider_id=provider_id, provider_version=provider_version)
    assert subject.provider_id == provider_id
    assert subject.provider_version == provider_version


def test_run_construction_with_explicit_run_id() -> None:
    run_id = new_qualification_run_id()
    run = _run(run_id=run_id)
    assert run.qualification_run_id == run_id
    assert run.status is QualificationStatus.PRODUCTION_QUALIFIED
    assert run.subject.provider_id == "postgresql"


def test_execution_owned_run_id_is_minted_before_run_construction() -> None:
    run_id = new_qualification_run_id()
    validate_qualification_run_id(run_id)
    run = _run(run_id=run_id)
    assert run.qualification_run_id == run_id


def test_new_qualification_run_id_produces_unique_values() -> None:
    first = new_qualification_run_id()
    second = new_qualification_run_id()
    assert first != second
    assert first.startswith("qual_run_")


@pytest.mark.parametrize("count", [-1, -5])
def test_result_summary_rejects_negative_counts(count: int) -> None:
    with pytest.raises(ValueError, match="passed must be >= 0"):
        ProviderQualificationResultSummary(passed=count, failed=0, skipped=0)


def test_result_summary_accepts_zero_counts() -> None:
    summary = ProviderQualificationResultSummary(passed=0, failed=0, skipped=0)
    assert summary.passed == 0
    assert summary.failed == 0
    assert summary.skipped == 0


def test_subject_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _subject().provider_id = "mutated"  # type: ignore[misc]


def test_executor_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _executor().executor_kind = "mutated"  # type: ignore[misc]


def test_result_summary_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _result_summary().passed = 99  # type: ignore[misc]


def test_environment_metadata_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _environment_metadata().mocks = False  # type: ignore[misc]


def test_run_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _run().status = QualificationStatus.REJECTED  # type: ignore[misc]


def test_evidence_tuple_is_immutable_on_run() -> None:
    run = _run()
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.evidence = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        run.evidence[0] = QualificationEvidence(  # type: ignore[index]
            kind=ProviderQualificationEvidenceKind.SOURCE_ANCHOR,
            code="anchor",
        )


def test_status_and_validity_separation_without_validity_on_run() -> None:
    run_id = new_qualification_run_id()
    run = _run(run_id=run_id)
    run_fields = {field.name for field in dataclasses.fields(ProviderQualificationRun)}
    assert "validity" not in run_fields

    validity_current = QualificationValidityRecord(
        qualification_run_id=run_id,
        validity_evaluation_id=new_validity_evaluation_id(),
        validity=QualificationEvidenceValidity.CURRENT,
        evaluated_at=_EVALUATED_AT_T1,
    )
    validity_stale = QualificationValidityRecord(
        qualification_run_id=run_id,
        validity_evaluation_id=new_validity_evaluation_id(),
        validity=QualificationEvidenceValidity.STALE,
        evaluated_at=_EVALUATED_AT_T2,
        reason="adapter_revision_changed",
    )

    assert validity_current.qualification_run_id == run.qualification_run_id
    assert validity_stale.qualification_run_id == run.qualification_run_id
    assert run.status is QualificationStatus.PRODUCTION_QUALIFIED
    assert validity_current.validity is QualificationEvidenceValidity.CURRENT
    assert validity_stale.validity is QualificationEvidenceValidity.STALE


@pytest.mark.parametrize(
    "executor_kind",
    [
        "local_cli",
        "ci_runner",
        "operator_workstation",
        "scheduled_qual_host",
    ],
)
def test_executor_neutrality(executor_kind: str) -> None:
    executor = _executor(executor_kind=executor_kind)
    assert executor.executor_kind == executor_kind


def test_environment_metadata_typing() -> None:
    metadata = ProviderQualificationEnvironmentMetadata(
        real_backend=False,
        mocks=True,
        sqlite_substitution=True,
        bounded_environment="sqlite-substitution",
    )
    assert metadata.real_backend is False
    assert metadata.mocks is True
    assert metadata.sqlite_substitution is True
    assert metadata.bounded_environment == "sqlite-substitution"


def test_evidence_generic_integration() -> None:
    evidence = QualificationEvidence(
        kind=ProviderQualificationEvidenceKind.REPRODUCIBILITY,
        code="repro.command",
        ref="docs/qualification/cw/postgresql.md",
        label="safe rerun reference",
    )
    run = ProviderQualificationRun(
        qualification_run_id=new_qualification_run_id(),
        subject=_subject(),
        status=QualificationStatus.QUALIFIED,
        executed_at=_EXECUTED_AT,
        executor=_executor(),
        result_summary=_result_summary(),
        evidence=(evidence,),
        reproducibility=None,
        limitations=(),
        source_revision="rev",
        environment_metadata=_environment_metadata(),
    )
    assert run.evidence[0].kind is ProviderQualificationEvidenceKind.REPRODUCIBILITY


def test_run_rejects_naive_executed_at() -> None:
    with pytest.raises(ValueError, match="executed_at must be timezone-aware datetime"):
        ProviderQualificationRun(
            qualification_run_id=new_qualification_run_id(),
            subject=_subject(),
            status=QualificationStatus.QUALIFIED,
            executed_at=datetime(2026, 8, 17, 12, 0, 0),
            executor=_executor(),
            result_summary=_result_summary(),
            evidence=(),
            reproducibility=None,
            limitations=(),
            source_revision="rev",
            environment_metadata=_environment_metadata(),
        )


def test_run_rejects_blank_limitation_entries() -> None:
    with pytest.raises(ValueError, match="limitations\\[0\\]"):
        ProviderQualificationRun(
            qualification_run_id=new_qualification_run_id(),
            subject=_subject(),
            status=QualificationStatus.QUALIFIED,
            executed_at=_EXECUTED_AT,
            executor=_executor(),
            result_summary=_result_summary(),
            evidence=(),
            reproducibility=None,
            limitations=("  ",),
            source_revision="rev",
            environment_metadata=_environment_metadata(),
        )


_RUN_A = QualificationRunId("qual_run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
_RUN_B = QualificationRunId("qual_run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")


def _validity_record(
    *,
    run_id: QualificationRunId = _RUN_A,
    validity: QualificationEvidenceValidity = QualificationEvidenceValidity.CURRENT,
) -> QualificationValidityRecord:
    return QualificationValidityRecord(
        qualification_run_id=run_id,
        validity_evaluation_id=new_validity_evaluation_id(),
        validity=validity,
        evaluated_at=_EVALUATED_AT_T1,
    )


def _interpretation(
    *,
    run_id: QualificationRunId = _RUN_A,
    validity: QualificationEvidenceValidity = QualificationEvidenceValidity.CURRENT,
    latest_record: QualificationValidityRecord | None = None,
) -> QualificationValidityInterpretation:
    record = latest_record or _validity_record(run_id=run_id, validity=validity)
    return QualificationValidityInterpretation(
        qualification_run_id=run_id,
        validity=validity,
        latest_record=record,
    )


@pytest.mark.parametrize(
    "validity",
    [
        QualificationEvidenceValidity.CURRENT,
        QualificationEvidenceValidity.STALE,
        QualificationEvidenceValidity.REVOKED,
    ],
)
def test_validity_interpretation_accepts_consistent_object(
    validity: QualificationEvidenceValidity,
) -> None:
    interpretation = _interpretation(validity=validity)
    assert interpretation.qualification_run_id == interpretation.latest_record.qualification_run_id
    assert interpretation.validity == interpretation.latest_record.validity


@pytest.mark.parametrize(
    ("run_id", "error_type", "match"),
    [
        (123, TypeError, "qualification_run_id must be str"),
        ("qual_run_invalid", ValueError, "qualification_run_id suffix"),
    ],
)
def test_validity_interpretation_rejects_invalid_qualification_run_id(
    run_id: object,
    error_type: type[Exception],
    match: str,
) -> None:
    record = _validity_record()
    with pytest.raises(error_type, match=match):
        QualificationValidityInterpretation(
            qualification_run_id=run_id,  # type: ignore[arg-type]
            validity=QualificationEvidenceValidity.CURRENT,
            latest_record=record,
        )


def test_validity_interpretation_rejects_invalid_validity_type() -> None:
    record = _validity_record()
    with pytest.raises(TypeError, match="validity must be QualificationEvidenceValidity"):
        QualificationValidityInterpretation(
            qualification_run_id=_RUN_A,
            validity="current",  # type: ignore[arg-type]
            latest_record=record,
        )


def test_validity_interpretation_rejects_invalid_latest_record_type() -> None:
    with pytest.raises(TypeError, match="latest_record must be QualificationValidityRecord"):
        QualificationValidityInterpretation(
            qualification_run_id=_RUN_A,
            validity=QualificationEvidenceValidity.CURRENT,
            latest_record={"qualification_run_id": _RUN_A},  # type: ignore[arg-type]
        )


def test_validity_interpretation_rejects_run_id_mismatch() -> None:
    record = _validity_record(run_id=_RUN_B)
    with pytest.raises(
        ValueError,
        match="latest_record.qualification_run_id must match qualification_run_id",
    ):
        QualificationValidityInterpretation(
            qualification_run_id=_RUN_A,
            validity=QualificationEvidenceValidity.CURRENT,
            latest_record=record,
        )


def test_validity_interpretation_rejects_validity_mismatch() -> None:
    record = _validity_record(validity=QualificationEvidenceValidity.CURRENT)
    with pytest.raises(ValueError, match="latest_record.validity must match validity"):
        QualificationValidityInterpretation(
            qualification_run_id=_RUN_A,
            validity=QualificationEvidenceValidity.STALE,
            latest_record=record,
        )


def test_validity_interpretation_is_immutable() -> None:
    interpretation = _interpretation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        interpretation.validity = QualificationEvidenceValidity.REVOKED  # type: ignore[misc]


def test_validity_record_is_immutable() -> None:
    record = QualificationValidityRecord(
        qualification_run_id=new_qualification_run_id(),
        validity_evaluation_id=new_validity_evaluation_id(),
        validity=QualificationEvidenceValidity.CURRENT,
        evaluated_at=_EVALUATED_AT_T1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.validity = QualificationEvidenceValidity.REVOKED  # type: ignore[misc]


@pytest.mark.parametrize(
    ("constructor", "kwargs", "field_name"),
    [
        (
            ProviderQualificationSubject,
            {
                "provider_id": 123,
                "provider_version": "16.6",
                "capability_id": "collaborative_work.persistence.v1",
                "domain": "collaborative_work",
                "intergrax_revision": "rev",
                "qualification_suite_id": "suite",
                "qualification_suite_version": "1.0.0",
                "environment_id": "env",
            },
            "provider_id",
        ),
        (
            ProviderQualificationSubject,
            {
                "provider_id": "postgresql",
                "provider_version": "16.6",
                "capability_id": "collaborative_work.persistence.v1",
                "domain": "collaborative_work",
                "intergrax_revision": "rev",
                "qualification_suite_id": "suite",
                "qualification_suite_version": "1.0.0",
                "environment_id": "env",
                "adapter_identity": 123,
            },
            "adapter_identity",
        ),
        (
            ProviderQualificationExecutor,
            {"executor_kind": 123, "executor_id": "qual-host-01"},
            "executor_kind",
        ),
        (
            ProviderQualificationResultSummary,
            {"passed": 0, "failed": 0, "skipped": 0, "label": 123},
            "label",
        ),
        (
            ProviderQualificationEnvironmentMetadata,
            {
                "real_backend": True,
                "mocks": False,
                "sqlite_substitution": False,
                "bounded_environment": 123,
            },
            "bounded_environment",
        ),
    ],
)
def test_contract_rejects_non_string_text_fields(
    constructor: type,
    kwargs: dict,
    field_name: str,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be str"):
        constructor(**kwargs)


def test_run_rejects_non_string_reproducibility() -> None:
    with pytest.raises(TypeError, match="reproducibility must be str"):
        ProviderQualificationRun(
            qualification_run_id=new_qualification_run_id(),
            subject=_subject(),
            status=QualificationStatus.QUALIFIED,
            executed_at=_EXECUTED_AT,
            executor=_executor(),
            result_summary=_result_summary(),
            evidence=(),
            reproducibility=123,
            limitations=(),
            source_revision="rev",
            environment_metadata=_environment_metadata(),
        )


def test_run_rejects_non_string_limitation_element() -> None:
    with pytest.raises(TypeError, match="limitations\\[0\\] must be str"):
        ProviderQualificationRun(
            qualification_run_id=new_qualification_run_id(),
            subject=_subject(),
            status=QualificationStatus.QUALIFIED,
            executed_at=_EXECUTED_AT,
            executor=_executor(),
            result_summary=_result_summary(),
            evidence=(),
            reproducibility=None,
            limitations=(123,),
            source_revision="rev",
            environment_metadata=_environment_metadata(),
        )


def test_validity_record_rejects_non_string_reason() -> None:
    with pytest.raises(TypeError, match="reason must be str"):
        QualificationValidityRecord(
            qualification_run_id=new_qualification_run_id(),
            validity_evaluation_id=new_validity_evaluation_id(),
            validity=QualificationEvidenceValidity.STALE,
            evaluated_at=_EVALUATED_AT_T1,
            reason=123,
        )


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"real_backend": "yes", "mocks": False, "sqlite_substitution": False}, "real_backend"),
        ({"real_backend": 1, "mocks": False, "sqlite_substitution": False}, "real_backend"),
        ({"real_backend": True, "mocks": 0, "sqlite_substitution": False}, "mocks"),
        (
            {"real_backend": True, "mocks": False, "sqlite_substitution": None},
            "sqlite_substitution",
        ),
    ],
)
def test_environment_metadata_rejects_non_bool_fields(kwargs: dict, field_name: str) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be bool"):
        ProviderQualificationEnvironmentMetadata(**kwargs)


def test_environment_metadata_accepts_exact_bool_values() -> None:
    metadata = ProviderQualificationEnvironmentMetadata(
        real_backend=True,
        mocks=False,
        sqlite_substitution=True,
    )
    assert metadata.real_backend is True
    assert metadata.mocks is False
    assert metadata.sqlite_substitution is True


def test_run_rejects_string_evidence_kind() -> None:
    invalid_evidence = QualificationEvidence(
        kind="suite_execution",
        code="suite.passed",
    )
    with pytest.raises(TypeError, match="evidence\\[0\\].kind must be ProviderQualificationEvidenceKind"):
        ProviderQualificationRun(
            qualification_run_id=new_qualification_run_id(),
            subject=_subject(),
            status=QualificationStatus.QUALIFIED,
            executed_at=_EXECUTED_AT,
            executor=_executor(),
            result_summary=_result_summary(),
            evidence=(invalid_evidence,),
            reproducibility=None,
            limitations=(),
            source_revision="rev",
            environment_metadata=_environment_metadata(),
        )


def test_run_rejects_foreign_qualification_evidence_kind() -> None:
    foreign_evidence = QualificationEvidence(
        kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
        code="domain.qualified",
    )
    with pytest.raises(TypeError, match="evidence\\[0\\].kind must be ProviderQualificationEvidenceKind"):
        ProviderQualificationRun(
            qualification_run_id=new_qualification_run_id(),
            subject=_subject(),
            status=QualificationStatus.QUALIFIED,
            executed_at=_EXECUTED_AT,
            executor=_executor(),
            result_summary=_result_summary(),
            evidence=(foreign_evidence,),
            reproducibility=None,
            limitations=(),
            source_revision="rev",
            environment_metadata=_environment_metadata(),
        )
