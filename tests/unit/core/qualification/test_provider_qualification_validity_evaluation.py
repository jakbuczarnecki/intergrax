# © Artur Czarnecki. All rights reserved.

"""Provider qualification validity evaluation tests (PROVIDER-QUAL-5)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.core.qualification import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
    ProviderQualificationValidityContext,
    QualificationEvidence,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationStatus,
    ValidityEvaluationId,
    evaluate_provider_qualification_validity,
    establish_current_qualification_validity,
    get_current_qualification_validity,
    new_qualification_run_id,
    new_validity_evaluation_id,
    record_provider_qualification_validity_revocation,
    validity_context_from_run,
)
from intergrax.core.qualification.validity_evaluation import (
    QualificationValidityEstablishmentError,
    QualificationValidityNotFoundError,
    resolve_latest_qualification_validity,
)

pytestmark = pytest.mark.unit

_EXECUTED_AT = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_T1 = datetime(2026, 8, 17, 13, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_T2 = datetime(2026, 8, 18, 9, 0, 0, tzinfo=timezone.utc)
_FIXED_RUN_ID = QualificationRunId("qual_run_0123456789abcdef0123456789abcdef")
_FIXED_EVAL_ID_CURRENT = ValidityEvaluationId("valid_eval_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
_FIXED_EVAL_ID_STALE = ValidityEvaluationId("valid_eval_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
_EVALUATED_AT_T3 = datetime(2026, 8, 19, 10, 0, 0, tzinfo=timezone.utc)
_FIXED_EVAL_ID_REVOKED = ValidityEvaluationId("valid_eval_cccccccccccccccccccccccccccccccc")
_FIXED_EVAL_ID_LATER = ValidityEvaluationId("valid_eval_dddddddddddddddddddddddddddddddd")


def _subject(
    *,
    provider_id: str = "postgresql",
    provider_version: str = "16.6",
    adapter_identity: str | None = "intergrax.integrations.providers.relational_store.postgresql",
    qualification_suite_version: str = "1.0.0",
    intergrax_revision: str = "bd657b431e2c020da0a89de45f6f3b448a48867a",
) -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id=provider_id,
        provider_version=provider_version,
        capability_id="collaborative_work.persistence.v1",
        domain="collaborative_work",
        intergrax_revision=intergrax_revision,
        qualification_suite_id="cw.postgresql.repository.v1",
        qualification_suite_version=qualification_suite_version,
        environment_id="local-docker-qual-host",
        adapter_identity=adapter_identity,
    )


def _run(
    *,
    run_id: QualificationRunId = _FIXED_RUN_ID,
    subject: ProviderQualificationSubject | None = None,
    source_revision: str = "bd657b431e2c020da0a89de45f6f3b448a48867a",
) -> ProviderQualificationRun:
    return ProviderQualificationRun(
        qualification_run_id=run_id,
        subject=subject or _subject(),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        executed_at=_EXECUTED_AT,
        executor=ProviderQualificationExecutor(
            executor_kind="local_cli",
            executor_id="qual-host-01",
            executor_version="2026.08.17",
        ),
        result_summary=ProviderQualificationResultSummary(
            passed=42,
            failed=0,
            skipped=3,
            label="cw.postgresql.repository.v1",
        ),
        evidence=(
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="suite.passed",
                ref="tests/integration/cw/test_postgresql_repository.py",
            ),
        ),
        reproducibility="uv run pytest tests/integration/cw/test_postgresql_repository.py",
        limitations=("bounded local docker host",),
        source_revision=source_revision,
        environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
            bounded_environment="docker-postgres-16",
        ),
    )


def _context_from_run(run: ProviderQualificationRun) -> ProviderQualificationValidityContext:
    return validity_context_from_run(run)


def test_production_qualified_run_can_have_current_validity() -> None:
    run = _run()
    context = _context_from_run(run)
    record = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    assert run.status is QualificationStatus.PRODUCTION_QUALIFIED
    assert record.validity is QualificationEvidenceValidity.CURRENT
    assert record.qualification_run_id == run.qualification_run_id


def test_same_run_later_becomes_stale_without_mutating_run() -> None:
    run = _run()
    matching_context = _context_from_run(run)
    current_record = evaluate_provider_qualification_validity(
        run,
        matching_context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    stale_context = ProviderQualificationValidityContext(
        provider_id=matching_context.provider_id,
        provider_version=matching_context.provider_version,
        capability_id=matching_context.capability_id,
        domain=matching_context.domain,
        intergrax_revision="new_revision_anchor",
        qualification_suite_id=matching_context.qualification_suite_id,
        qualification_suite_version=matching_context.qualification_suite_version,
        environment_id=matching_context.environment_id,
        source_revision="new_revision_anchor",
        adapter_identity=matching_context.adapter_identity,
    )
    stale_record = evaluate_provider_qualification_validity(
        run,
        stale_context,
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )

    assert current_record.validity is QualificationEvidenceValidity.CURRENT
    assert stale_record.validity is QualificationEvidenceValidity.STALE
    assert stale_record.reason == "intergrax_revision_changed"
    assert run.status is QualificationStatus.PRODUCTION_QUALIFIED


def test_current_record_preserved_after_stale_record_added() -> None:
    run = _run()
    matching_context = _context_from_run(run)
    current_record = evaluate_provider_qualification_validity(
        run,
        matching_context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    stale_context = ProviderQualificationValidityContext(
        provider_id=matching_context.provider_id,
        provider_version="16.7",
        capability_id=matching_context.capability_id,
        domain=matching_context.domain,
        intergrax_revision=matching_context.intergrax_revision,
        qualification_suite_id=matching_context.qualification_suite_id,
        qualification_suite_version=matching_context.qualification_suite_version,
        environment_id=matching_context.environment_id,
        source_revision=matching_context.source_revision,
        adapter_identity=matching_context.adapter_identity,
    )
    stale_record = evaluate_provider_qualification_validity(
        run,
        stale_context,
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )
    interpretation = get_current_qualification_validity(
        run.qualification_run_id,
        (current_record, stale_record),
    )
    assert interpretation.validity is QualificationEvidenceValidity.STALE
    assert interpretation.latest_record == stale_record
    assert current_record.validity is QualificationEvidenceValidity.CURRENT


def test_revoked_overrides_normal_reliance() -> None:
    run = _run()
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_withdrew_evidence",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )
    interpretation = get_current_qualification_validity(run.qualification_run_id, (revoked,))
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert run.status is QualificationStatus.PRODUCTION_QUALIFIED


def test_revoked_is_terminal_after_later_current() -> None:
    run = _run()
    context = _context_from_run(run)
    current_t1 = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_withdrew_evidence",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    current_t3 = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=_FIXED_EVAL_ID_LATER,
    )
    interpretation = get_current_qualification_validity(
        run.qualification_run_id,
        (current_t1, revoked, current_t3),
    )
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert interpretation.latest_record == revoked


def test_revoked_is_terminal_after_later_stale() -> None:
    run = _run()
    matching_context = _context_from_run(run)
    stale_context = ProviderQualificationValidityContext(
        provider_id=matching_context.provider_id,
        provider_version="16.7",
        capability_id=matching_context.capability_id,
        domain=matching_context.domain,
        intergrax_revision=matching_context.intergrax_revision,
        qualification_suite_id=matching_context.qualification_suite_id,
        qualification_suite_version=matching_context.qualification_suite_version,
        environment_id=matching_context.environment_id,
        source_revision=matching_context.source_revision,
        adapter_identity=matching_context.adapter_identity,
    )
    stale_t1 = evaluate_provider_qualification_validity(
        run,
        stale_context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_withdrew_evidence",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    stale_t3 = evaluate_provider_qualification_validity(
        run,
        stale_context,
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=_FIXED_EVAL_ID_LATER,
    )
    interpretation = get_current_qualification_validity(
        run.qualification_run_id,
        (stale_t1, revoked, stale_t3),
    )
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert interpretation.latest_record == revoked


def test_later_current_without_revocation_wins() -> None:
    run = _run()
    context = _context_from_run(run)
    current_t1 = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    stale_context = ProviderQualificationValidityContext(
        provider_id=context.provider_id,
        provider_version="16.7",
        capability_id=context.capability_id,
        domain=context.domain,
        intergrax_revision=context.intergrax_revision,
        qualification_suite_id=context.qualification_suite_id,
        qualification_suite_version=context.qualification_suite_version,
        environment_id=context.environment_id,
        source_revision=context.source_revision,
        adapter_identity=context.adapter_identity,
    )
    stale_t2 = evaluate_provider_qualification_validity(
        run,
        stale_context,
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )
    current_t3 = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=_FIXED_EVAL_ID_LATER,
    )
    interpretation = get_current_qualification_validity(
        run.qualification_run_id,
        (current_t1, stale_t2, current_t3),
    )
    assert interpretation.validity is QualificationEvidenceValidity.CURRENT
    assert interpretation.latest_record == current_t3


def test_revoked_record_preserved_in_history() -> None:
    run = _run()
    context = _context_from_run(run)
    current = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_withdrew_evidence",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    later_current = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=_FIXED_EVAL_ID_LATER,
    )
    history = (current, revoked, later_current)
    assert resolve_latest_qualification_validity(history) == revoked
    assert len(history) == 3
    assert revoked.validity is QualificationEvidenceValidity.REVOKED


def test_non_revoked_tie_breaking_uses_validity_evaluation_id() -> None:
    run = _run()
    context = _context_from_run(run)
    shared_time = _EVALUATED_AT_T2
    earlier = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=shared_time,
        validity_evaluation_id=ValidityEvaluationId(
            "valid_eval_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
    )
    later = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=shared_time,
        validity_evaluation_id=ValidityEvaluationId(
            "valid_eval_ffffffffffffffffffffffffffffffff",
        ),
    )
    resolved = resolve_latest_qualification_validity((earlier, later))
    assert resolved == later


def test_corrupt_validity_scope_fails_closed() -> None:
    run = _run()
    other_run_id = new_qualification_run_id()
    foreign_record = evaluate_provider_qualification_validity(
        run,
        _context_from_run(run),
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    foreign_record = type(foreign_record)(
        qualification_run_id=other_run_id,
        validity_evaluation_id=foreign_record.validity_evaluation_id,
        validity=foreign_record.validity,
        evaluated_at=foreign_record.evaluated_at,
        reason=foreign_record.reason,
        evaluation_context=foreign_record.evaluation_context,
    )
    with pytest.raises(QualificationValidityEstablishmentError, match="mismatch"):
        establish_current_qualification_validity(run.qualification_run_id, (foreign_record,))


def test_unknown_qualification_run_id_without_records_is_explicit() -> None:
    run = _run()
    with pytest.raises(QualificationValidityNotFoundError):
        get_current_qualification_validity(run.qualification_run_id, ())
    with pytest.raises(QualificationValidityEstablishmentError):
        establish_current_qualification_validity(run.qualification_run_id, ())


def test_validity_uses_qualification_run_id_not_proof_id() -> None:
    run = _run()
    record = evaluate_provider_qualification_validity(
        run,
        _context_from_run(run),
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    assert record.qualification_run_id == run.qualification_run_id
    assert str(record.qualification_run_id).startswith("qual_run_")


@pytest.mark.parametrize("provider_id", ["postgresql", "oracle", "mysql"])
def test_provider_neutral_validity_evaluation(provider_id: str) -> None:
    run = _run(subject=_subject(provider_id=provider_id))
    record = evaluate_provider_qualification_validity(
        run,
        _context_from_run(run),
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    assert record.validity is QualificationEvidenceValidity.CURRENT


def test_exact_same_context_is_current() -> None:
    run = _run()
    record = evaluate_provider_qualification_validity(
        run,
        _context_from_run(run),
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    assert record.validity is QualificationEvidenceValidity.CURRENT


def test_provider_version_mismatch_is_stale() -> None:
    run = _run()
    context = ProviderQualificationValidityContext(
        provider_id="postgresql",
        provider_version="16.7",
        capability_id="collaborative_work.persistence.v1",
        domain="collaborative_work",
        intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        qualification_suite_id="cw.postgresql.repository.v1",
        qualification_suite_version="1.0.0",
        environment_id="local-docker-qual-host",
        source_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        adapter_identity="intergrax.integrations.providers.relational_store.postgresql",
    )
    record = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    assert record.validity is QualificationEvidenceValidity.STALE
    assert record.reason == "provider_version_changed"


def test_qualification_suite_version_changed_is_stale() -> None:
    run = _run()
    base = _context_from_run(run)
    context = ProviderQualificationValidityContext(
        provider_id=base.provider_id,
        provider_version=base.provider_version,
        capability_id=base.capability_id,
        domain=base.domain,
        intergrax_revision=base.intergrax_revision,
        qualification_suite_id=base.qualification_suite_id,
        qualification_suite_version="1.1.0",
        environment_id=base.environment_id,
        source_revision=base.source_revision,
        adapter_identity=base.adapter_identity,
    )
    record = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    assert record.validity is QualificationEvidenceValidity.STALE
    assert record.reason == "qualification_suite_version_changed"


def test_adapter_revision_changed_is_stale() -> None:
    run = _run()
    base = _context_from_run(run)
    context = ProviderQualificationValidityContext(
        provider_id=base.provider_id,
        provider_version=base.provider_version,
        capability_id=base.capability_id,
        domain=base.domain,
        intergrax_revision=base.intergrax_revision,
        qualification_suite_id=base.qualification_suite_id,
        qualification_suite_version=base.qualification_suite_version,
        environment_id=base.environment_id,
        source_revision="new_adapter_source_revision",
        adapter_identity=base.adapter_identity,
    )
    record = evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    assert record.validity is QualificationEvidenceValidity.STALE
    assert record.reason == "source_revision_changed"


def test_informational_package_metadata_change_does_not_invalidate() -> None:
    run = _run(
        subject=ProviderQualificationSubject(
            provider_id="postgresql",
            provider_version="16.6",
            capability_id="collaborative_work.persistence.v1",
            domain="collaborative_work",
            intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
            qualification_suite_id="cw.postgresql.repository.v1",
            qualification_suite_version="1.0.0",
            environment_id="local-docker-qual-host",
            adapter_identity="intergrax.integrations.providers.relational_store.postgresql",
            package_name="intergrax-postgresql",
            package_version="0.1.0",
        ),
    )
    record = evaluate_provider_qualification_validity(
        run,
        _context_from_run(run),
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    assert record.validity is QualificationEvidenceValidity.CURRENT
