# © Artur Czarnecki. All rights reserved.

"""Provider requalification decision tests (PROVIDER-QUAL-6)."""

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
    QualificationValidityRecord,
    ValidityEvaluationId,
    determine_provider_requalification_requirement,
    establish_provider_requalification_requirement,
    get_current_qualification_validity,
    prepare_provider_requalification_run_identity,
    record_provider_qualification_validity_revocation,
)
from intergrax.core.qualification.requalification import (
    ProviderRequalificationPreparationError,
)
from intergrax.core.qualification.validity_evaluation import (
    QualificationValidityEstablishmentError,
)

pytestmark = pytest.mark.unit

_EXECUTED_AT = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
_DECIDED_AT = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
_RUN_A = QualificationRunId("qual_run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
_RUN_B = QualificationRunId("qual_run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
_RUN_C = QualificationRunId("qual_run_cccccccccccccccccccccccccccccccc")
_EVAL_CURRENT = ValidityEvaluationId("valid_eval_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
_EVAL_STALE = ValidityEvaluationId("valid_eval_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
_EVAL_REVOKED = ValidityEvaluationId("valid_eval_cccccccccccccccccccccccccccccccc")


def _subject(*, provider_id: str = "postgresql") -> ProviderQualificationSubject:
    return ProviderQualificationSubject(
        provider_id=provider_id,
        provider_version="16.6",
        capability_id="collaborative_work.persistence.v1",
        domain="collaborative_work",
        intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        qualification_suite_id="cw.postgresql.repository.v1",
        qualification_suite_version="1.0.0",
        environment_id="local-docker-qual-host",
        adapter_identity="intergrax.integrations.providers.relational_store.postgresql",
    )


def _run(
    *,
    run_id: QualificationRunId = _RUN_A,
    provider_id: str = "postgresql",
) -> ProviderQualificationRun:
    return ProviderQualificationRun(
        qualification_run_id=run_id,
        subject=_subject(provider_id=provider_id),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        executed_at=_EXECUTED_AT,
        executor=ProviderQualificationExecutor(
            executor_kind="local_cli",
            executor_id="qual-host-01",
        ),
        result_summary=ProviderQualificationResultSummary(
            passed=42,
            failed=0,
            skipped=0,
        ),
        evidence=(
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="suite.passed",
                ref="tests/integration/cw/test_postgresql_repository.py",
            ),
        ),
        reproducibility=None,
        limitations=(),
        source_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
        ),
    )


def _current_record(
    run_id: QualificationRunId = _RUN_A,
) -> QualificationValidityRecord:
    return QualificationValidityRecord(
        qualification_run_id=run_id,
        validity_evaluation_id=_EVAL_CURRENT,
        validity=QualificationEvidenceValidity.CURRENT,
        evaluated_at=_DECIDED_AT,
    )


def _stale_record(
    *,
    run_id: QualificationRunId = _RUN_A,
    reason: str = "provider_version_changed",
) -> QualificationValidityRecord:
    return QualificationValidityRecord(
        qualification_run_id=run_id,
        validity_evaluation_id=_EVAL_STALE,
        validity=QualificationEvidenceValidity.STALE,
        evaluated_at=_DECIDED_AT,
        reason=reason,
        evaluation_context=ProviderQualificationValidityContext(
            provider_id="postgresql",
            provider_version="16.7",
            capability_id="collaborative_work.persistence.v1",
            domain="collaborative_work",
            intergrax_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
            qualification_suite_id="cw.postgresql.repository.v1",
            qualification_suite_version="1.0.0",
            environment_id="local-docker-qual-host",
            source_revision="bd657b431e2c020da0a89de45f6f3b448a48867a",
        ),
    )


def test_current_does_not_require_requalification() -> None:
    interpretation = get_current_qualification_validity(_RUN_A, (_current_record(),))
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    assert decision.required is False
    assert decision.based_on_validity is QualificationEvidenceValidity.CURRENT
    assert decision.prior_run_remains_terminal is False


def test_stale_requires_requalification() -> None:
    interpretation = get_current_qualification_validity(_RUN_A, (_stale_record(),))
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    assert decision.required is True
    assert decision.based_on_validity is QualificationEvidenceValidity.STALE


def test_stale_reason_is_preserved() -> None:
    interpretation = get_current_qualification_validity(
        _RUN_A,
        (_stale_record(reason="qualification_suite_version_changed"),),
    )
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    assert decision.reason == "qualification_suite_version_changed"


def test_revoked_old_run_remains_terminal() -> None:
    revoked = record_provider_qualification_validity_revocation(
        _RUN_A,
        reason="manual_revocation",
        evaluated_at=_DECIDED_AT,
        validity_evaluation_id=_EVAL_REVOKED,
    )
    interpretation = get_current_qualification_validity(_RUN_A, (revoked,))
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert decision.prior_run_remains_terminal is True
    assert decision.required is True
    assert decision.reason == "manual_revocation"


def test_revoked_later_stale_does_not_reactivate_old_run() -> None:
    revoked = record_provider_qualification_validity_revocation(
        _RUN_A,
        reason="manual_revocation",
        evaluated_at=_DECIDED_AT,
        validity_evaluation_id=_EVAL_REVOKED,
    )
    later_stale = _stale_record()
    interpretation = get_current_qualification_validity(_RUN_A, (revoked, later_stale))
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    assert decision.prior_run_remains_terminal is True
    assert decision.required is True


def test_missing_validity_fails_closed() -> None:
    with pytest.raises(QualificationValidityEstablishmentError):
        establish_provider_requalification_requirement(
            _RUN_A,
            (),
            decided_at=_DECIDED_AT,
        )


def test_historical_run_object_remains_unchanged() -> None:
    run = _run()
    before = (
        run.qualification_run_id,
        run.subject,
        run.status,
        run.executed_at,
        run.source_revision,
    )
    interpretation = get_current_qualification_validity(
        run.qualification_run_id,
        (_stale_record(run_id=run.qualification_run_id),),
    )
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    identity = prepare_provider_requalification_run_identity(decision)
    after = (
        run.qualification_run_id,
        run.subject,
        run.status,
        run.executed_at,
        run.source_revision,
    )
    assert before == after
    assert identity.prior_qualification_run_id == run.qualification_run_id


def test_new_run_receives_distinct_qualification_run_id() -> None:
    interpretation = get_current_qualification_validity(_RUN_A, (_stale_record(),))
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    identity = prepare_provider_requalification_run_identity(decision)
    assert identity.new_qualification_run_id != identity.prior_qualification_run_id
    assert identity.new_qualification_run_id != _RUN_A


def test_current_does_not_prepare_new_run_identity() -> None:
    interpretation = get_current_qualification_validity(_RUN_A, (_current_record(),))
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    with pytest.raises(ProviderRequalificationPreparationError):
        prepare_provider_requalification_run_identity(decision)


def test_multiple_requalifications_preserve_append_only_history() -> None:
    run_a = _run(run_id=_RUN_A)
    run_b = _run(run_id=_RUN_B, provider_id="sqlite")
    run_c = _run(run_id=_RUN_C, provider_id="oracle")

    decision_a = determine_provider_requalification_requirement(
        get_current_qualification_validity(run_a.qualification_run_id, (_stale_record(run_id=_RUN_A),)),
        decided_at=_DECIDED_AT,
    )
    identity_a = prepare_provider_requalification_run_identity(decision_a)

    decision_b = determine_provider_requalification_requirement(
        get_current_qualification_validity(run_b.qualification_run_id, (_stale_record(run_id=_RUN_B),)),
        decided_at=_DECIDED_AT,
    )
    identity_b = prepare_provider_requalification_run_identity(decision_b)

    decision_c = determine_provider_requalification_requirement(
        get_current_qualification_validity(run_c.qualification_run_id, (_stale_record(run_id=_RUN_C),)),
        decided_at=_DECIDED_AT,
    )
    identity_c = prepare_provider_requalification_run_identity(decision_c)

    run_ids = {
        run_a.qualification_run_id,
        run_b.qualification_run_id,
        run_c.qualification_run_id,
        identity_a.new_qualification_run_id,
        identity_b.new_qualification_run_id,
        identity_c.new_qualification_run_id,
    }
    assert len(run_ids) == 6


@pytest.mark.parametrize(
    "provider_id",
    ("postgresql", "sqlite", "oracle"),
)
def test_provider_neutral_requalification_decision(provider_id: str) -> None:
    run = _run(provider_id=provider_id)
    interpretation = get_current_qualification_validity(
        run.qualification_run_id,
        (_stale_record(run_id=run.qualification_run_id, reason="adapter_identity_changed"),),
    )
    decision = determine_provider_requalification_requirement(
        interpretation,
        decided_at=_DECIDED_AT,
    )
    assert decision.required is True
    assert decision.reason == "adapter_identity_changed"
    assert decision.based_on_validity is QualificationEvidenceValidity.STALE
