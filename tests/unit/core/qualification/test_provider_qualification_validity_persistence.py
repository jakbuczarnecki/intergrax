# © Artur Czarnecki. All rights reserved.

"""Provider qualification validity persistence tests (PROVIDER-QUAL-5)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.core.qualification import (
    ProviderQualificationValidityContext,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationValidityRecord,
    ValidityEvaluationId,
    evaluate_provider_qualification_validity,
    new_validity_evaluation_id,
    record_provider_qualification_validity_revocation,
    validity_context_from_run,
)
from intergrax.core.qualification.validity_evaluation import (
    QualificationValidityEstablishmentError,
    get_current_qualification_validity,
)
from intergrax.core.qualification.validity_persistence import (
    DocumentStoreProviderQualificationValidityPersistence,
    ProviderQualificationValidityPersistenceConflictError,
    ProviderQualificationValidityPersistenceIntegrityError,
    decode_qualification_validity_record,
    encode_qualification_validity_record,
    proof_receipt_to_qualification_validity_record,
    qualification_validity_record_to_proof_receipt,
    wire_provider_qualification_validity_persistence,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from tests.unit.core.qualification.test_provider_qualification_persistence import _run

pytestmark = pytest.mark.unit

_EVALUATED_AT_T1 = datetime(2026, 8, 17, 13, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_T2 = datetime(2026, 8, 18, 9, 0, 0, tzinfo=timezone.utc)
_FIXED_EVAL_ID_CURRENT = ValidityEvaluationId("valid_eval_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
_FIXED_EVAL_ID_STALE = ValidityEvaluationId("valid_eval_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
_EVALUATED_AT_T3 = datetime(2026, 8, 19, 10, 0, 0, tzinfo=timezone.utc)
_FIXED_EVAL_ID_REVOKED = ValidityEvaluationId("valid_eval_cccccccccccccccccccccccccccccccc")
_FIXED_EVAL_ID_LATER = ValidityEvaluationId("valid_eval_dddddddddddddddddddddddddddddddd")
_FIXED_EVAL_ID_REVOKED_OLDER = ValidityEvaluationId("valid_eval_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
_FIXED_EVAL_ID_REVOKED_NEWER = ValidityEvaluationId("valid_eval_ffffffffffffffffffffffffffffffff")
_TZ_EASTERN = timezone(timedelta(hours=3))
_EVALUATED_AT_UTC_NOON = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_EASTERN_LATER_STRING = datetime(2026, 9, 1, 14, 30, 0, tzinfo=_TZ_EASTERN)


def _assert_matches_pure_resolver(
    persistence: DocumentStoreProviderQualificationValidityPersistence,
    qualification_run_id: QualificationRunId,
    history: tuple[QualificationValidityRecord, ...],
) -> None:
    pure = get_current_qualification_validity(qualification_run_id, history)
    persisted = persistence.get_current_validity(qualification_run_id)
    assert persisted == pure


def _current_record(run_id: QualificationRunId) -> QualificationValidityRecord:
    run = _run(run_id=run_id)
    return evaluate_provider_qualification_validity(
        run,
        validity_context_from_run(run),
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )


def _stale_record(run_id: QualificationRunId) -> QualificationValidityRecord:
    run = _run(run_id=run_id)
    base = validity_context_from_run(run)
    context = ProviderQualificationValidityContext(
        provider_id=base.provider_id,
        provider_version="16.7",
        capability_id=base.capability_id,
        domain=base.domain,
        intergrax_revision=base.intergrax_revision,
        qualification_suite_id=base.qualification_suite_id,
        qualification_suite_version=base.qualification_suite_version,
        environment_id=base.environment_id,
        source_revision=base.source_revision,
        adapter_identity=base.adapter_identity,
    )
    return evaluate_provider_qualification_validity(
        run,
        context,
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )


def test_append_evaluation_round_trip_and_history() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    current = _current_record(run.qualification_run_id)
    stale = _stale_record(run.qualification_run_id)

    persistence.append_evaluation(current)
    persistence.append_evaluation(stale)
    history = persistence.list_evaluations(run.qualification_run_id)

    assert len(history) == 2
    assert history[0] == current
    assert history[1] == stale


def test_latest_view_returns_stale_after_append() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    persistence.append_evaluation(_current_record(run.qualification_run_id))
    persistence.append_evaluation(_stale_record(run.qualification_run_id))

    interpretation = persistence.get_current_validity(run.qualification_run_id)
    assert interpretation is not None
    assert interpretation.validity is QualificationEvidenceValidity.STALE


def test_append_is_idempotent_for_identical_evaluation() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    record = _current_record(run.qualification_run_id)

    first = persistence.append_evaluation(record)
    second = persistence.append_evaluation(record)

    assert first == record
    assert second == record
    assert len(persistence.list_evaluations(run.qualification_run_id)) == 1


def test_conflicting_duplicate_evaluation_fails_closed() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    record = _current_record(run.qualification_run_id)
    persistence.append_evaluation(record)
    conflicting = replace(record, reason="fabricated_reason")

    with pytest.raises(
        ProviderQualificationValidityPersistenceConflictError,
        match="conflicting provider qualification validity evaluation",
    ):
        persistence.append_evaluation(conflicting)


def test_historical_provider_qualification_run_remains_unchanged() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    run_persistence_store = InMemoryDocumentStore()
    from intergrax.core.qualification.persistence import (
        DocumentStoreProviderQualificationPersistence,
    )

    run_persistence = DocumentStoreProviderQualificationPersistence(run_persistence_store)
    validity_persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    persisted_run = run_persistence.persist(run)
    validity_persistence.append_evaluation(_stale_record(run.qualification_run_id))

    loaded_run = run_persistence.get_by_qualification_run_id(run.qualification_run_id)
    assert loaded_run == persisted_run


def test_unknown_qualification_run_id_returns_none_for_current_view() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    unknown = QualificationRunId("qual_run_99999999999999999999999999999999")
    assert persistence.get_current_validity(unknown) is None


def test_establish_current_validity_fails_closed_without_records() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    with pytest.raises(QualificationValidityEstablishmentError):
        persistence.establish_current_validity(run.qualification_run_id)


def test_proof_receipt_projection_uses_qualification_run_id() -> None:
    run = _run()
    record = _current_record(run.qualification_run_id)
    receipt = qualification_validity_record_to_proof_receipt(record)
    round_trip = proof_receipt_to_qualification_validity_record(receipt)

    assert receipt.metadata["qualification_run_id"] == str(run.qualification_run_id)
    assert round_trip == record


def test_decode_corrupt_validity_evidence_fails_closed() -> None:
    with pytest.raises(ProviderQualificationValidityPersistenceIntegrityError):
        decode_qualification_validity_record({"schema_version": "wrong"})


def test_encode_decode_round_trip() -> None:
    run = _run()
    record = _current_record(run.qualification_run_id)
    payload = encode_qualification_validity_record(record)
    decoded = decode_qualification_validity_record(payload)
    assert decoded == record


def test_revoked_record_persists_and_resolves() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_revoked",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    persistence.append_evaluation(revoked)
    interpretation = persistence.get_current_validity(run.qualification_run_id)
    assert interpretation is not None
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED


def test_terminal_revocation_current_view_after_later_current_append() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    current = _current_record(run.qualification_run_id)
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_revoked",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    later_current = replace(
        current,
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=_FIXED_EVAL_ID_LATER,
    )
    persistence.append_evaluation(current)
    persistence.append_evaluation(revoked)
    persistence.append_evaluation(later_current)

    interpretation = persistence.get_current_validity(run.qualification_run_id)
    history = persistence.list_evaluations(run.qualification_run_id)

    assert interpretation is not None
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert len(history) == 3
    assert history[0] == current
    assert history[1] == revoked
    assert history[2] == later_current


def test_get_current_validity_uses_bounded_decode_not_full_history() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    current = _current_record(run.qualification_run_id)
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_revoked",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    persistence.append_evaluation(current)
    persistence.append_evaluation(revoked)

    decode_calls = 0
    original_decode = persistence._document_to_record

    def counting_decode(document: object) -> QualificationValidityRecord:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(document)

    persistence._document_to_record = counting_decode  # type: ignore[method-assign]

    for index in range(98):
        persistence.append_evaluation(
            replace(
                current,
                evaluated_at=_EVALUATED_AT_T3 + timedelta(seconds=index),
                validity_evaluation_id=new_validity_evaluation_id(),
            ),
        )

    interpretation = persistence.get_current_validity(run.qualification_run_id)

    assert interpretation is not None
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert decode_calls == 1


def test_get_current_validity_without_revocation_decodes_single_latest_record() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    current = _current_record(run.qualification_run_id)
    persistence.append_evaluation(current)

    decode_calls = 0
    original_decode = persistence._document_to_record

    def counting_decode(document: object) -> QualificationValidityRecord:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(document)

    persistence._document_to_record = counting_decode  # type: ignore[method-assign]

    for index in range(99):
        persistence.append_evaluation(
            replace(
                current,
                evaluated_at=_EVALUATED_AT_T3 + timedelta(seconds=index),
                validity_evaluation_id=new_validity_evaluation_id(),
            ),
        )

    interpretation = persistence.get_current_validity(run.qualification_run_id)

    assert interpretation is not None
    assert interpretation.validity is QualificationEvidenceValidity.CURRENT
    assert decode_calls == 1


def test_wire_provider_qualification_validity_persistence() -> None:
    store = InMemoryDocumentStore()
    persistence = wire_provider_qualification_validity_persistence(document_store=store)
    assert isinstance(persistence, DocumentStoreProviderQualificationValidityPersistence)


def test_multiple_revoked_records_choose_newest_revoked() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    older_revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="first_revocation",
        evaluated_at=_EVALUATED_AT_T1,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    newer_revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="second_revocation",
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED_NEWER,
    )
    persistence.append_evaluation(older_revoked)
    persistence.append_evaluation(newer_revoked)

    interpretation = persistence.get_current_validity(run.qualification_run_id)
    history = persistence.list_evaluations(run.qualification_run_id)

    assert interpretation is not None
    assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    assert interpretation.latest_record == newer_revoked
    _assert_matches_pure_resolver(persistence, run.qualification_run_id, history)


def test_revoked_same_timestamp_tie_breaks_by_validity_evaluation_id() -> None:
    run = _run()
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationValidityPersistence(store)
    shared_time = _EVALUATED_AT_T2
    earlier_revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="first_revocation",
        evaluated_at=shared_time,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED,
    )
    later_revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="second_revocation",
        evaluated_at=shared_time,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED_NEWER,
    )
    persistence.append_evaluation(earlier_revoked)
    persistence.append_evaluation(later_revoked)

    interpretation = persistence.get_current_validity(run.qualification_run_id)
    history = persistence.list_evaluations(run.qualification_run_id)

    assert interpretation is not None
    assert interpretation.latest_record == later_revoked
    _assert_matches_pure_resolver(persistence, run.qualification_run_id, history)


def test_evaluated_at_persists_as_canonical_utc() -> None:
    run = _run()
    record = replace(
        _current_record(run.qualification_run_id),
        evaluated_at=_EVALUATED_AT_EASTERN_LATER_STRING,
    )
    payload = encode_qualification_validity_record(record)
    assert payload["evaluated_at"] == "2026-09-01T11:30:00+00:00"


def test_encode_decode_preserves_same_instant_across_timezones() -> None:
    run = _run()
    record = replace(
        _current_record(run.qualification_run_id),
        evaluated_at=_EVALUATED_AT_EASTERN_LATER_STRING,
    )
    decoded = decode_qualification_validity_record(encode_qualification_validity_record(record))
    assert decoded.evaluated_at == _EVALUATED_AT_EASTERN_LATER_STRING.astimezone(timezone.utc)
    assert decoded.evaluated_at == record.evaluated_at


def test_mixed_timezone_current_view_matches_pure_resolver() -> None:
    run = _run()
    utc_record = replace(
        _current_record(run.qualification_run_id),
        evaluated_at=_EVALUATED_AT_UTC_NOON,
        validity_evaluation_id=_FIXED_EVAL_ID_CURRENT,
    )
    offset_record = replace(
        _current_record(run.qualification_run_id),
        evaluated_at=_EVALUATED_AT_EASTERN_LATER_STRING,
        validity_evaluation_id=_FIXED_EVAL_ID_STALE,
    )

    for records in ((utc_record, offset_record), (offset_record, utc_record)):
        store_b = InMemoryDocumentStore()
        persistence_b = DocumentStoreProviderQualificationValidityPersistence(store_b)
        for record in records:
            persistence_b.append_evaluation(record)
        history = persistence_b.list_evaluations(run.qualification_run_id)
        interpretation = persistence_b.get_current_validity(run.qualification_run_id)
        assert interpretation is not None
        assert interpretation.latest_record == utc_record
        _assert_matches_pure_resolver(persistence_b, run.qualification_run_id, history)


def test_revoked_mixed_timezone_current_view_matches_pure_resolver() -> None:
    run = _run()
    utc_revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="utc_revocation",
        evaluated_at=_EVALUATED_AT_UTC_NOON,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED_NEWER,
    )
    offset_revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="offset_revocation",
        evaluated_at=_EVALUATED_AT_EASTERN_LATER_STRING,
        validity_evaluation_id=_FIXED_EVAL_ID_REVOKED_OLDER,
    )

    for records in ((utc_revoked, offset_revoked), (offset_revoked, utc_revoked)):
        store = InMemoryDocumentStore()
        persistence = DocumentStoreProviderQualificationValidityPersistence(store)
        for record in records:
            persistence.append_evaluation(record)
        history = persistence.list_evaluations(run.qualification_run_id)
        interpretation = persistence.get_current_validity(run.qualification_run_id)
        assert interpretation is not None
        assert interpretation.validity is QualificationEvidenceValidity.REVOKED
        assert interpretation.latest_record == utc_revoked
        _assert_matches_pure_resolver(persistence, run.qualification_run_id, history)


def test_decode_naive_evaluated_at_fails_closed() -> None:
    run = _run()
    record = _current_record(run.qualification_run_id)
    payload = encode_qualification_validity_record(record)
    payload = dict(payload)
    payload["evaluated_at"] = "2026-08-17T13:00:00"
    with pytest.raises(
        ProviderQualificationValidityPersistenceIntegrityError,
        match="timezone-aware datetime required",
    ):
        decode_qualification_validity_record(payload)
