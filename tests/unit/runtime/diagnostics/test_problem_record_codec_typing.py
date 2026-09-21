# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC1-R10 — strong typing closure for problem persistence codec."""

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_record_codec import (
    _encode_legacy_problem_payload_v1,
    decode_legacy_problem_record_with_occurrences,
    decode_problem_record,
    encode_problem_record,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PERSISTENCE_SCHEMA_V3 = "intergrax.diagnostic_problem.persistence.v3"
_PERSISTENCE_SCHEMA_V1 = "intergrax.diagnostic_problem.persistence.v1"
_TENANT = "tenant-codec-typing"


def test_encode_problem_record_v3_bounded_mapping() -> None:
    problem = sample_problem(tenant_id=_TENANT)
    encoded = encode_problem_record(problem)
    assert encoded["schema_version"] == _PERSISTENCE_SCHEMA_V3
    payload = encoded["payload"]
    assert isinstance(payload, dict)
    assert all(isinstance(key, str) for key in payload)


def test_encode_decode_roundtrip_preserves_problem() -> None:
    problem = sample_problem(tenant_id=_TENANT, record_version=2, occurrence_count=3)
    decoded = decode_problem_record(encode_problem_record(problem))
    assert decoded == problem


def test_invalid_occurrence_count_fail_closed() -> None:
    problem = sample_problem(tenant_id=_TENANT)
    record = encode_problem_record(problem)
    payload = dict(record["payload"])
    payload["occurrence_count"] = "not-an-int"
    record = {**record, "payload": payload}
    with pytest.raises(ProblemPersistenceIntegrityError):
        decode_problem_record(record)


def test_invalid_record_version_fail_closed() -> None:
    problem = sample_problem(tenant_id=_TENANT)
    record = encode_problem_record(problem)
    payload = dict(record["payload"])
    payload["record_version"] = {"bad": True}
    record = {**record, "payload": payload}
    with pytest.raises(ProblemPersistenceIntegrityError):
        decode_problem_record(record)


def test_legacy_v1_decode_with_inline_occurrences() -> None:
    from intergrax.runtime.diagnostics.persistence_conformance import (
        _sample_subject_ref,
    )

    problem = sample_problem(tenant_id=_TENANT, occurrence_count=1)
    subject_ref = _sample_subject_ref(tenant_id=_TENANT)
    occurrences = sample_occurrences(subject_refs=(subject_ref,))
    bounded = sample_problem(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        subject_refs=(subject_ref,),
        occurrence_count=1,
    )
    legacy = {
        "schema_version": _PERSISTENCE_SCHEMA_V1,
        "payload": _encode_legacy_problem_payload_v1(
            problem=bounded,
            current_subject_refs=(subject_ref,),
            occurrences=occurrences,
        ),
    }
    decoded_problem, decoded_occurrences, decoded_subject_refs = (
        decode_legacy_problem_record_with_occurrences(legacy)
    )
    assert decoded_problem.problem_id == bounded.problem_id
    assert len(decoded_occurrences) == 1
    assert len(decoded_subject_refs) == 1
