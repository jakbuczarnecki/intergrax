# © Artur Czarnecki. All rights reserved.

"""Completion alignment trace schema decoder integrity (OBS-DIAG-RECERT-P2A-R2)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import mint_run_id, validate_run_id
from intergrax.runtime.nexus.tracing.execution.completion_alignment import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionAlignmentDiagV1,
    CompletionMode,
    decode_completion_alignment_diag_v1,
)

_EXPECTED_WIRE_KEYS = frozenset(
    {
        "schema_id",
        "run_id",
        "node_id",
        "completion_mode",
        "alignment_status",
        "alignment_direction",
        "mismatch_reason",
        "correctable",
        "supported_state_present",
        "supported_hypothesis_id",
        "supported_resolution",
    }
)


def _valid_payload_dict() -> dict[str, object]:
    run_id = mint_run_id()
    return {
        "schema_id": CompletionAlignmentDiagV1.schema_id(),
        "run_id": str(run_id),
        "node_id": "node_test",
        "completion_mode": CompletionMode.SUPPORTED_DIAGNOSIS.value,
        "alignment_status": AlignmentStatus.MISMATCH.value,
        "alignment_direction": AlignmentDirection.REVERSE.value,
        "mismatch_reason": "supported_diagnosis_without_supported_state",
        "correctable": True,
        "supported_state_present": False,
        "supported_hypothesis_id": "hyp-1",
        "supported_resolution": "fix-it",
    }


def _typed_from_dict(data: dict[str, object]) -> CompletionAlignmentDiagV1:
    run_id = validate_run_id(str(data["run_id"]))
    return CompletionAlignmentDiagV1(
        run_id=run_id,
        node_id=str(data["node_id"]),
        completion_mode=CompletionMode(str(data["completion_mode"])),
        alignment_status=AlignmentStatus(str(data["alignment_status"])),
        alignment_direction=AlignmentDirection(str(data["alignment_direction"])),
        mismatch_reason=data.get("mismatch_reason"),  # type: ignore[arg-type]
        correctable=bool(data["correctable"]),
        supported_state_present=bool(data["supported_state_present"]),
        supported_hypothesis_id=data.get("supported_hypothesis_id"),  # type: ignore[arg-type]
        supported_resolution=data.get("supported_resolution"),  # type: ignore[arg-type]
    )


def test_schema_id_and_version_unchanged() -> None:
    assert (
        CompletionAlignmentDiagV1.schema_id()
        == "intergrax.diag.completion.alignment.v1"
    )
    assert CompletionAlignmentDiagV1.schema_version() == 1


def test_wire_shape_keys_unchanged() -> None:
    payload = _typed_from_dict(_valid_payload_dict())
    assert frozenset(payload.to_dict()) == _EXPECTED_WIRE_KEYS


def test_valid_roundtrip() -> None:
    original = _typed_from_dict(_valid_payload_dict())
    decoded = decode_completion_alignment_diag_v1(original.to_dict())
    assert decoded == original


def test_optional_fields_absent_valid() -> None:
    data = _valid_payload_dict()
    data.pop("mismatch_reason")
    data.pop("supported_hypothesis_id")
    data.pop("supported_resolution")
    decoded = decode_completion_alignment_diag_v1(data)
    assert decoded is not None
    assert decoded.mismatch_reason is None
    assert decoded.supported_hypothesis_id is None
    assert decoded.supported_resolution is None


def test_optional_fields_none_valid() -> None:
    data = _valid_payload_dict()
    data["mismatch_reason"] = None
    data["supported_hypothesis_id"] = None
    data["supported_resolution"] = None
    decoded = decode_completion_alignment_diag_v1(data)
    assert decoded is not None


def test_malformed_mismatch_reason_rejected() -> None:
    data = _valid_payload_dict()
    data["mismatch_reason"] = 123
    assert decode_completion_alignment_diag_v1(data) is None


def test_malformed_supported_hypothesis_id_rejected() -> None:
    data = _valid_payload_dict()
    data["supported_hypothesis_id"] = False
    assert decode_completion_alignment_diag_v1(data) is None


def test_malformed_supported_resolution_rejected() -> None:
    data = _valid_payload_dict()
    data["supported_resolution"] = {"x": "y"}
    assert decode_completion_alignment_diag_v1(data) is None
