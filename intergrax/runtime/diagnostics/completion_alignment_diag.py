# © Artur Czarnecki. All rights reserved.

"""Canonical completion alignment diagnostic contract (DS-E2E-15J-O2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


class CompletionMode(StrEnum):
    UNRESOLVED = "unresolved"
    SUPPORTED_DIAGNOSIS = "supported_diagnosis"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    UNKNOWN = "unknown"


class AlignmentStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"


class AlignmentDirection(StrEnum):
    NONE = "none"
    FORWARD = "forward"
    REVERSE = "reverse"
    UNKNOWN = "unknown"


def completion_mode_from_literal(value: str) -> CompletionMode:
    try:
        return CompletionMode(value)
    except ValueError:
        return CompletionMode.UNKNOWN


@dataclass(frozen=True, slots=True)
class CompletionAlignmentDiagV1(DiagnosticPayload):
    run_id: RunId
    node_id: str
    completion_mode: CompletionMode
    alignment_status: AlignmentStatus
    alignment_direction: AlignmentDirection
    mismatch_reason: str | None
    correctable: bool
    supported_state_present: bool
    supported_hypothesis_id: str | None
    supported_resolution: str | None

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.completion.alignment.v1"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    @property
    def schema_id_value(self) -> str:
        return self.schema_id()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id(),
            "run_id": str(validate_run_id(self.run_id)),
            "node_id": self.node_id,
            "completion_mode": self.completion_mode.value,
            "alignment_status": self.alignment_status.value,
            "alignment_direction": self.alignment_direction.value,
            "mismatch_reason": self.mismatch_reason,
            "correctable": self.correctable,
            "supported_state_present": self.supported_state_present,
            "supported_hypothesis_id": self.supported_hypothesis_id,
            "supported_resolution": self.supported_resolution,
        }

    def redact(self) -> CompletionAlignmentDiagV1:
        return self


def _required_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str):
        return None
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, bool):
        return None
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None | object:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        return object()
    return value


def decode_completion_alignment_diag_v1(
    payload: Mapping[str, object],
) -> CompletionAlignmentDiagV1 | None:
    """Strict decode for persisted trace payloads (fail-closed on shape drift)."""
    run_id_raw = _required_str(payload, "run_id")
    node_id = _required_str(payload, "node_id")
    completion_mode_raw = _required_str(payload, "completion_mode")
    alignment_status_raw = _required_str(payload, "alignment_status")
    alignment_direction_raw = _required_str(payload, "alignment_direction")
    correctable = _required_bool(payload, "correctable")
    supported_state_present = _required_bool(payload, "supported_state_present")
    if (
        run_id_raw is None
        or node_id is None
        or completion_mode_raw is None
        or alignment_status_raw is None
        or alignment_direction_raw is None
        or correctable is None
        or supported_state_present is None
    ):
        return None
    mismatch_reason = _optional_str(payload, "mismatch_reason")
    if mismatch_reason is object():
        return None
    supported_hypothesis_id = _optional_str(payload, "supported_hypothesis_id")
    if supported_hypothesis_id is object():
        return None
    supported_resolution = _optional_str(payload, "supported_resolution")
    if supported_resolution is object():
        return None
    try:
        alignment_status = AlignmentStatus(alignment_status_raw)
        alignment_direction = AlignmentDirection(alignment_direction_raw)
    except ValueError:
        return None
    try:
        run_id = validate_run_id(run_id_raw)
    except (TypeError, ValueError):
        return None
    return CompletionAlignmentDiagV1(
        run_id=run_id,
        node_id=node_id,
        completion_mode=completion_mode_from_literal(completion_mode_raw),
        alignment_status=alignment_status,
        alignment_direction=alignment_direction,
        mismatch_reason=mismatch_reason,
        correctable=correctable,
        supported_state_present=supported_state_present,
        supported_hypothesis_id=supported_hypothesis_id,
        supported_resolution=supported_resolution,
    )
