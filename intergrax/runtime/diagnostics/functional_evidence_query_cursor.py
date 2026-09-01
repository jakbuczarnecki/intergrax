# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Authenticated keyset cursor codec for functional evidence queries (DIAG-FUNCTIONAL-1-R1)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.runtime.diagnostics.functional_evidence import PipelineEvidenceKind

_FUNCTIONAL_EVIDENCE_CURSOR_SCHEMA = "intergrax.functional_evidence_query_cursor.v1"
_FUNCTIONAL_EVIDENCE_CURSOR_MAX_TOKEN_LENGTH = 4096
_MIN_FUNCTIONAL_EVIDENCE_CURSOR_SECRET_BYTES = 32


class FunctionalEvidenceQueryCursorError(Exception):
    """Raised when a functional evidence continuation cursor is invalid or scope-mismatched."""


class FunctionalEvidenceQueryCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["intergrax.functional_evidence_query_cursor.v1"]
    tenant_id: str
    task_id: str
    run_id: str
    attempt_id: str | None
    kind: str | None
    last_recorded_at: datetime
    last_evidence_id: str


class FunctionalEvidenceQueryCursorCodec:
    """Authenticated, query-bound codec for functional evidence keyset cursors."""

    def __init__(self, *, secret: bytes) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("functional_evidence_cursor_secret_invalid")
        if len(secret) < _MIN_FUNCTIONAL_EVIDENCE_CURSOR_SECRET_BYTES:
            raise ValueError("functional_evidence_cursor_secret_too_short")
        self._secret = secret

    @staticmethod
    def _canonical_payload(payload: FunctionalEvidenceQueryCursorPayloadV1) -> bytes:
        return json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _encode_base64url(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_base64url(value: object) -> bytes:
        if not isinstance(value, str) or not value or len(value) % 4 == 1:
            raise ValueError
        padding = "=" * (-len(value) % 4)
        return base64.b64decode(
            (value + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )

    def encode(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId | None,
        kind: PipelineEvidenceKind | None,
        last_recorded_at: datetime,
        last_evidence_id: EventId,
    ) -> str:
        payload = FunctionalEvidenceQueryCursorPayloadV1(
            schema_version=_FUNCTIONAL_EVIDENCE_CURSOR_SCHEMA,
            tenant_id=tenant_id,
            task_id=str(task_id),
            run_id=str(run_id),
            attempt_id=str(attempt_id) if attempt_id is not None else None,
            kind=kind.value if kind is not None else None,
            last_recorded_at=last_recorded_at,
            last_evidence_id=str(last_evidence_id),
        )
        canonical = self._canonical_payload(payload)
        signature = hmac.new(self._secret, canonical, digestmod="sha256").digest()
        envelope = {
            "payload": self._encode_base64url(canonical),
            "signature": self._encode_base64url(signature),
        }
        encoded = self._encode_base64url(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        if len(encoded) > _FUNCTIONAL_EVIDENCE_CURSOR_MAX_TOKEN_LENGTH:
            raise FunctionalEvidenceQueryCursorError("functional evidence cursor too long")
        return encoded

    def decode(
        self,
        cursor: str,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId | None,
        kind: PipelineEvidenceKind | None,
    ) -> FunctionalEvidenceQueryCursorPayloadV1:
        if not isinstance(cursor, str) or not cursor:
            raise FunctionalEvidenceQueryCursorError("functional evidence cursor is required")
        if len(cursor) > _FUNCTIONAL_EVIDENCE_CURSOR_MAX_TOKEN_LENGTH:
            raise FunctionalEvidenceQueryCursorError("functional evidence cursor too long")
        try:
            envelope_bytes = self._decode_base64url(cursor)
            envelope = json.loads(envelope_bytes.decode("utf-8"))
            payload_bytes = self._decode_base64url(envelope["payload"])
            provided_signature = self._decode_base64url(envelope["signature"])
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            binascii.Error,
        ) as exc:
            raise FunctionalEvidenceQueryCursorError("invalid functional evidence cursor") from exc

        try:
            payload = FunctionalEvidenceQueryCursorPayloadV1.model_validate_json(
                payload_bytes.decode("utf-8"),
                strict=True,
            )
        except ValidationError as exc:
            raise FunctionalEvidenceQueryCursorError("invalid functional evidence cursor") from exc

        expected_signature = hmac.new(
            self._secret,
            payload_bytes,
            digestmod="sha256",
        ).digest()
        if not hmac.compare_digest(provided_signature, expected_signature):
            raise FunctionalEvidenceQueryCursorError("functional evidence cursor signature invalid")

        expected_attempt = str(attempt_id) if attempt_id is not None else None
        expected_kind = kind.value if kind is not None else None
        if (
            payload.tenant_id != tenant_id
            or payload.task_id != str(task_id)
            or payload.run_id != str(run_id)
            or payload.attempt_id != expected_attempt
            or payload.kind != expected_kind
        ):
            raise FunctionalEvidenceQueryCursorError("functional evidence cursor scope mismatch")
        return payload


__all__ = [
    "FunctionalEvidenceQueryCursorCodec",
    "FunctionalEvidenceQueryCursorError",
    "FunctionalEvidenceQueryCursorPayloadV1",
]
