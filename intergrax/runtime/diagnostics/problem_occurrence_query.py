# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded ProblemOccurrence query cursor codec (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId

_OCCURRENCE_CURSOR_SCHEMA = "intergrax.diagnostic_problem_occurrence_cursor.v1"
_OCCURRENCE_CURSOR_MAX_TOKEN_LENGTH = 4096
_MIN_OCCURRENCE_CURSOR_SECRET_BYTES = 32


class ProblemOccurrenceQueryCursorError(Exception):
    """Raised when an occurrence continuation cursor is invalid or query-mismatched."""


class ProblemOccurrenceCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["intergrax.diagnostic_problem_occurrence_cursor.v1"]
    tenant_id: str
    problem_id: str
    store_cursor: str


class ProblemOccurrenceQueryCursorCodec:
    """Authenticated, query-bound codec for ProblemOccurrence continuation cursors."""

    def __init__(self, *, secret: bytes) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("problem_occurrence_cursor_secret_invalid")
        if len(secret) < _MIN_OCCURRENCE_CURSOR_SECRET_BYTES:
            raise ValueError("problem_occurrence_cursor_secret_too_short")
        self._secret = secret

    @staticmethod
    def _canonical_payload(payload: ProblemOccurrenceCursorPayloadV1) -> bytes:
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
        problem_id: ProblemId,
        store_cursor: str,
    ) -> str:
        payload = ProblemOccurrenceCursorPayloadV1(
            schema_version=_OCCURRENCE_CURSOR_SCHEMA,
            tenant_id=tenant_id,
            problem_id=str(problem_id),
            store_cursor=store_cursor,
        )
        canonical = self._canonical_payload(payload)
        signature = hmac.new(self._secret, canonical, digestmod="sha256").digest()
        envelope = {
            "payload": self._encode_base64url(canonical),
            "signature": self._encode_base64url(signature),
        }
        return self._encode_base64url(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )

    def decode(
        self,
        cursor: str,
        *,
        tenant_id: str,
        problem_id: ProblemId,
    ) -> str:
        if not isinstance(cursor, str) or not cursor:
            raise ProblemOccurrenceQueryCursorError("occurrence cursor is required")
        if len(cursor) > _OCCURRENCE_CURSOR_MAX_TOKEN_LENGTH:
            raise ProblemOccurrenceQueryCursorError("occurrence cursor too long")
        try:
            envelope_bytes = self._decode_base64url(cursor)
            envelope = json.loads(envelope_bytes.decode("utf-8"))
            payload_b64 = envelope["payload"]
            signature_b64 = envelope["signature"]
            payload_bytes = self._decode_base64url(payload_b64)
            provided_signature = self._decode_base64url(signature_b64)
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            binascii.Error,
        ) as exc:
            raise ProblemOccurrenceQueryCursorError("invalid occurrence cursor") from exc

        expected_signature = hmac.new(
            self._secret,
            payload_bytes,
            digestmod="sha256",
        ).digest()
        if not hmac.compare_digest(provided_signature, expected_signature):
            raise ProblemOccurrenceQueryCursorError("occurrence cursor signature mismatch")

        try:
            payload = ProblemOccurrenceCursorPayloadV1.model_validate_json(payload_bytes)
        except ValueError as exc:
            raise ProblemOccurrenceQueryCursorError(
                "invalid occurrence cursor payload",
            ) from exc

        if payload.schema_version != _OCCURRENCE_CURSOR_SCHEMA:
            raise ProblemOccurrenceQueryCursorError(
                "unsupported occurrence cursor schema",
            )
        if payload.tenant_id != tenant_id:
            raise ProblemOccurrenceQueryCursorError(
                "occurrence cursor tenant_id mismatch",
            )
        if payload.problem_id != str(problem_id):
            raise ProblemOccurrenceQueryCursorError(
                "occurrence cursor problem_id mismatch",
            )
        return payload.store_cursor
