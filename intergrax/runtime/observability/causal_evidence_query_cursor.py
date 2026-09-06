# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authenticated query-bound cursor codec for causal evidence paging (DG-002 R1)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.contracts.execution_identity import RunId, TaskId

_CAUSAL_EVIDENCE_CURSOR_SCHEMA = "causal_evidence.cursor.v1"
_CAUSAL_EVIDENCE_CURSOR_MAX_TOKEN_LENGTH = 4096
_MIN_CAUSAL_EVIDENCE_CURSOR_SECRET_BYTES = 32


class CausalEvidenceQueryCursorError(Exception):
    """Raised when a causal evidence continuation cursor is invalid or query-mismatched."""


class CausalEvidenceQueryCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["causal_evidence.cursor.v1"]
    query_kind: Literal["execution", "transport"]
    tenant_id: str
    task_id: str | None
    run_id: str | None
    provider: str | None
    transport_task_id: str | None
    high_water: str | None
    last_recorded_at: datetime | None
    last_evidence_id: str | None
    store_cursor: str | None


class CausalEvidenceQueryCursorCodec:
    """Authenticated, query-bound codec for causal evidence continuation cursors."""

    def __init__(self, *, secret: bytes) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("causal_evidence_cursor_secret_invalid")
        if len(secret) < _MIN_CAUSAL_EVIDENCE_CURSOR_SECRET_BYTES:
            raise ValueError("causal_evidence_cursor_secret_too_short")
        self._secret = secret

    @staticmethod
    def _canonical_payload(payload: CausalEvidenceQueryCursorPayloadV1) -> bytes:
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

    def encode_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        high_water: str | None,
        last_recorded_at: datetime | None,
        last_evidence_id: str | None,
        store_cursor: str | None,
    ) -> str:
        payload = CausalEvidenceQueryCursorPayloadV1(
            schema_version=_CAUSAL_EVIDENCE_CURSOR_SCHEMA,
            query_kind="execution",
            tenant_id=tenant_id,
            task_id=str(task_id),
            run_id=str(run_id),
            provider=None,
            transport_task_id=None,
            high_water=high_water,
            last_recorded_at=last_recorded_at,
            last_evidence_id=last_evidence_id,
            store_cursor=store_cursor,
        )
        return self._encode_payload(payload)

    def encode_transport(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        high_water: str | None,
        last_recorded_at: datetime | None,
        last_evidence_id: str | None,
        store_cursor: str | None,
    ) -> str:
        payload = CausalEvidenceQueryCursorPayloadV1(
            schema_version=_CAUSAL_EVIDENCE_CURSOR_SCHEMA,
            query_kind="transport",
            tenant_id=tenant_id,
            task_id=None,
            run_id=None,
            provider=provider,
            transport_task_id=transport_task_id,
            high_water=high_water,
            last_recorded_at=last_recorded_at,
            last_evidence_id=last_evidence_id,
            store_cursor=store_cursor,
        )
        return self._encode_payload(payload)

    def _encode_payload(self, payload: CausalEvidenceQueryCursorPayloadV1) -> str:
        canonical = self._canonical_payload(payload)
        signature = hmac.new(self._secret, canonical, digestmod="sha256").digest()
        envelope = {
            "payload": self._encode_base64url(canonical),
            "signature": self._encode_base64url(signature),
        }
        encoded = self._encode_base64url(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        if len(encoded) > _CAUSAL_EVIDENCE_CURSOR_MAX_TOKEN_LENGTH:
            raise CausalEvidenceQueryCursorError("causal evidence cursor too long")
        return encoded

    def decode_execution(
        self,
        cursor: str,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> CausalEvidenceQueryCursorPayloadV1:
        payload = self._decode(cursor)
        if (
            payload.query_kind != "execution"
            or payload.tenant_id != tenant_id
            or payload.task_id != str(task_id)
            or payload.run_id != str(run_id)
        ):
            raise CausalEvidenceQueryCursorError("causal evidence cursor query mismatch")
        return payload

    def decode_transport(
        self,
        cursor: str,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
    ) -> CausalEvidenceQueryCursorPayloadV1:
        payload = self._decode(cursor)
        if (
            payload.query_kind != "transport"
            or payload.tenant_id != tenant_id
            or payload.provider != provider
            or payload.transport_task_id != transport_task_id
        ):
            raise CausalEvidenceQueryCursorError("causal evidence cursor query mismatch")
        return payload

    def _decode(self, cursor: str) -> CausalEvidenceQueryCursorPayloadV1:
        if not isinstance(cursor, str) or not cursor:
            raise CausalEvidenceQueryCursorError("causal evidence cursor is required")
        if len(cursor) > _CAUSAL_EVIDENCE_CURSOR_MAX_TOKEN_LENGTH:
            raise CausalEvidenceQueryCursorError("causal evidence cursor too long")
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
            raise CausalEvidenceQueryCursorError("invalid causal evidence cursor") from exc

        try:
            payload = CausalEvidenceQueryCursorPayloadV1.model_validate_json(
                payload_bytes.decode("utf-8"),
                strict=True,
            )
        except ValidationError as exc:
            raise CausalEvidenceQueryCursorError("invalid causal evidence cursor") from exc

        expected_signature = hmac.new(
            self._secret,
            payload_bytes,
            digestmod="sha256",
        ).digest()
        if not hmac.compare_digest(provided_signature, expected_signature):
            raise CausalEvidenceQueryCursorError("causal evidence cursor signature invalid")
        return payload


__all__ = [
    "CausalEvidenceQueryCursorCodec",
    "CausalEvidenceQueryCursorError",
    "CausalEvidenceQueryCursorPayloadV1",
]
