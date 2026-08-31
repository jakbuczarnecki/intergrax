# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded Problem list query contract and cursor codec (DIAG-ENTERPRISE-1)."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId, ProblemStatus

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.problem_lifecycle import Problem

_LIST_INDEX_SCHEMA = "intergrax.diagnostic_problem.list_index.v1"
_LIST_ROW_PREFIX = "list:"
_LIST_SCOPE_ALL = "all"
_MAX_LIST_INDEX_MICROS = 10**16
_MIN_LIST_INDEX_MICROS = 0
_LIST_CURSOR_SCHEMA = "intergrax.diagnostic_problem.list_cursor.v1"
_LIST_CURSOR_MAX_TOKEN_LENGTH = 4096
_PROBLEM_ID_FIELD = "problem_id"
_LAST_SEEN_AT_FIELD = "last_seen_at"
_STATUS_FIELD = "status"
_RECORD_VERSION_FIELD = "record_version"
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class ProblemListQueryCursorError(Exception):
    """Raised when a Problem list continuation cursor is invalid or query-mismatched."""


class ProblemListIndexTimestampError(ValueError):
    """Raised when ``last_seen_at`` is outside the supported list-index timestamp range."""


class ProblemListScope(StrEnum):
    """Derived read-index scope for tenant Problem listing."""

    ALL = _LIST_SCOPE_ALL
    OPEN = ProblemStatus.OPEN.value
    RESOLVED = ProblemStatus.RESOLVED.value


class ProblemListCursorPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["intergrax.diagnostic_problem.list_cursor.v1"]
    tenant_id: str
    status_filter: Literal["all", "open", "resolved"]
    store_cursor: str


class ProblemListQueryCursorCodec:
    """Authenticated, query-bound codec for Problem list continuation cursors."""

    def __init__(self, *, secret: bytes) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("problem_list_cursor_secret_invalid")
        self._secret = secret

    @staticmethod
    def _canonical_payload(payload: ProblemListCursorPayloadV1) -> bytes:
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
        status_filter: ProblemListScope,
        store_cursor: str,
    ) -> str:
        payload = ProblemListCursorPayloadV1(
            schema_version=_LIST_CURSOR_SCHEMA,
            tenant_id=tenant_id,
            status_filter=_scope_to_status_filter(status_filter),
            store_cursor=store_cursor,
        )
        canonical = self._canonical_payload(payload)
        mac = hmac.new(self._secret, canonical, digestmod="sha256").digest()
        envelope = {
            "mac": self._encode_base64url(mac),
            "payload": payload.model_dump(mode="json"),
        }
        encoded = self._encode_base64url(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if len(encoded) > _LIST_CURSOR_MAX_TOKEN_LENGTH:
            raise ProblemListQueryCursorError("problem_list_cursor_invalid")
        return encoded

    def decode(
        self,
        cursor: str,
        *,
        tenant_id: str,
        status_filter: ProblemListScope,
    ) -> str:
        if not isinstance(cursor, str) or not cursor:
            raise ProblemListQueryCursorError("problem_list_cursor_invalid")
        if len(cursor) > _LIST_CURSOR_MAX_TOKEN_LENGTH:
            raise ProblemListQueryCursorError("problem_list_cursor_invalid")
        try:
            envelope_bytes = self._decode_base64url(cursor)
            envelope = json.loads(envelope_bytes.decode("utf-8"))
            payload_raw = envelope["payload"]
            mac_raw = envelope["mac"]
            if not isinstance(payload_raw, dict) or not isinstance(mac_raw, str):
                raise ValueError
            payload = ProblemListCursorPayloadV1.model_validate(payload_raw)
            mac = self._decode_base64url(mac_raw)
        except (
            ValueError,
            KeyError,
            json.JSONDecodeError,
            binascii.Error,
            UnicodeDecodeError,
        ) as exc:
            raise ProblemListQueryCursorError("problem_list_cursor_invalid") from exc

        expected_mac = hmac.new(
            self._secret,
            self._canonical_payload(payload),
            digestmod="sha256",
        ).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise ProblemListQueryCursorError("problem_list_cursor_authentication_failed")

        if payload.schema_version != _LIST_CURSOR_SCHEMA:
            raise ProblemListQueryCursorError("problem_list_cursor_invalid")
        if payload.tenant_id != tenant_id:
            raise ProblemListQueryCursorError("problem_list_cursor_tenant_mismatch")
        if payload.status_filter != _scope_to_status_filter(status_filter):
            raise ProblemListQueryCursorError("problem_list_cursor_status_mismatch")
        return payload.store_cursor


def problem_list_scope_for_status(status: ProblemStatus | None) -> ProblemListScope:
    if status is None:
        return ProblemListScope.ALL
    if status is ProblemStatus.OPEN:
        return ProblemListScope.OPEN
    if status is ProblemStatus.RESOLVED:
        return ProblemListScope.RESOLVED
    raise TypeError(f"status must be ProblemStatus or None, got {type(status).__name__}")


def problem_list_row_key_prefix(scope: ProblemListScope) -> str:
    return f"{_LIST_ROW_PREFIX}{scope.value}:"


def problem_list_order_key(problem: Problem) -> tuple[float, str]:
    return (-problem.last_seen_at.timestamp(), str(problem.problem_id))


def sort_problems_for_public_list(problems: list[Problem]) -> tuple[Problem, ...]:
    problems.sort(key=problem_list_order_key)
    return tuple(problems)


def list_scopes_for_status(status: ProblemStatus) -> tuple[ProblemListScope, ...]:
    scope = ProblemListScope(status.value)
    return (ProblemListScope.ALL, scope)


def encode_list_index_data(
    *,
    problem_id: ProblemId,
    last_seen_at: datetime,
    status: ProblemStatus,
    record_version: int,
) -> dict[str, str]:
    _validate_list_index_record_version(record_version)
    _validate_list_index_timestamp(last_seen_at)
    return {
        "schema_version": _LIST_INDEX_SCHEMA,
        _PROBLEM_ID_FIELD: str(problem_id),
        _LAST_SEEN_AT_FIELD: _encode_datetime(last_seen_at),
        _STATUS_FIELD: status.value,
        _RECORD_VERSION_FIELD: str(record_version),
    }


def decode_list_index_data(
    data: object,
) -> tuple[ProblemId, datetime, ProblemStatus, int]:
    if not isinstance(data, dict):
        raise ValueError("invalid diagnostic problem list index")
    schema_version = data.get("schema_version")
    if schema_version != _LIST_INDEX_SCHEMA:
        raise ValueError("unsupported diagnostic problem list index schema")
    problem_id = data.get(_PROBLEM_ID_FIELD)
    last_seen_at = data.get(_LAST_SEEN_AT_FIELD)
    status = data.get(_STATUS_FIELD)
    record_version = data.get(_RECORD_VERSION_FIELD)
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("invalid diagnostic problem list index reference")
    if not isinstance(last_seen_at, str) or not last_seen_at:
        raise ValueError("invalid diagnostic problem list index timestamp")
    if status not in {ProblemStatus.OPEN.value, ProblemStatus.RESOLVED.value}:
        raise ValueError("invalid diagnostic problem list index status")
    if not isinstance(record_version, str) or not record_version.isdigit():
        raise ValueError("invalid diagnostic problem list index record_version")
    parsed_version = int(record_version)
    _validate_list_index_record_version(parsed_version)
    parsed_last_seen = _decode_datetime(last_seen_at)
    _validate_list_index_timestamp(parsed_last_seen)
    return ProblemId(problem_id), parsed_last_seen, ProblemStatus(status), parsed_version


def list_index_row_key(*, scope: ProblemListScope, problem: Problem) -> str:
    inverted_micros = _invert_last_seen_micros(problem.last_seen_at)
    return f"{problem_list_row_key_prefix(scope)}{inverted_micros:020d}:{problem.problem_id}"


def _scope_to_status_filter(scope: ProblemListScope) -> Literal["all", "open", "resolved"]:
    if scope is ProblemListScope.ALL:
        return "all"
    if scope is ProblemListScope.OPEN:
        return "open"
    return "resolved"


def _invert_last_seen_micros(last_seen_at: datetime) -> int:
    micros = _datetime_to_epoch_micros(last_seen_at)
    return _MAX_LIST_INDEX_MICROS - micros


def _datetime_to_epoch_micros(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = value.astimezone(UTC) - _UTC_EPOCH
    return (
        delta.days * 86_400 * 1_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _validate_list_index_timestamp(value: datetime) -> None:
    micros = _datetime_to_epoch_micros(value)
    if micros < _MIN_LIST_INDEX_MICROS or micros > _MAX_LIST_INDEX_MICROS:
        raise ProblemListIndexTimestampError(
            "diagnostic problem list index timestamp out of supported range",
        )


def _validate_list_index_record_version(record_version: int) -> None:
    if type(record_version) is not int or isinstance(record_version, bool) or record_version < 1:
        raise ValueError("invalid diagnostic problem list index record_version")


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _decode_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
