from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.identity import (
    LIVE_CONTRACT_VERSION,
    validate_capability_identity,
)
from intergrax.utils import attribute_access

HARD_MAX_LIVE_CALLS = 50
HARD_MAX_TOTAL_DURATION_MS = 300_000
HARD_MAX_RESULT_ITEMS = 500
HARD_MAX_RESULT_BYTES = 16_777_216
HARD_MAX_PROVIDER_PAGES = 100
HARD_MAX_PROVIDER_REQUESTS = 100
HARD_MAX_UPSTREAM_ITEMS = 10_000
HARD_MAX_PROVIDER_PAGE_SIZE = 100
HARD_MAX_CONTENT_BYTES_PER_ITEM = 1_048_576
MAX_SAFE_LOCATOR_LENGTH = 2_048

_FORBIDDEN_LOCATOR_TERMS = re.compile(
    r"(?:access[_-]?token|refresh[_-]?token|authorization|bearer|"
    r"credential|password|secret|api[_-]?key|continuation|cursor|"
    r"presign|signature|x-amz|sig=)",
    re.IGNORECASE,
)
_FORBIDDEN_LOCATOR_QUERY_KEYS = frozenset(
    {
        "access_token",
        "refresh_token",
        "authorization",
        "bearer",
        "credential",
        "password",
        "secret",
        "api_key",
        "sig",
        "signature",
        "x_amz_signature",
        "x_amz_credential",
        "continuation",
        "cursor",
    }
)


class LiveResultRetentionV1(StrEnum):
    EPHEMERAL = "ephemeral"
    RECEIPT_ONLY = "receipt_only"


class KnowledgeQueryAudienceV1(StrEnum):
    PERSONAL = "personal"
    SHARED = "shared"


def _nonblank(value: str, field_name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_invalid")
    cleaned = value.strip()
    if not cleaned or cleaned != value or len(cleaned) > maximum:
        raise ValueError(f"{field_name}_invalid")
    return cleaned


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    return value


def _positive(value: int, field_name: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > maximum:
        raise ValueError(f"{field_name}_must_be_positive_and_finite")
    return value


class EffectiveLiveCallBudgetV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    max_live_calls: int = Field(..., ge=0, le=HARD_MAX_LIVE_CALLS)
    max_total_duration_ms: int = Field(..., ge=1, le=HARD_MAX_TOTAL_DURATION_MS)
    deadline_monotonic: float | None = Field(default=None, ge=0)
    max_result_items: int = Field(..., ge=1, le=HARD_MAX_RESULT_ITEMS)
    max_result_bytes: int = Field(..., ge=1, le=HARD_MAX_RESULT_BYTES)
    max_provider_pages: int = Field(
        HARD_MAX_PROVIDER_PAGES, ge=1, le=HARD_MAX_PROVIDER_PAGES
    )
    max_provider_requests: int = Field(
        HARD_MAX_PROVIDER_REQUESTS, ge=1, le=HARD_MAX_PROVIDER_REQUESTS
    )
    max_upstream_items: int = Field(
        HARD_MAX_UPSTREAM_ITEMS, ge=1, le=HARD_MAX_UPSTREAM_ITEMS
    )
    max_provider_page_size: int = Field(
        HARD_MAX_PROVIDER_PAGE_SIZE, ge=1, le=HARD_MAX_PROVIDER_PAGE_SIZE
    )
    max_content_bytes_per_item: int = Field(
        HARD_MAX_CONTENT_BYTES_PER_ITEM,
        ge=1,
        le=HARD_MAX_CONTENT_BYTES_PER_ITEM,
    )


class ValidatedLiveCapabilityCallV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    call_id: str = Field(..., min_length=1, max_length=128)
    capability_id: str = Field(..., min_length=1, max_length=128)
    contract_version: str = Field(..., min_length=1, max_length=32)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    validated_request: BaseModel
    effective_budget: EffectiveLiveCallBudgetV1
    audience_context_ref: str | None = Field(default=None, max_length=128)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    source_kind: str = Field(..., min_length=1, max_length=64)

    @field_validator("call_id", "capability_id", "connection_ref", "live_access_binding_id")
    @classmethod
    def _valid_ids(cls, value: str, info) -> str:
        return _nonblank(value, info.field_name, 128)

    @field_validator("provider_id")
    @classmethod
    def _valid_provider(cls, value: str) -> str:
        return _nonblank(value, "provider_id", 64)

    @field_validator("source_kind")
    @classmethod
    def _valid_source(cls, value: str) -> str:
        return _nonblank(value, "source_kind", 64)

    @field_validator("contract_version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if value != LIVE_CONTRACT_VERSION:
            raise ValueError("live_contract_version_unsupported")
        return value

    def assert_identity(self) -> None:
        validate_capability_identity(
            capability_id=self.capability_id,
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            source_kind=self.source_kind,
            contract_version=self.contract_version,
        )


class LiveExecutionOutcomeV1(StrEnum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    FAILED = "failed"


class LiveCapabilityExecutionContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    audience: KnowledgeQueryAudienceV1
    started_at: datetime
    deadline_monotonic: float = Field(..., ge=0)
    retention: LiveResultRetentionV1

    _validate_ids = field_validator("run_id", "tenant_id", "workspace_id")(
        lambda value, info: _nonblank(value, info.field_name, 128)
    )
    _validate_started_at = field_validator("started_at")(
        lambda value: _aware(value, "started_at")
    )


class LiveCapabilityResultItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    remote_item_id: str = Field(..., min_length=1, max_length=512)
    safe_display_name: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., max_length=16_777_216)
    content_hash: str = Field(..., min_length=64, max_length=64)
    retrieved_at: datetime
    remote_updated_at: datetime | None = None
    safe_locator: str | None = Field(default=None, max_length=2048)
    truncated: bool = False

    _validate_ids = field_validator("remote_item_id", "safe_display_name")(
        lambda value, info: _nonblank(value, info.field_name, 512)
    )
    _validate_retrieved_at = field_validator("retrieved_at")(
        lambda value: _aware(value, "retrieved_at")
    )
    _validate_remote_updated_at = field_validator("remote_updated_at")(
        lambda value: None if value is None else _aware(value, "remote_updated_at")
    )
    _validate_locator = field_validator("safe_locator")(
        lambda value: safe_locator_or_none(value)
    )


class LiveExecutionReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    call_id: str = Field(..., min_length=1)
    live_access_binding_id: str = Field(..., min_length=1)
    provider_id: str = Field(..., min_length=1)
    source_kind: str = Field(..., min_length=1)
    capability_id: str = Field(..., min_length=1)
    contract_version: str = Field(..., min_length=1)
    started_at: datetime
    completed_at: datetime
    item_count: int = Field(..., ge=0)
    byte_count: int = Field(..., ge=0)
    result_hash: str = Field(..., min_length=64, max_length=64)
    truncated: bool = False
    normalized_outcome: str = Field(..., min_length=1)
    error_code: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_content_hash_adapter(cls, data: Any) -> Any:
        if isinstance(data, dict) and "result_hash" not in data and "content_hash" in data:
            migrated = dict(data)
            migrated["result_hash"] = migrated.pop("content_hash")
            return migrated
        return data

    @field_validator("started_at", "completed_at")
    @classmethod
    def _timestamps_aware(cls, value: datetime) -> datetime:
        return _aware(value, "receipt_timestamp")

    @field_validator("result_hash")
    @classmethod
    def _sha256_result_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("receipt_result_hash_invalid")
        return value


class LiveCapabilityExecutionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    call_id: str = Field(..., min_length=1, max_length=128)
    normalized_outcome: LiveExecutionOutcomeV1
    items: tuple[LiveCapabilityResultItemV1, ...] = ()
    item_count: int = Field(..., ge=0)
    byte_count: int = Field(..., ge=0)
    started_at: datetime
    completed_at: datetime
    truncated: bool = False
    error_code: str | None = None
    receipt: LiveExecutionReceiptV1 | None = None
    provider_id: str | None = None
    integration_kind: IntegrationCategory | None = None
    source_kind: str | None = None
    capability_id: str | None = None
    contract_version: str | None = None
    live_access_binding_id: str | None = None
    connection_ref: str | None = None
    remote_resource_id: str | None = None

    _validate_call_id = field_validator("call_id")(
        lambda value: _nonblank(value, "call_id", 128)
    )
    _validate_timestamps = field_validator("started_at", "completed_at")(
        lambda value, info: _aware(value, info.field_name)
    )
    _validate_error_code = field_validator("error_code")(
        lambda value: None if value is None else _nonblank(value, "error_code", 128)
    )


@runtime_checkable
class LiveCapabilityHandlerV1(Protocol):
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    capability_id: str
    contract_version: str
    request_schema_ref: str
    result_schema_ref: str
    expected_request_model: type[BaseModel]

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context: LiveCapabilityExecutionContextV1,
    ) -> LiveCapabilityExecutionResultV1:
        ...


def _locator_has_forbidden_query(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.username is not None or parsed.password is not None:
        return True
    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized = key.strip().lower().replace("-", "_")
        if normalized in _FORBIDDEN_LOCATOR_QUERY_KEYS:
            return True
    return False


def safe_locator_or_none(value: str | None) -> str | None:
    """Return a bounded, secret-free locator; unsafe input is removed silently."""

    if value is None or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > MAX_SAFE_LOCATOR_LENGTH
        or any(ord(char) < 32 or ord(char) == 127 for char in cleaned)
        or _FORBIDDEN_LOCATOR_TERMS.search(cleaned) is not None
    ):
        return None
    parsed = urlparse(cleaned)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            return None
        if _locator_has_forbidden_query(cleaned):
            return None
    return cleaned


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def evidence_id_for_call(
    *,
    provider_id: str,
    integration_kind: IntegrationCategory | str,
    source_kind: str,
    capability_id: str,
    contract_version: str,
    live_access_binding_id: str,
    connection_ref: str,
    remote_resource_id: str | None,
    call_id: str,
    remote_item_id: str,
) -> str:
    return "live:" + canonical_sha256(
        {
            "provider_id": provider_id,
            "integration_kind": (
                integration_kind.value
                if isinstance(integration_kind, IntegrationCategory)
                else integration_kind
            ),
            "source_kind": source_kind,
            "capability_id": capability_id,
            "contract_version": contract_version,
            "live_access_binding_id": live_access_binding_id,
            "connection_ref": connection_ref,
            "remote_resource_id": remote_resource_id,
            "call_id": call_id,
            "remote_item_id": remote_item_id,
        }
    )


def result_hash_for_items(
    *,
    items: Iterable[object],
    normalized_outcome: str,
    error_code: str | None,
    item_count: int,
    byte_count: int,
) -> str:
    ordered_items = [
        {
            "position": position,
            "remote_item_id": attribute_access.optional(item, "remote_item_id", None),
            "content_hash": attribute_access.optional(item, "content_hash", None),
            "truncated": bool(attribute_access.optional_bool(item, "truncated")),
        }
        for position, item in enumerate(items)
    ]
    return canonical_sha256(
        {
            "items": ordered_items,
            "normalized_outcome": normalized_outcome,
            "error_code": error_code,
            "item_count": item_count,
            "byte_count": byte_count,
        }
    )


def immutable_mapping(
    values: Mapping[str, object],
) -> Mapping[str, object]:
    return MappingProxyType(dict(values))
