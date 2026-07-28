# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Drive knowledge-read: delta inventory for one known drive ID."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_collection_page,
    validate_msgraph_continuation_url,
)

MSGRAPH_DRIVE_SOURCE_KIND = "drive"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_DRIVE_RESPONSE = "unexpected Microsoft Graph drive response"
_INVALID_DRIVE_CONTINUATION = "invalid Microsoft Graph drive continuation"
_INVALID_DRIVE_LIMIT = "invalid Microsoft Graph drive delta page limit"
_MAX_MSGRAPH_ID_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MIN_DELTA_LIMIT = 1
_MAX_DELTA_LIMIT = 200

_DRIVE_SELECT = (
    "id,name,parentReference,webUrl,eTag,cTag,size,file,folder,package,deleted,root,"
    "createdDateTime,lastModifiedDateTime"
)

class MsGraphDriveItemKind(StrEnum):
    FILE = "file"
    FOLDER = "folder"
    PACKAGE = "package"
    OTHER = "other"
    DELETED = "deleted"


def validate_msgraph_drive_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_drive_item_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def _validate_msgraph_opaque_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    if len(trimmed) > _MAX_MSGRAPH_ID_LEN:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return trimmed


def _parse_timezone_aware_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        return None
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _parse_size_bytes(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    if value < 0:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return value


def _normalize_model_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return value.astimezone(timezone.utc)


def _parse_optional_provider_string(mapping: dict[str, object], key: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return trimmed


def _parse_required_provider_string(mapping: dict[str, object], key: str) -> str:
    if key not in mapping:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    parsed = _parse_optional_provider_string(mapping, key)
    if parsed is None:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return parsed


def _facet_is_present(payload: dict[str, object], key: str) -> bool:
    if key not in payload:
        return False
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)
    return True


def _parse_parent_reference(payload: dict[str, object], validated_drive_id: str) -> str | None:
    parent_reference = payload.get("parentReference")
    if parent_reference is None:
        return None
    if not isinstance(parent_reference, dict):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE)

    if "driveId" in parent_reference:
        try:
            parent_drive_id = validate_msgraph_drive_id(parent_reference["driveId"])
        except ValueError:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None
        if parent_drive_id != validated_drive_id:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    if "id" not in parent_reference:
        return None

    try:
        return validate_msgraph_drive_item_id(parent_reference["id"])
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None


class MsGraphDriveItem(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    remote_id: str
    drive_id: str
    parent_remote_id: str | None = None

    kind: MsGraphDriveItemKind
    name: str | None = None

    e_tag: str | None = Field(default=None, repr=False)
    c_tag: str | None = Field(default=None, repr=False)

    size_bytes: int | None = None
    mime_type: str | None = None

    created_at: datetime | None = None
    last_modified_at: datetime | None = None

    web_url: str | None = Field(default=None, repr=False)
    is_root: bool = False

    deleted_state: str | None = None

    @field_validator("remote_id", "drive_id", mode="before")
    @classmethod
    def _validate_required_ids(cls, value: object) -> str:
        return validate_msgraph_drive_id(value)

    @field_validator("parent_remote_id", mode="before")
    @classmethod
    def _validate_parent_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_drive_item_id(value)

    @field_validator(
        "name",
        "e_tag",
        "c_tag",
        "mime_type",
        "web_url",
        "deleted_state",
        mode="before",
    )
    @classmethod
    def _validate_optional_string_fields(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return trimmed

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size(cls, value: object) -> int | None:
        return _parse_size_bytes(value)

    @field_validator("created_at", "last_modified_at", mode="before")
    @classmethod
    def _validate_datetime_fields(cls, value: object) -> datetime | None:
        return _normalize_model_datetime(value)

    @model_validator(mode="after")
    def _validate_item_shape(self) -> MsGraphDriveItem:
        if self.kind == MsGraphDriveItemKind.DELETED:
            return self

        if self.kind not in {
            MsGraphDriveItemKind.FILE,
            MsGraphDriveItemKind.FOLDER,
            MsGraphDriveItemKind.PACKAGE,
            MsGraphDriveItemKind.OTHER,
        }:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        if self.name is None:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        if self.last_modified_at is None:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return self


class MsGraphDriveDeltaPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[MsGraphDriveItem, ...]
    continuation: MsGraphKnowledgeContinuation = Field(repr=False)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items_input(cls, value: object) -> object:
        if not isinstance(value, tuple):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphDriveItem):
                raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation_input(cls, value: object) -> object:
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        if value.kind not in {
            MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            MsGraphKnowledgeContinuationKind.DELTA,
        }:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphDriveDeltaPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE

    @property
    def is_complete(self) -> bool:
        return self.continuation.kind == MsGraphKnowledgeContinuationKind.DELTA


def _safe_construct_drive_item(**kwargs: object) -> MsGraphDriveItem:
    try:
        return MsGraphDriveItem(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None


def _safe_construct_drive_delta_page(**kwargs: object) -> MsGraphDriveDeltaPage:
    try:
        return MsGraphDriveDeltaPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None


@runtime_checkable
class MsGraphDriveKnowledgeReadClient(Protocol):
    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphDriveDeltaPage:
        ...


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _extract_drive_id_from_delta_path(path: str, *, graph_base_path: str) -> str | None:
    normalized = path.rstrip("/") or "/"
    expected_prefix = f"{graph_base_path.rstrip('/')}/drives/"
    expected_suffix = "/root/delta"

    if not normalized.startswith(expected_prefix):
        return None
    if not normalized.endswith(expected_suffix):
        return None

    drive_segment = normalized[len(expected_prefix) : -len(expected_suffix)]
    if not drive_segment or "/" in drive_segment:
        return None
    return unquote(drive_segment)


def validate_msgraph_drive_delta_continuation(
    continuation: object,
    *,
    drive_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_DRIVE_CONTINUATION) from None
    if continuation.kind not in {
        MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        MsGraphKnowledgeContinuationKind.DELTA,
    }:
        raise IntegrationConfigurationError(_INVALID_DRIVE_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            continuation.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_DRIVE_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted_drive_id = _extract_drive_id_from_delta_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted_drive_id is None:
        raise IntegrationConfigurationError(_INVALID_DRIVE_CONTINUATION) from None

    try:
        validated_drive_id = validate_msgraph_drive_id(drive_id)
        validated_extracted_drive_id = validate_msgraph_drive_id(extracted_drive_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_DRIVE_CONTINUATION) from None

    if validated_extracted_drive_id != validated_drive_id:
        raise IntegrationConfigurationError(_INVALID_DRIVE_CONTINUATION) from None

    return continuation


def parse_msgraph_drive_item(
    payload: object,
    *,
    expected_drive_id: str,
) -> MsGraphDriveItem:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    try:
        validated_drive_id = validate_msgraph_drive_id(expected_drive_id)
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    try:
        remote_id = validate_msgraph_drive_item_id(payload.get("id"))
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    try:
        parent_remote_id = _parse_parent_reference(payload, validated_drive_id)
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    try:
        has_file = _facet_is_present(payload, "file")
        has_folder = _facet_is_present(payload, "folder")
        has_package = _facet_is_present(payload, "package")
        has_deleted = _facet_is_present(payload, "deleted")
        is_root = _facet_is_present(payload, "root")
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    type_facet_count = sum((has_file, has_folder, has_package, has_deleted))
    if type_facet_count > 1:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

    if has_deleted:
        kind = MsGraphDriveItemKind.DELETED
    elif has_folder:
        kind = MsGraphDriveItemKind.FOLDER
    elif has_package:
        kind = MsGraphDriveItemKind.PACKAGE
    elif has_file:
        kind = MsGraphDriveItemKind.FILE
    else:
        kind = MsGraphDriveItemKind.OTHER

    try:
        if kind == MsGraphDriveItemKind.DELETED:
            deleted_obj = payload["deleted"]
            if not isinstance(deleted_obj, dict):
                raise ValueError(_MALFORMED_DRIVE_RESPONSE)
            deleted_state = _parse_optional_provider_string(deleted_obj, "state")
            name = _parse_optional_provider_string(payload, "name")
            e_tag = _parse_optional_provider_string(payload, "eTag")
            c_tag = _parse_optional_provider_string(payload, "cTag")
            web_url = _parse_optional_provider_string(payload, "webUrl")
            size_bytes = _parse_size_bytes(payload.get("size")) if "size" in payload else None
            created_at = (
                _parse_timezone_aware_datetime(payload.get("createdDateTime"))
                if "createdDateTime" in payload
                else None
            )
            last_modified_at = (
                _parse_timezone_aware_datetime(payload.get("lastModifiedDateTime"))
                if "lastModifiedDateTime" in payload
                else None
            )
            mime_type = None
            return _safe_construct_drive_item(
                remote_id=remote_id,
                drive_id=validated_drive_id,
                parent_remote_id=parent_remote_id,
                kind=kind,
                name=name,
                e_tag=e_tag,
                c_tag=c_tag,
                size_bytes=size_bytes,
                mime_type=mime_type,
                created_at=created_at,
                last_modified_at=last_modified_at,
                web_url=web_url,
                is_root=is_root,
                deleted_state=deleted_state,
            )

        name = _parse_required_provider_string(payload, "name")
        e_tag = _parse_optional_provider_string(payload, "eTag")
        c_tag = _parse_optional_provider_string(payload, "cTag")
        web_url = _parse_optional_provider_string(payload, "webUrl")
        size_bytes = _parse_size_bytes(payload.get("size")) if "size" in payload else None

        mime_type: str | None = None
        if kind == MsGraphDriveItemKind.FILE:
            file_obj = payload["file"]
            if not isinstance(file_obj, dict):
                raise ValueError(_MALFORMED_DRIVE_RESPONSE)
            mime_type = _parse_optional_provider_string(file_obj, "mimeType")

        created_at = _parse_timezone_aware_datetime(payload.get("createdDateTime"))
        last_modified_at = _parse_timezone_aware_datetime(payload.get("lastModifiedDateTime"))
        if last_modified_at is None:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE)

        return _safe_construct_drive_item(
            remote_id=remote_id,
            drive_id=validated_drive_id,
            parent_remote_id=parent_remote_id,
            kind=kind,
            name=name,
            e_tag=e_tag,
            c_tag=c_tag,
            size_bytes=size_bytes,
            mime_type=mime_type,
            created_at=created_at,
            last_modified_at=last_modified_at,
            web_url=web_url,
            is_root=is_root,
        )
    except ValueError:
        raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None


def _deduplicate_drive_items(items: tuple[MsGraphDriveItem, ...]) -> tuple[MsGraphDriveItem, ...]:
    last_by_id: dict[str, MsGraphDriveItem] = {}
    order: list[str] = []
    for item in items:
        if item.remote_id not in last_by_id:
            order.append(item.remote_id)
        else:
            order.remove(item.remote_id)
            order.append(item.remote_id)
        last_by_id[item.remote_id] = item
    return tuple(last_by_id[remote_id] for remote_id in order)


def _validate_delta_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_DRIVE_LIMIT)
    if limit < _MIN_DELTA_LIMIT or limit > _MAX_DELTA_LIMIT:
        raise IntegrationConfigurationError(_INVALID_DRIVE_LIMIT)
    return limit


def _build_drive_delta_page(
    *,
    raw_items: tuple[dict[str, object], ...],
    continuation: MsGraphKnowledgeContinuation,
    expected_drive_id: str,
) -> MsGraphDriveDeltaPage:
    parsed_items = tuple(
        parse_msgraph_drive_item(item, expected_drive_id=expected_drive_id) for item in raw_items
    )
    deduplicated = _deduplicate_drive_items(parsed_items)
    return _safe_construct_drive_delta_page(items=deduplicated, continuation=continuation)


class MsGraphDriveKnowledgeReader:
    """Drive-scoped delta reader over the shared Graph knowledge transport."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphDriveDeltaPage:
        validated_drive_id = validate_msgraph_drive_id(drive_id)
        validated_limit = _validate_delta_limit(limit)

        if continuation is None:
            quoted_drive_id = quote(validated_drive_id, safe="")
            path = f"/drives/{quoted_drive_id}/root/delta"
            payload = self._transport.get_initial_json(
                path=path,
                params={
                    "$top": validated_limit,
                    "$select": _DRIVE_SELECT,
                },
            )
        else:
            validated_continuation = validate_msgraph_drive_delta_continuation(
                continuation,
                drive_id=validated_drive_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=True,
        )
        if collection_page.continuation is None:
            raise ValueError(_MALFORMED_DRIVE_RESPONSE) from None

        return _build_drive_delta_page(
            raw_items=collection_page.items,
            continuation=collection_page.continuation,
            expected_drive_id=validated_drive_id,
        )
