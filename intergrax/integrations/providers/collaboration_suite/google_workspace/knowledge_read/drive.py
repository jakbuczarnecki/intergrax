# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Drive knowledge-read: scoped inventory, metadata and incremental change feed."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspacePageToken,
    parse_google_workspace_collection_page,
)

GOOGLE_DRIVE_SOURCE_KIND = "drive"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)

_INVALID_SCOPE_MESSAGE = "invalid Google Drive scope"
_INVALID_IDENTIFIER_MESSAGE = "invalid Google Drive identifier"
_INVALID_PAGE_LIMIT_MESSAGE = "invalid Google Drive page limit"
_INVALID_PAGE_TOKEN_MESSAGE = "invalid Google Drive page token"
_UNEXPECTED_RESPONSE_MESSAGE = "unexpected Google Drive provider response"
_INCOMPLETE_INVENTORY_MESSAGE = "Google Drive inventory response is incomplete"

_MAX_DRIVE_ID_LENGTH = 1024
_MAX_NAME_LENGTH = 1024
_MAX_WEB_LINK_LENGTH = 4096
_MAX_CHECKSUM_LENGTH = 128
_MAX_REVISION_ID_LENGTH = 1024

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_SHARED_DRIVE_LIST_LIMIT_MIN = 1
_SHARED_DRIVE_LIST_LIMIT_MAX = 100
_ITEM_PAGE_LIMIT_MIN = 1
_ITEM_PAGE_LIMIT_MAX = 200

_FILE_FIELDS = (
    "id,name,mimeType,parents,driveId,webViewLink,createdTime,modifiedTime,"
    "size,md5Checksum,version,headRevisionId,trashed,"
    "shortcutDetails(targetId,targetMimeType),capabilities(canDownload)"
)
_INVENTORY_FIELDS = f"nextPageToken,incompleteSearch,files({_FILE_FIELDS})"
_SHARED_DRIVE_LIST_FIELDS = "nextPageToken,drives(id,name,createdTime,hidden)"
_CHANGE_FIELDS = (
    f"nextPageToken,newStartPageToken,changes("
    f"changeType,removed,fileId,time,driveId,file({_FILE_FIELDS}))"
)


class GoogleDriveScopeKind(StrEnum):
    USER = "user"
    SHARED_DRIVE = "shared_drive"


class GoogleDriveItemKind(StrEnum):
    BLOB = "blob"
    FOLDER = "folder"
    NATIVE_DOCUMENT = "native_document"
    SHORTCUT = "shortcut"
    OTHER = "other"


def _validate_drive_identifier(value: object) -> str:
    if type(value) is not str:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    if len(trimmed) > _MAX_DRIVE_ID_LENGTH:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    return trimmed


def _validate_nonblank_bounded_string(
    value: object,
    *,
    max_length: int,
    message: str,
) -> str:
    if type(value) is not str:
        raise ValueError(message)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(message)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(message)
    if len(trimmed) > max_length:
        raise ValueError(message)
    return trimmed


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return parsed.astimezone(timezone.utc)


def _normalize_model_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value.astimezone(timezone.utc)


def _parse_non_negative_decimal_string(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if type(value) is int:
        if value < 0:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value or any(ch.isspace() for ch in value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value[0] in {"+", "-"} or "." in value:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value.isdigit():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return int(value)


def _classify_item_kind(mime_type: str) -> GoogleDriveItemKind:
    if mime_type == "application/vnd.google-apps.folder":
        return GoogleDriveItemKind.FOLDER
    if mime_type == "application/vnd.google-apps.shortcut":
        return GoogleDriveItemKind.SHORTCUT
    if mime_type.startswith("application/vnd.google-apps."):
        return GoogleDriveItemKind.NATIVE_DOCUMENT
    return GoogleDriveItemKind.BLOB


def _require_exact_int(value: object, message: str) -> int:
    if type(value) is not int:
        raise IntegrationConfigurationError(message)
    return value


def _validate_page_limit(
    limit: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    validated = _require_exact_int(limit, _INVALID_PAGE_LIMIT_MESSAGE)
    if not minimum <= validated <= maximum:
        raise IntegrationConfigurationError(_INVALID_PAGE_LIMIT_MESSAGE)
    return validated


def _validate_optional_page_token(
    page_token: object,
) -> GoogleWorkspacePageToken | None:
    if page_token is None:
        return None
    if not isinstance(page_token, GoogleWorkspacePageToken):
        raise IntegrationConfigurationError(_INVALID_PAGE_TOKEN_MESSAGE)
    return page_token


def _validate_required_page_token(page_token: object) -> GoogleWorkspacePageToken:
    if not isinstance(page_token, GoogleWorkspacePageToken):
        raise IntegrationConfigurationError(_INVALID_PAGE_TOKEN_MESSAGE)
    return page_token


def _validate_scope(scope: object) -> GoogleDriveScope:
    if not isinstance(scope, GoogleDriveScope):
        raise IntegrationConfigurationError(_INVALID_SCOPE_MESSAGE)
    return scope


def _parse_provider_bool(mapping: dict[str, object], key: str) -> bool:
    if key not in mapping:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    value = mapping[key]
    if type(value) is not bool:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _parse_optional_provider_bool(mapping: dict[str, object], key: str) -> bool | None:
    if key not in mapping:
        return None
    return _parse_provider_bool(mapping, key)


def _parse_parent_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    seen: set[str] = set()
    parent_ids: list[str] = []
    for item in value:
        validated = _validate_drive_identifier(item)
        if validated in seen:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen.add(validated)
        parent_ids.append(validated)
    return tuple(parent_ids)


def _parse_optional_drive_id(value: object) -> str | None:
    if value is None:
        return None
    return _validate_drive_identifier(value)


def _validate_shared_drive_id_agreement(
    scope: GoogleDriveScope,
    *,
    change_drive_id: str | None,
    item_drive_id: str | None,
) -> None:
    if scope.kind is not GoogleDriveScopeKind.SHARED_DRIVE:
        return
    expected = scope.drive_id
    if change_drive_id is not None and change_drive_id != expected:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if item_drive_id is not None and item_drive_id != expected:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


class GoogleDriveScope(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: GoogleDriveScopeKind
    drive_id: str | None = Field(default=None, repr=False)

    @field_validator("drive_id", mode="before")
    @classmethod
    def _validate_drive_id_field(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_drive_identifier(value)

    @model_validator(mode="after")
    def _validate_scope_invariants(self) -> GoogleDriveScope:
        if self.kind is GoogleDriveScopeKind.USER:
            if self.drive_id is not None:
                raise ValueError(_INVALID_SCOPE_MESSAGE)
        elif self.kind is GoogleDriveScopeKind.SHARED_DRIVE:
            if self.drive_id is None:
                raise ValueError(_INVALID_SCOPE_MESSAGE)
        return self


class GoogleDriveSharedDrive(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    remote_id: str
    name: str
    created_at: datetime
    hidden: bool

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return _validate_drive_identifier(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_NAME_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_created_at(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            return _normalize_model_datetime(value)
        return _parse_timezone_aware_datetime(value)

    @field_validator("hidden", mode="before")
    @classmethod
    def _validate_hidden(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value


class GoogleDriveSharedDrivePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[GoogleDriveSharedDrive, ...]
    next_page_token: GoogleWorkspacePageToken | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _reject_duplicate_ids(self) -> GoogleDriveSharedDrivePage:
        seen: set[str] = set()
        for item in self.items:
            if item.remote_id in seen:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen.add(item.remote_id)
        return self


class GoogleDriveItem(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    remote_id: str
    scope: GoogleDriveScope
    kind: GoogleDriveItemKind

    name: str
    mime_type: str
    parent_ids: tuple[str, ...]
    drive_id: str | None = Field(default=None, repr=False)

    created_at: datetime
    modified_at: datetime

    size_bytes: int | None = None
    md5_checksum: str | None = Field(default=None, repr=False)
    version: int
    head_revision_id: str | None = Field(default=None, repr=False)

    web_view_link: str | None = Field(default=None, repr=False)
    can_download: bool

    shortcut_target_id: str | None = Field(default=None, repr=False)
    shortcut_target_mime_type: str | None = None

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return _validate_drive_identifier(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_NAME_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @field_validator("mime_type", mode="before")
    @classmethod
    def _validate_mime_type(cls, value: object) -> str:
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_NAME_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @field_validator("parent_ids", mode="before")
    @classmethod
    def _validate_parent_ids(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, tuple):
            seen: set[str] = set()
            for item in value:
                validated = _validate_drive_identifier(item)
                if validated in seen:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                seen.add(validated)
            return value
        return _parse_parent_ids(value)

    @field_validator("drive_id", mode="before")
    @classmethod
    def _validate_drive_id_field(cls, value: object) -> str | None:
        return _parse_optional_drive_id(value)

    @field_validator("created_at", "modified_at", mode="before")
    @classmethod
    def _normalize_datetimes(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            return _normalize_model_datetime(value)
        return _parse_timezone_aware_datetime(value)

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _parse_size_bytes(cls, value: object) -> int | None:
        if value is None:
            return None
        return _parse_non_negative_decimal_string(value)

    @field_validator("version", mode="before")
    @classmethod
    def _parse_version(cls, value: object) -> int:
        return _parse_non_negative_decimal_string(value)

    @field_validator("md5_checksum", mode="before")
    @classmethod
    def _validate_md5_checksum(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_CHECKSUM_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @field_validator("head_revision_id", mode="before")
    @classmethod
    def _validate_head_revision_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_REVISION_ID_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @field_validator("web_view_link", mode="before")
    @classmethod
    def _validate_web_view_link(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_WEB_LINK_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @field_validator("can_download", mode="before")
    @classmethod
    def _validate_can_download(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value

    @field_validator("shortcut_target_id", mode="before")
    @classmethod
    def _validate_shortcut_target_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_drive_identifier(value)

    @field_validator("shortcut_target_mime_type", mode="before")
    @classmethod
    def _validate_shortcut_target_mime_type(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_nonblank_bounded_string(
            value,
            max_length=_MAX_NAME_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    @model_validator(mode="after")
    def _validate_item_invariants(self) -> GoogleDriveItem:
        if self.scope.kind is GoogleDriveScopeKind.SHARED_DRIVE:
            if self.drive_id is not None and self.drive_id != self.scope.drive_id:
                raise ValueError(_INVALID_SCOPE_MESSAGE)

        if self.kind is GoogleDriveItemKind.SHORTCUT:
            if self.shortcut_target_id is None or self.shortcut_target_mime_type is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        else:
            if self.shortcut_target_id is not None or self.shortcut_target_mime_type is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

        if self.kind is GoogleDriveItemKind.FOLDER:
            if self.md5_checksum is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

        return self


class GoogleDriveItemPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[GoogleDriveItem, ...]
    next_page_token: GoogleWorkspacePageToken | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _reject_duplicate_ids(self) -> GoogleDriveItemPage:
        seen: set[str] = set()
        for item in self.items:
            if item.remote_id in seen:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen.add(item.remote_id)
        return self


class GoogleDriveChange(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    file_id: str
    scope: GoogleDriveScope
    removed: bool
    changed_at: datetime
    item: GoogleDriveItem | None = None

    @field_validator("file_id", mode="before")
    @classmethod
    def _validate_file_id(cls, value: object) -> str:
        return _validate_drive_identifier(value)

    @field_validator("removed", mode="before")
    @classmethod
    def _validate_removed(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value

    @field_validator("changed_at", mode="before")
    @classmethod
    def _normalize_changed_at(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            return _normalize_model_datetime(value)
        return _parse_timezone_aware_datetime(value)

    @model_validator(mode="after")
    def _validate_change_invariants(self) -> GoogleDriveChange:
        if self.removed:
            return self
        if self.item is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if self.item.remote_id != self.file_id:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if self.item.scope != self.scope:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleDriveChangePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    changes: tuple[GoogleDriveChange, ...]
    next_page_token: GoogleWorkspacePageToken | None = Field(default=None, repr=False)
    new_start_page_token: GoogleWorkspacePageToken | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _validate_exactly_one_token(self) -> GoogleDriveChangePage:
        has_next = self.next_page_token is not None
        has_new_start = self.new_start_page_token is not None
        if has_next == has_new_start:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


def _safe_construct_shared_drive(**kwargs: object) -> GoogleDriveSharedDrive:
    try:
        return GoogleDriveSharedDrive(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


def _safe_construct_item(**kwargs: object) -> GoogleDriveItem:
    try:
        return GoogleDriveItem(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


def _safe_construct_change(**kwargs: object) -> GoogleDriveChange:
    try:
        return GoogleDriveChange(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


def _parse_shared_drive_from_provider(mapping: dict[str, object]) -> GoogleDriveSharedDrive:
    return _safe_construct_shared_drive(
        remote_id=mapping.get("id"),
        name=mapping.get("name"),
        created_at=mapping.get("createdTime"),
        hidden=mapping.get("hidden"),
    )


def _parse_item_from_provider(
    mapping: dict[str, object],
    *,
    scope: GoogleDriveScope,
    expected_remote_id: str | None = None,
) -> GoogleDriveItem:
    trashed = _parse_optional_provider_bool(mapping, "trashed")
    if trashed is True:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    mime_type = mapping.get("mimeType")
    if not isinstance(mime_type, str) or not mime_type.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    kind = _classify_item_kind(mime_type.strip())

    shortcut_target_id: str | None = None
    shortcut_target_mime_type: str | None = None
    if kind is GoogleDriveItemKind.SHORTCUT:
        shortcut_details = mapping.get("shortcutDetails")
        if not isinstance(shortcut_details, dict):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        shortcut_target_id = _validate_drive_identifier(shortcut_details.get("targetId"))
        shortcut_target_mime_type = _validate_nonblank_bounded_string(
            shortcut_details.get("targetMimeType"),
            max_length=_MAX_NAME_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    capabilities = mapping.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    can_download = _parse_provider_bool(capabilities, "canDownload")

    remote_id = mapping.get("id")
    if expected_remote_id is not None:
        validated_expected = _validate_drive_identifier(expected_remote_id)
        validated_remote = _validate_drive_identifier(remote_id)
        if validated_remote != validated_expected:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        validated_remote = _validate_drive_identifier(remote_id)

    drive_id = _parse_optional_drive_id(mapping.get("driveId"))
    _validate_shared_drive_id_agreement(
        scope,
        change_drive_id=None,
        item_drive_id=drive_id,
    )

    parents_value = mapping.get("parents")
    if parents_value is None:
        parent_ids: tuple[str, ...] = ()
    else:
        parent_ids = _parse_parent_ids(parents_value)

    size_value = mapping.get("size")
    if kind in {GoogleDriveItemKind.FOLDER, GoogleDriveItemKind.SHORTCUT}:
        size_bytes = None
        if size_value is not None:
            size_bytes = _parse_non_negative_decimal_string(size_value)
    else:
        size_bytes = (
            _parse_non_negative_decimal_string(size_value) if size_value is not None else None
        )

    md5_value = mapping.get("md5Checksum")
    md5_checksum: str | None = None
    if md5_value is not None:
        if kind is not GoogleDriveItemKind.BLOB:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        md5_checksum = _validate_nonblank_bounded_string(
            md5_value,
            max_length=_MAX_CHECKSUM_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    head_revision_value = mapping.get("headRevisionId")
    head_revision_id: str | None = None
    if head_revision_value is not None:
        head_revision_id = _validate_nonblank_bounded_string(
            head_revision_value,
            max_length=_MAX_REVISION_ID_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    web_view_value = mapping.get("webViewLink")
    web_view_link: str | None = None
    if web_view_value is not None:
        web_view_link = _validate_nonblank_bounded_string(
            web_view_value,
            max_length=_MAX_WEB_LINK_LENGTH,
            message=_UNEXPECTED_RESPONSE_MESSAGE,
        )

    return _safe_construct_item(
        remote_id=validated_remote,
        scope=scope,
        kind=kind,
        name=mapping.get("name"),
        mime_type=mime_type,
        parent_ids=parent_ids,
        drive_id=drive_id,
        created_at=mapping.get("createdTime"),
        modified_at=mapping.get("modifiedTime"),
        size_bytes=size_bytes,
        md5_checksum=md5_checksum,
        version=mapping.get("version"),
        head_revision_id=head_revision_id,
        web_view_link=web_view_link,
        can_download=can_download,
        shortcut_target_id=shortcut_target_id,
        shortcut_target_mime_type=shortcut_target_mime_type,
    )


def _parse_change_from_provider(
    mapping: dict[str, object],
    *,
    scope: GoogleDriveScope,
) -> GoogleDriveChange | None:
    change_type = mapping.get("changeType")
    if not isinstance(change_type, str):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if change_type == "drive":
        return None
    if change_type != "file":
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    removed = _parse_provider_bool(mapping, "removed")
    file_id = _validate_drive_identifier(mapping.get("fileId"))
    changed_at = _parse_timezone_aware_datetime(mapping.get("time"))

    change_drive_id = _parse_optional_drive_id(mapping.get("driveId"))
    _validate_shared_drive_id_agreement(
        scope,
        change_drive_id=change_drive_id,
        item_drive_id=None,
    )

    item: GoogleDriveItem | None = None
    if not removed:
        file_payload = mapping.get("file")
        if not isinstance(file_payload, dict):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        item = _parse_item_from_provider(
            file_payload,
            scope=scope,
            expected_remote_id=file_id,
        )
        _validate_shared_drive_id_agreement(
            scope,
            change_drive_id=change_drive_id,
            item_drive_id=item.drive_id,
        )

    return _safe_construct_change(
        file_id=file_id,
        scope=scope,
        removed=removed,
        changed_at=changed_at,
        item=item,
    )


def _parse_optional_page_token_from_payload(
    payload: dict[str, object],
    key: str,
) -> GoogleWorkspacePageToken | None:
    if key not in payload:
        return None
    raw = payload[key]
    if not isinstance(raw, str):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return GoogleWorkspacePageToken(value=raw)


def _validate_incomplete_search(payload: dict[str, object]) -> None:
    if "incompleteSearch" not in payload:
        return
    value = payload["incompleteSearch"]
    if type(value) is not bool:
        raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value:
        raise IntegrationDependencyError(_INCOMPLETE_INVENTORY_MESSAGE)


def _validate_file_id_for_path(file_id: str) -> str:
    validated = _validate_drive_identifier(file_id)
    if "/" in validated or "\\" in validated:
        raise IntegrationConfigurationError(_INVALID_IDENTIFIER_MESSAGE)
    return validated


@runtime_checkable
class GoogleDriveKnowledgeReadClient(Protocol):
    def list_shared_drives_page(
        self,
        *,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 100,
    ) -> GoogleDriveSharedDrivePage:
        ...

    def read_items_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 200,
    ) -> GoogleDriveItemPage:
        ...

    def read_item(
        self,
        *,
        scope: GoogleDriveScope,
        file_id: str,
    ) -> GoogleDriveItem:
        ...

    def read_start_page_token(
        self,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleWorkspacePageToken:
        ...

    def read_changes_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken,
        limit: int = 200,
    ) -> GoogleDriveChangePage:
        ...


class GoogleDriveKnowledgeReader:
    """Stateless Google Drive knowledge reader using one shared transport."""

    def __init__(
        self,
        *,
        transport: GoogleWorkspaceTransport,
    ) -> None:
        if not isinstance(transport, GoogleWorkspaceTransport):
            raise IntegrationConfigurationError(_UNEXPECTED_RESPONSE_MESSAGE)
        self._transport = transport

    def list_shared_drives_page(
        self,
        *,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 100,
    ) -> GoogleDriveSharedDrivePage:
        validated_limit = _validate_page_limit(
            limit,
            minimum=_SHARED_DRIVE_LIST_LIMIT_MIN,
            maximum=_SHARED_DRIVE_LIST_LIMIT_MAX,
        )
        validated_token = _validate_optional_page_token(page_token)

        params: dict[str, object] = {
            "pageSize": validated_limit,
            "fields": _SHARED_DRIVE_LIST_FIELDS,
        }
        if validated_token is not None:
            params["pageToken"] = validated_token.value

        payload = self._transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/drives",
            params=params,
        )
        return self._parse_shared_drive_page(payload)

    def read_items_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken | None = None,
        limit: int = 200,
    ) -> GoogleDriveItemPage:
        validated_scope = _validate_scope(scope)
        validated_limit = _validate_page_limit(
            limit,
            minimum=_ITEM_PAGE_LIMIT_MIN,
            maximum=_ITEM_PAGE_LIMIT_MAX,
        )
        validated_token = _validate_optional_page_token(page_token)

        params: dict[str, object] = {
            "pageSize": validated_limit,
            "spaces": "drive",
            "q": "trashed = false",
            "supportsAllDrives": True,
            "fields": _INVENTORY_FIELDS,
        }
        if validated_scope.kind is GoogleDriveScopeKind.USER:
            params["corpora"] = "user"
            params["includeItemsFromAllDrives"] = False
        else:
            params["corpora"] = "drive"
            params["driveId"] = validated_scope.drive_id
            params["includeItemsFromAllDrives"] = True

        if validated_token is not None:
            params["pageToken"] = validated_token.value

        payload = self._transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            params=params,
        )
        return self._parse_item_page(payload, scope=validated_scope)

    def read_item(
        self,
        *,
        scope: GoogleDriveScope,
        file_id: str,
    ) -> GoogleDriveItem:
        validated_scope = _validate_scope(scope)
        validated_file_id = _validate_file_id_for_path(file_id)
        encoded_id = quote(validated_file_id, safe="")

        payload = self._transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path=f"/files/{encoded_id}",
            params={
                "supportsAllDrives": True,
                "fields": _FILE_FIELDS,
            },
        )
        if not isinstance(payload, dict):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            item = _parse_item_from_provider(
                payload,
                scope=validated_scope,
                expected_remote_id=validated_file_id,
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None
        return item

    def read_start_page_token(
        self,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleWorkspacePageToken:
        validated_scope = _validate_scope(scope)

        params: dict[str, object] = {"supportsAllDrives": True}
        if validated_scope.kind is GoogleDriveScopeKind.SHARED_DRIVE:
            params["driveId"] = validated_scope.drive_id

        payload = self._transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/changes/startPageToken",
            params=params,
        )
        if not isinstance(payload, dict):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        if "startPageToken" not in payload:
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        raw_token = payload["startPageToken"]
        if not isinstance(raw_token, str):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            return GoogleWorkspacePageToken(value=raw_token)
        except (ValueError, TypeError):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None

    def read_changes_page(
        self,
        *,
        scope: GoogleDriveScope,
        page_token: GoogleWorkspacePageToken,
        limit: int = 200,
    ) -> GoogleDriveChangePage:
        validated_scope = _validate_scope(scope)
        validated_token = _validate_required_page_token(page_token)
        validated_limit = _validate_page_limit(
            limit,
            minimum=_ITEM_PAGE_LIMIT_MIN,
            maximum=_ITEM_PAGE_LIMIT_MAX,
        )

        params: dict[str, object] = {
            "pageToken": validated_token.value,
            "pageSize": validated_limit,
            "spaces": "drive",
            "includeRemoved": True,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "fields": _CHANGE_FIELDS,
        }
        if validated_scope.kind is GoogleDriveScopeKind.SHARED_DRIVE:
            params["driveId"] = validated_scope.drive_id

        payload = self._transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/changes",
            params=params,
        )
        return self._parse_change_page(payload, scope=validated_scope)

    def _parse_shared_drive_page(self, payload: object) -> GoogleDriveSharedDrivePage:
        try:
            collection = parse_google_workspace_collection_page(
                payload,
                items_field="drives",
            )
            items = tuple(
                _parse_shared_drive_from_provider(item) for item in collection.items
            )
            return GoogleDriveSharedDrivePage(
                items=items,
                next_page_token=collection.next_page_token,
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None

    def _parse_item_page(
        self,
        payload: object,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleDriveItemPage:
        if not isinstance(payload, dict):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            _validate_incomplete_search(payload)
            collection = parse_google_workspace_collection_page(
                payload,
                items_field="files",
            )
            items = tuple(
                _parse_item_from_provider(item, scope=scope) for item in collection.items
            )
            return GoogleDriveItemPage(
                items=items,
                next_page_token=collection.next_page_token,
            )
        except IntegrationDependencyError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None

    def _parse_change_page(
        self,
        payload: object,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleDriveChangePage:
        if not isinstance(payload, dict):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            raw_changes = payload.get("changes")
            if not isinstance(raw_changes, list):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            parsed_changes: list[GoogleDriveChange] = []
            for raw_change in raw_changes:
                if not isinstance(raw_change, dict):
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                change = _parse_change_from_provider(raw_change, scope=scope)
                if change is not None:
                    parsed_changes.append(change)
            next_token = _parse_optional_page_token_from_payload(payload, "nextPageToken")
            new_start_token = _parse_optional_page_token_from_payload(
                payload,
                "newStartPageToken",
            )
            return GoogleDriveChangePage(
                changes=tuple(parsed_changes),
                next_page_token=next_token,
                new_start_page_token=new_start_token,
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None
