# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Drive knowledge-read: bounded blob download and deterministic native export."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceBinaryTransport,
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
    normalize_google_workspace_media_type,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GoogleDriveItem,
    GoogleDriveItemKind,
    GoogleDriveKnowledgeReader,
    GoogleDriveScope,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)

DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES = 25 * 1024 * 1024
ABSOLUTE_GOOGLE_DRIVE_CONTENT_MAX_BYTES = 100 * 1024 * 1024
GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES = 10 * 1024 * 1024

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_INVALID_ITEM_MESSAGE = "invalid Google Drive content item"
_INVALID_MAX_BYTES_MESSAGE = "invalid Google Drive content max_bytes"
_INVALID_BINARY_RESULT_MESSAGE = "invalid Google Drive binary content result"
_BINARY_TRANSPORT_REQUIRED_MESSAGE = (
    "Google Workspace transport does not support binary content"
)

_NATIVE_EXPORT_MIME: dict[str, str] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
    "application/vnd.google-apps.drawing": "application/pdf",
    "application/vnd.google-apps.script": "application/vnd.google-apps.script+json",
}

_UNSUPPORTED_NATIVE_MIMES = frozenset(
    {
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.site",
        "application/vnd.google-apps.map",
        "application/vnd.google-apps.fusiontable",
        "application/vnd.google-apps.jam",
        "application/vnd.google-apps.vid",
    }
)


class GoogleDriveContentChanged(IntegrationDependencyError):
    """Drive file revision changed during content download."""

    def __init__(self) -> None:
        super().__init__("Google Drive file changed during content download")


class GoogleDriveContentTooLarge(IntegrationConfigurationError):
    """Drive file exceeds the configured content byte limit."""

    def __init__(self) -> None:
        super().__init__("Google Drive file exceeds the configured content limit")


class GoogleDriveContentUnavailable(IntegrationConfigurationError):
    """Drive file content is not available for download."""

    def __init__(self) -> None:
        super().__init__("Google Drive file content is unavailable")


class GoogleDriveUnsupportedContent(IntegrationConfigurationError):
    """Drive file content type is unsupported."""

    def __init__(self) -> None:
        super().__init__("Google Drive file content type is unsupported")


class GoogleDriveContentMode(StrEnum):
    BLOB = "blob"
    EXPORT = "export"


class GoogleDriveFileContent(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    item: GoogleDriveItem = Field(repr=False)
    mode: GoogleDriveContentMode
    content_mime_type: str
    data: bytes = Field(repr=False)
    size_bytes: int
    content_hash: str = Field(repr=False)

    @field_validator("item", mode="before")
    @classmethod
    def _validate_item(cls, value: object) -> GoogleDriveItem:
        return _reconstruct_item(value)

    @field_validator("content_mime_type", mode="before")
    @classmethod
    def _validate_content_mime_type(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError(_INVALID_ITEM_MESSAGE)
        try:
            return normalize_google_workspace_media_type(value)
        except (TypeError, ValueError):
            raise ValueError(_INVALID_ITEM_MESSAGE) from None

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: object) -> bytes:
        if type(value) is not bytes:
            raise ValueError(_INVALID_ITEM_MESSAGE)
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError(_INVALID_ITEM_MESSAGE)
        if value < 0:
            raise ValueError(_INVALID_ITEM_MESSAGE)
        return value

    @field_validator("content_hash", mode="before")
    @classmethod
    def _validate_content_hash(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(_INVALID_ITEM_MESSAGE)
        if not _CONTENT_HASH_PATTERN.match(value):
            raise ValueError(_INVALID_ITEM_MESSAGE)
        return value

    @model_validator(mode="after")
    def _validate_content_invariants(self) -> GoogleDriveFileContent:
        if not self.item.can_download:
            raise ValueError(_INVALID_ITEM_MESSAGE)
        if self.size_bytes != len(self.data):
            raise ValueError(_INVALID_ITEM_MESSAGE)
        expected_hash = hashlib.sha256(self.data).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError(_INVALID_ITEM_MESSAGE)
        if self.mode is GoogleDriveContentMode.BLOB:
            if self.item.kind is not GoogleDriveItemKind.BLOB:
                raise ValueError(_INVALID_ITEM_MESSAGE)
            if self.content_mime_type != self.item.mime_type:
                raise ValueError(_INVALID_ITEM_MESSAGE)
        elif self.mode is GoogleDriveContentMode.EXPORT:
            if self.item.kind is not GoogleDriveItemKind.NATIVE_DOCUMENT:
                raise ValueError(_INVALID_ITEM_MESSAGE)
            expected_export = _NATIVE_EXPORT_MIME.get(self.item.mime_type)
            if expected_export is None or self.content_mime_type != expected_export:
                raise ValueError(_INVALID_ITEM_MESSAGE)
        return self


@runtime_checkable
class GoogleDriveContentReadClient(Protocol):
    def read_drive_file_content(
        self,
        *,
        item: GoogleDriveItem,
        max_bytes: int = DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    ) -> GoogleDriveFileContent:
        ...


def _validate_max_bytes(max_bytes: object) -> int:
    if type(max_bytes) is not int:
        raise IntegrationConfigurationError(_INVALID_MAX_BYTES_MESSAGE) from None
    if not 1 <= max_bytes <= ABSOLUTE_GOOGLE_DRIVE_CONTENT_MAX_BYTES:
        raise IntegrationConfigurationError(_INVALID_MAX_BYTES_MESSAGE) from None
    return max_bytes


def _validate_blob_mime_type(mime_type: str) -> None:
    try:
        normalize_google_workspace_media_type(mime_type)
    except (TypeError, ValueError):
        raise GoogleDriveUnsupportedContent() from None


def _resolve_export_mime(mime_type: str) -> str:
    if mime_type in _UNSUPPORTED_NATIVE_MIMES:
        raise GoogleDriveUnsupportedContent() from None
    export_mime = _NATIVE_EXPORT_MIME.get(mime_type)
    if export_mime is None:
        if mime_type.startswith("application/vnd.google-apps."):
            raise GoogleDriveUnsupportedContent() from None
        raise GoogleDriveUnsupportedContent() from None
    return export_mime


def _reconstruct_item(item: object) -> GoogleDriveItem:
    if type(item) is not GoogleDriveItem:
        raise IntegrationConfigurationError(_INVALID_ITEM_MESSAGE) from None
    try:
        snapshot = item.model_dump(mode="python")
        scope_data = snapshot.get("scope")
        if isinstance(scope_data, dict):
            snapshot["scope"] = GoogleDriveScope(**scope_data)
    except Exception:
        raise IntegrationConfigurationError(_INVALID_ITEM_MESSAGE) from None
    try:
        validated = GoogleDriveItem(**snapshot)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_ITEM_MESSAGE) from None
    return validated


def _copy_and_validate_binary_payload(
    payload: object,
    *,
    expected_content_type: str,
    effective_max_bytes: int,
) -> GoogleWorkspaceBinaryPayload:
    if type(payload) is not GoogleWorkspaceBinaryPayload:
        raise IntegrationDependencyError(_INVALID_BINARY_RESULT_MESSAGE) from None
    try:
        validated = GoogleWorkspaceBinaryPayload(
            data=payload.data,
            content_type=payload.content_type,
        )
    except (TypeError, ValueError):
        raise IntegrationDependencyError(_INVALID_BINARY_RESULT_MESSAGE) from None
    if validated.content_type != expected_content_type:
        raise IntegrationDependencyError(_INVALID_BINARY_RESULT_MESSAGE) from None
    if len(validated.data) > effective_max_bytes:
        raise GoogleDriveContentTooLarge() from None
    return validated


def _validate_content_item(item: GoogleDriveItem) -> GoogleDriveItem:
    if item.kind is GoogleDriveItemKind.FOLDER:
        raise GoogleDriveUnsupportedContent() from None
    if item.kind is GoogleDriveItemKind.SHORTCUT:
        raise GoogleDriveUnsupportedContent() from None
    if item.kind is GoogleDriveItemKind.OTHER:
        raise GoogleDriveUnsupportedContent() from None
    if not item.can_download:
        raise GoogleDriveContentUnavailable() from None
    if item.kind is GoogleDriveItemKind.BLOB:
        _validate_blob_mime_type(item.mime_type)
    elif item.kind is GoogleDriveItemKind.NATIVE_DOCUMENT:
        _resolve_export_mime(item.mime_type)
    else:
        raise GoogleDriveUnsupportedContent() from None
    return item


def _revision_fields_match(left: GoogleDriveItem, right: GoogleDriveItem) -> bool:
    return (
        left.remote_id == right.remote_id
        and left.scope == right.scope
        and left.kind == right.kind
        and left.mime_type == right.mime_type
        and left.version == right.version
        and left.modified_at == right.modified_at
        and left.size_bytes == right.size_bytes
        and left.md5_checksum == right.md5_checksum
        and left.head_revision_id == right.head_revision_id
        and left.can_download == right.can_download
    )


def _verify_revision_match(
    current: GoogleDriveItem,
    *,
    expected: GoogleDriveItem,
) -> None:
    if current.can_download is False and expected.can_download is True:
        raise GoogleDriveContentUnavailable() from None
    if not _revision_fields_match(current, expected):
        raise GoogleDriveContentChanged() from None


def _encode_file_id(file_id: str) -> str:
    if "/" in file_id or "\\" in file_id:
        raise IntegrationConfigurationError(_INVALID_ITEM_MESSAGE) from None
    return quote(file_id, safe="")


def _safe_construct_file_content(**kwargs: object) -> GoogleDriveFileContent:
    try:
        return GoogleDriveFileContent(**kwargs)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise IntegrationConfigurationError(_INVALID_ITEM_MESSAGE) from None


class GoogleDriveContentReader:
    """Stateless Google Drive content reader using one shared transport."""

    def __init__(
        self,
        *,
        transport: object,
    ) -> None:
        if not isinstance(transport, GoogleWorkspaceTransport):
            raise IntegrationConfigurationError(_BINARY_TRANSPORT_REQUIRED_MESSAGE) from None
        if not isinstance(transport, GoogleWorkspaceBinaryTransport):
            raise IntegrationConfigurationError(_BINARY_TRANSPORT_REQUIRED_MESSAGE) from None
        self._transport = transport
        self._metadata_reader = GoogleDriveKnowledgeReader(transport=transport)

    def read_drive_file_content(
        self,
        *,
        item: GoogleDriveItem,
        max_bytes: int = DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    ) -> GoogleDriveFileContent:
        validated_item = _validate_content_item(_reconstruct_item(item))
        validated_max_bytes = _validate_max_bytes(max_bytes)

        if (
            validated_item.size_bytes is not None
            and validated_item.size_bytes > validated_max_bytes
        ):
            raise GoogleDriveContentTooLarge() from None

        metadata_before = self._metadata_reader.read_item(
            scope=validated_item.scope,
            file_id=validated_item.remote_id,
        )
        _verify_revision_match(metadata_before, expected=validated_item)
        if not metadata_before.can_download:
            raise GoogleDriveContentUnavailable() from None

        encoded_id = _encode_file_id(validated_item.remote_id)

        if validated_item.kind is GoogleDriveItemKind.BLOB:
            mode = GoogleDriveContentMode.BLOB
            content_mime_type = validated_item.mime_type
            effective_max_bytes = validated_max_bytes
            try:
                payload = self._transport.get_bytes(
                    source_kind=GoogleWorkspaceSourceKind.DRIVE,
                    relative_path=f"/files/{encoded_id}",
                    params={
                        "alt": "media",
                        "supportsAllDrives": True,
                    },
                    expected_content_type=content_mime_type,
                    max_bytes=effective_max_bytes,
                    range_limited=True,
                )
            except GoogleWorkspaceApiError as exc:
                if exc.kind is GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE:
                    raise GoogleDriveContentTooLarge() from None
                raise
        else:
            mode = GoogleDriveContentMode.EXPORT
            content_mime_type = _resolve_export_mime(validated_item.mime_type)
            effective_max_bytes = min(
                validated_max_bytes,
                GOOGLE_DRIVE_NATIVE_EXPORT_MAX_BYTES,
            )
            try:
                payload = self._transport.get_bytes(
                    source_kind=GoogleWorkspaceSourceKind.DRIVE,
                    relative_path=f"/files/{encoded_id}/export",
                    params={"mimeType": content_mime_type},
                    expected_content_type=content_mime_type,
                    max_bytes=effective_max_bytes,
                    range_limited=False,
                )
            except GoogleWorkspaceApiError as exc:
                if exc.kind is GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE:
                    raise GoogleDriveContentTooLarge() from None
                raise

        payload = _copy_and_validate_binary_payload(
            payload,
            expected_content_type=content_mime_type,
            effective_max_bytes=effective_max_bytes,
        )

        metadata_after = self._metadata_reader.read_item(
            scope=validated_item.scope,
            file_id=validated_item.remote_id,
        )
        _verify_revision_match(metadata_after, expected=metadata_before)

        data = payload.data
        if mode is GoogleDriveContentMode.BLOB:
            if metadata_before.size_bytes is not None and metadata_before.size_bytes != len(data):
                raise GoogleDriveContentChanged() from None
            if metadata_before.md5_checksum is not None:
                computed_md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
                if computed_md5 != metadata_before.md5_checksum:
                    raise GoogleDriveContentChanged() from None

        content_hash = hashlib.sha256(data).hexdigest()
        return _safe_construct_file_content(
            item=metadata_after,
            mode=mode,
            content_mime_type=content_mime_type,
            data=data,
            size_bytes=len(data),
            content_hash=content_hash,
        )
