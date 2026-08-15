# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace Drive knowledge source adapter."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
    GoogleDriveChange,
    GoogleDriveChangePage,
    GoogleDriveItem,
    GoogleDriveItemKind,
    GoogleDriveItemPage,
    GoogleDriveScope,
    GoogleDriveScopeKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive_content import (
    DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
    GoogleDriveContentChanged,
    GoogleDriveContentTooLarge,
    GoogleDriveContentUnavailable,
    GoogleDriveContentProfile,
    GoogleDriveFileContent,
    GoogleDriveUnsupportedContent,
    resolve_google_drive_content_profile,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
    GoogleWorkspacePageToken,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

GOOGLE_DRIVE_USER_SCOPE_TYPE = "google_workspace_drive_user"
GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE = "google_workspace_drive_shared_drive"
GOOGLE_DRIVE_USER_SCOPE_ID = "user"

GOOGLE_DRIVE_CURSOR_VERSION = "google_workspace.drive.cursor.v1"
GOOGLE_DRIVE_ITEM_METADATA_VERSION = "google_workspace.drive.item.v1"

_INVALID_SCOPE_MESSAGE = "Google Workspace Drive knowledge source scope is invalid"
_INVALID_CURSOR_MESSAGE = "Google Workspace Drive knowledge cursor is invalid"
_INVALID_PROVIDER_RESPONSE_MESSAGE = (
    "Google Workspace Drive knowledge provider response is invalid"
)
_INVALID_DESCRIPTOR_MESSAGE = "Google Workspace Drive file descriptor is invalid"
_CONFIGURATION_ERROR_MESSAGE = "Google Workspace Drive knowledge page limit is invalid"
_DEPENDENCY_UNAVAILABLE_MESSAGE = (
    "Google Workspace Drive knowledge dependency is unavailable"
)
_UNSUPPORTED_PERMISSIONS_MESSAGE = (
    "Authoritative Google Drive permissions projection is not implemented"
)
_INTEGRATION_REQUIRED_MESSAGE = (
    "Google Workspace Drive knowledge adapter requires "
    "Google Workspace collaboration-suite integration"
)

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_CURSOR_TOKEN_LEN = 4096
_MAX_ENCODED_CURSOR_LENGTH = 24_576
_CURSOR_ALPHABET = re.compile(r"^[A-Za-z0-9_-]+$")
_PROVIDER_PAGE_LIMIT = 200

_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "scope_kind",
        "drive_id",
        "item_kind",
        "mime_type",
        "parent_ids",
        "created_at",
        "modified_at",
        "size_bytes",
        "md5_checksum",
        "head_revision_id",
        "can_download",
        "shortcut_target_id",
        "shortcut_target_mime_type",
        "content_supported",
        "content_transport_mode",
        "content_mime_type",
    }
)

_ITEM_TYPE_BY_KIND: dict[GoogleDriveItemKind, str] = {
    GoogleDriveItemKind.BLOB: "google_workspace_drive_blob",
    GoogleDriveItemKind.FOLDER: "google_workspace_drive_folder",
    GoogleDriveItemKind.NATIVE_DOCUMENT: "google_workspace_drive_native_document",
    GoogleDriveItemKind.SHORTCUT: "google_workspace_drive_shortcut",
    GoogleDriveItemKind.OTHER: "google_workspace_drive_other",
}

_T = TypeVar("_T")


def _validate_cursor_token(value: object) -> str:
    if type(value) is not str:
        raise ValueError("token must be a string")
    if value != value.strip():
        raise ValueError("token must not have surrounding whitespace")
    if not value:
        raise ValueError("token must not be blank")
    if _ASCII_CONTROL.search(value):
        raise ValueError("token must not contain control characters")
    if len(value) > _MAX_CURSOR_TOKEN_LEN:
        raise ValueError("token exceeds maximum length")
    return value


class _GoogleDriveCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["google_workspace.drive.cursor.v1"]
    scope_kind: Literal["user", "shared_drive"]
    drive_id: str | None = Field(default=None, repr=False)
    phase: Literal["inventory", "changes"]
    inventory_page_token: str | None = Field(default=None, repr=False)
    change_page_token: str = Field(repr=False)

    @field_validator("drive_id", mode="before")
    @classmethod
    def _validate_drive_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_cursor_token(value)

    @field_validator("inventory_page_token", mode="before")
    @classmethod
    def _validate_inventory_page_token(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_cursor_token(value)

    @field_validator("change_page_token", mode="before")
    @classmethod
    def _validate_change_page_token(cls, value: object) -> str:
        return _validate_cursor_token(value)

    @model_validator(mode="after")
    def _validate_invariants(self) -> _GoogleDriveCursor:
        if self.scope_kind == "user":
            if self.drive_id is not None:
                raise ValueError("user scope cursor must not contain drive_id")
        elif self.scope_kind == "shared_drive":
            if self.drive_id is None:
                raise ValueError("shared_drive cursor requires drive_id")
            GoogleDriveScope(kind=GoogleDriveScopeKind.SHARED_DRIVE, drive_id=self.drive_id)
        if self.phase == "inventory":
            if self.inventory_page_token is None:
                raise ValueError("inventory phase requires inventory_page_token")
        elif self.phase == "changes":
            if self.inventory_page_token is not None:
                raise ValueError("changes phase must not contain inventory_page_token")
        return self


class GoogleWorkspaceDriveKnowledgeAdapter:
    """Thin mapping from Google Workspace Drive integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return GOOGLE_DRIVE_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=True,
            content_fetch=True,
            binary_content=True,
            rich_text_content=False,
            structured_content=False,
            permissions=False,
            tombstones=True,
            remote_versions=True,
            reconciliation=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        return KnowledgeScopeInfo(
            source=validated_source,
            capabilities=self.capabilities,
            safe_display_name=validated_source.scope.safe_display_name,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        google_integration = self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        scope = self._source_to_scope(validated_source)
        provider_limit = self._validate_limit(limit)
        decoded = self._decode_cursor(cursor, scope=scope)

        if decoded is None:
            return await self._read_initial_inventory(
                integration=google_integration,
                scope=scope,
                provider_limit=provider_limit,
            )
        if decoded.phase == "inventory":
            return await self._read_inventory_continuation(
                integration=google_integration,
                scope=scope,
                cursor=decoded,
                provider_limit=provider_limit,
            )
        return await self._read_changes_page(
            integration=google_integration,
            scope=scope,
            cursor=decoded,
            provider_limit=provider_limit,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        google_integration = self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        scope = self._source_to_scope(validated_source)
        provider_item = self._descriptor_to_provider_item(
            item,
            source=validated_source,
            scope=scope,
        )
        result = await self._invoke_integration(
            lambda: google_integration.read_drive_file_content(
                item=provider_item,
                max_bytes=DEFAULT_GOOGLE_DRIVE_CONTENT_MAX_BYTES,
            ),
            cursor_in_use=False,
        )
        self._validate_fetched_content(result, provider_item=provider_item)
        return KnowledgeContent(
            mode=KnowledgeContentMode.BINARY,
            binary=result.data,
            mime_type=result.content_mime_type,
            content_hash=result.content_hash,
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self._require_google_integration(integration=integration)
        validated_source = self._validate_source(source)
        reconstructed = self._reconstruct_descriptor(item)
        self._validate_item_provenance(reconstructed, source=validated_source)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=_UNSUPPORTED_PERMISSIONS_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    async def _read_initial_inventory(
        self,
        *,
        integration: GoogleWorkspaceCollaborationSuiteIntegration,
        scope: GoogleDriveScope,
        provider_limit: int,
    ) -> KnowledgePage:
        start_token = await self._invoke_integration(
            lambda: integration.read_drive_start_page_token(scope=scope),
            cursor_in_use=False,
        )
        start_token_value = self._page_token_value(start_token)
        page = await self._invoke_integration(
            lambda: integration.read_drive_items_page(
                scope=scope,
                page_token=None,
                limit=provider_limit,
            ),
            cursor_in_use=False,
        )
        try:
            validated_page = self._reconstruct_item_page(page, scope=scope)
            changes = tuple(self._item_to_change(item) for item in validated_page.items)
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if validated_page.next_page_token is not None:
            next_cursor = self._encode_cursor(
                _GoogleDriveCursor(
                    schema_version=GOOGLE_DRIVE_CURSOR_VERSION,
                    scope_kind=scope.kind.value,
                    drive_id=scope.drive_id,
                    phase="inventory",
                    inventory_page_token=validated_page.next_page_token.value,
                    change_page_token=start_token_value,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        checkpoint = self._encode_cursor(
            _GoogleDriveCursor(
                schema_version=GOOGLE_DRIVE_CURSOR_VERSION,
                scope_kind=scope.kind.value,
                drive_id=scope.drive_id,
                phase="changes",
                inventory_page_token=None,
                change_page_token=start_token_value,
            )
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=checkpoint,
            has_more=False,
        )

    async def _read_inventory_continuation(
        self,
        *,
        integration: GoogleWorkspaceCollaborationSuiteIntegration,
        scope: GoogleDriveScope,
        cursor: _GoogleDriveCursor,
        provider_limit: int,
    ) -> KnowledgePage:
        page_token = GoogleWorkspacePageToken(value=cursor.inventory_page_token or "")
        page = await self._invoke_integration(
            lambda: integration.read_drive_items_page(
                scope=scope,
                page_token=page_token,
                limit=provider_limit,
            ),
            cursor_in_use=True,
        )
        try:
            validated_page = self._reconstruct_item_page(page, scope=scope)
            changes = tuple(self._item_to_change(item) for item in validated_page.items)
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        captured_change_token = cursor.change_page_token
        if validated_page.next_page_token is not None:
            next_cursor = self._encode_cursor(
                _GoogleDriveCursor(
                    schema_version=GOOGLE_DRIVE_CURSOR_VERSION,
                    scope_kind=scope.kind.value,
                    drive_id=scope.drive_id,
                    phase="inventory",
                    inventory_page_token=validated_page.next_page_token.value,
                    change_page_token=captured_change_token,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        checkpoint = self._encode_cursor(
            _GoogleDriveCursor(
                schema_version=GOOGLE_DRIVE_CURSOR_VERSION,
                scope_kind=scope.kind.value,
                drive_id=scope.drive_id,
                phase="changes",
                inventory_page_token=None,
                change_page_token=captured_change_token,
            )
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=checkpoint,
            has_more=False,
        )

    async def _read_changes_page(
        self,
        *,
        integration: GoogleWorkspaceCollaborationSuiteIntegration,
        scope: GoogleDriveScope,
        cursor: _GoogleDriveCursor,
        provider_limit: int,
    ) -> KnowledgePage:
        page_token = GoogleWorkspacePageToken(value=cursor.change_page_token)
        page = await self._invoke_integration(
            lambda: integration.read_drive_changes_page(
                scope=scope,
                page_token=page_token,
                limit=provider_limit,
            ),
            cursor_in_use=True,
        )
        try:
            validated_page = self._reconstruct_change_page(page, scope=scope)
            changes = tuple(
                self._change_to_knowledge_change(change) for change in validated_page.changes
            )
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if validated_page.next_page_token is not None:
            next_cursor = self._encode_cursor(
                _GoogleDriveCursor(
                    schema_version=GOOGLE_DRIVE_CURSOR_VERSION,
                    scope_kind=scope.kind.value,
                    drive_id=scope.drive_id,
                    phase="changes",
                    inventory_page_token=None,
                    change_page_token=validated_page.next_page_token.value,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        new_start = validated_page.new_start_page_token
        if new_start is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        checkpoint = self._encode_cursor(
            _GoogleDriveCursor(
                schema_version=GOOGLE_DRIVE_CURSOR_VERSION,
                scope_kind=scope.kind.value,
                drive_id=scope.drive_id,
                phase="changes",
                inventory_page_token=None,
                change_page_token=new_start.value,
            )
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=checkpoint,
            has_more=False,
        )

    def _invalid_scope_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message=_INVALID_SCOPE_MESSAGE,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            retryable=False,
        )

    def _invalid_cursor_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_CURSOR,
            safe_message=_INVALID_CURSOR_MESSAGE,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            retryable=False,
        )

    def _invalid_provider_response_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            retryable=False,
        )

    def _invalid_descriptor_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message=_INVALID_DESCRIPTOR_MESSAGE,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            retryable=False,
        )

    def _integration_required_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message=_INTEGRATION_REQUIRED_MESSAGE,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            retryable=False,
        )

    def _require_google_integration(
        self,
        integration: object,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        if not isinstance(integration, GoogleWorkspaceCollaborationSuiteIntegration):
            raise self._integration_required_error()
        return integration

    def _reconstruct_source(self, source: object) -> KnowledgeSourceRef:
        try:
            if type(source) is not KnowledgeSourceRef:
                raise ValueError("invalid source type")
            scope_raw = source.scope
            if type(scope_raw.safe_display_name) is not str:
                raise ValueError("invalid safe_display_name")
            snapshot = source.model_dump(mode="python")
            scope_data = snapshot.get("scope")
            if isinstance(scope_data, dict):
                snapshot["scope"] = KnowledgeSourceScope(**scope_data)
            return KnowledgeSourceRef(**snapshot)
        except Exception:
            raise self._invalid_scope_error() from None

    def _copy_page_token(self, value: object) -> GoogleWorkspacePageToken:
        try:
            if type(value) is not GoogleWorkspacePageToken:
                raise ValueError("invalid page token type")
            token_value = value.value
            validated_value = _validate_cursor_token(token_value)
            return GoogleWorkspacePageToken(value=validated_value)
        except Exception:
            raise self._invalid_provider_response_error() from None

    def _page_token_value(self, token: object) -> str:
        return self._copy_page_token(token).value

    def _validate_source(self, source: KnowledgeSourceRef) -> KnowledgeSourceRef:
        reconstructed = self._reconstruct_source(source)
        if (
            reconstructed.provider_id != self.provider_id
            or reconstructed.integration_kind != self.integration_kind
            or reconstructed.source_kind != self.source_kind
        ):
            raise self._invalid_scope_error()
        scope = reconstructed.scope
        if type(scope.safe_display_name) is not str:
            raise self._invalid_scope_error()
        if scope.remote_scope_type not in {
            GOOGLE_DRIVE_USER_SCOPE_TYPE,
            GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
        }:
            raise self._invalid_scope_error()
        if scope.parameters:
            raise self._invalid_scope_error()
        if scope.remote_scope_type == GOOGLE_DRIVE_USER_SCOPE_TYPE:
            if scope.remote_scope_id != GOOGLE_DRIVE_USER_SCOPE_ID:
                raise self._invalid_scope_error()
            try:
                GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
            except (ValueError, TypeError, ValidationError):
                raise self._invalid_scope_error() from None
        else:
            try:
                GoogleDriveScope(
                    kind=GoogleDriveScopeKind.SHARED_DRIVE,
                    drive_id=scope.remote_scope_id,
                )
            except (ValueError, TypeError, ValidationError):
                raise self._invalid_scope_error() from None
        return reconstructed

    def _source_to_scope(self, source: KnowledgeSourceRef) -> GoogleDriveScope:
        scope = source.scope
        if scope.remote_scope_type == GOOGLE_DRIVE_USER_SCOPE_TYPE:
            return GoogleDriveScope(kind=GoogleDriveScopeKind.USER)
        return GoogleDriveScope(
            kind=GoogleDriveScopeKind.SHARED_DRIVE,
            drive_id=scope.remote_scope_id,
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return min(limit, _PROVIDER_PAGE_LIMIT)

    def _validate_item_provenance(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        provenance = item.provenance
        if (
            provenance.provider_id != source.provider_id
            or provenance.source_kind != source.source_kind
        ):
            raise self._invalid_descriptor_error()

    def _reconstruct_item_page(
        self,
        page: object,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleDriveItemPage:
        try:
            if type(page) is not GoogleDriveItemPage:
                raise ValueError("invalid page type")
            items_raw = page.items
            if type(items_raw) not in (list, tuple):
                raise ValueError("invalid items")
            rebuilt_items: list[GoogleDriveItem] = []
            for item in items_raw:
                if type(item) is not GoogleDriveItem:
                    raise ValueError("invalid item type")
                item_snapshot = item.model_dump(mode="python")
                scope_data = item_snapshot.get("scope")
                if isinstance(scope_data, dict):
                    item_snapshot["scope"] = GoogleDriveScope(**scope_data)
                rebuilt_items.append(GoogleDriveItem(**item_snapshot))
            next_token = None
            if page.next_page_token is not None:
                next_token = self._copy_page_token(page.next_page_token)
            validated = GoogleDriveItemPage(
                items=tuple(rebuilt_items),
                next_page_token=next_token,
            )
        except VendorKnowledgeError:
            raise
        except Exception:
            raise ValueError("invalid item page") from None
        seen_remote_ids: set[str] = set()
        for item in validated.items:
            if item.scope != scope:
                raise ValueError("item scope mismatch")
            if item.remote_id in seen_remote_ids:
                raise ValueError("duplicate remote id")
            seen_remote_ids.add(item.remote_id)
        return validated

    def _reconstruct_change_page(
        self,
        page: object,
        *,
        scope: GoogleDriveScope,
    ) -> GoogleDriveChangePage:
        try:
            if type(page) is not GoogleDriveChangePage:
                raise ValueError("invalid page type")
            changes_raw = page.changes
            if type(changes_raw) not in (list, tuple):
                raise ValueError("invalid changes")
            rebuilt_changes: list[GoogleDriveChange] = []
            for change in changes_raw:
                if type(change) is not GoogleDriveChange:
                    raise ValueError("invalid change type")
                change_snapshot = change.model_dump(mode="python")
                scope_data = change_snapshot.get("scope")
                if isinstance(scope_data, dict):
                    change_snapshot["scope"] = GoogleDriveScope(**scope_data)
                nested_item = change.item
                if nested_item is not None:
                    if type(nested_item) is not GoogleDriveItem:
                        raise ValueError("invalid nested item type")
                    item_snapshot = nested_item.model_dump(mode="python")
                    item_scope = item_snapshot.get("scope")
                    if isinstance(item_scope, dict):
                        item_snapshot["scope"] = GoogleDriveScope(**item_scope)
                    change_snapshot["item"] = GoogleDriveItem(**item_snapshot)
                rebuilt_changes.append(GoogleDriveChange(**change_snapshot))
            next_token = None
            if page.next_page_token is not None:
                next_token = self._copy_page_token(page.next_page_token)
            new_start = None
            if page.new_start_page_token is not None:
                new_start = self._copy_page_token(page.new_start_page_token)
            validated = GoogleDriveChangePage(
                changes=tuple(rebuilt_changes),
                next_page_token=next_token,
                new_start_page_token=new_start,
            )
        except VendorKnowledgeError:
            raise
        except Exception:
            raise ValueError("invalid change page") from None
        for change in validated.changes:
            if change.scope != scope:
                raise ValueError("change scope mismatch")
            if not change.removed and change.item is not None:
                if change.item.remote_id != change.file_id:
                    raise ValueError("active change item id mismatch")
        return validated

    def _item_to_change(self, item: GoogleDriveItem) -> KnowledgeChange:
        descriptor = self._item_to_descriptor(item)
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=item.remote_id,
            descriptor=descriptor,
        )

    def _change_to_knowledge_change(self, change: GoogleDriveChange) -> KnowledgeChange:
        if change.removed:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=change.file_id,
                descriptor=None,
            )
        if change.item is None:
            raise ValueError("active change missing item")
        descriptor = self._item_to_descriptor(change.item)
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=change.file_id,
            descriptor=descriptor,
        )

    def _resolve_content_profile(
        self,
        item: GoogleDriveItem,
    ) -> GoogleDriveContentProfile | None:
        try:
            return resolve_google_drive_content_profile(item)
        except GoogleDriveUnsupportedContent:
            return None

    def _parent_remote_id(self, parent_ids: tuple[str, ...]) -> str | None:
        if len(parent_ids) == 1:
            return parent_ids[0]
        return None

    def _item_to_descriptor(self, item: GoogleDriveItem) -> KnowledgeItemDescriptor:
        profile = self._resolve_content_profile(item)
        content_supported = profile is not None
        if profile is not None:
            content_mode = KnowledgeContentMode.BINARY
            content_available = item.can_download
            content_transport_mode = profile.mode.value
            content_mime_type = profile.content_mime_type
        else:
            content_mode = KnowledgeContentMode.STRUCTURED_RECORD
            content_available = False
            content_transport_mode = None
            content_mime_type = None

        metadata: dict[str, object] = {
            "schema_version": GOOGLE_DRIVE_ITEM_METADATA_VERSION,
            "scope_kind": item.scope.kind.value,
            "drive_id": item.drive_id,
            "item_kind": item.kind.value,
            "mime_type": item.mime_type,
            "parent_ids": list(item.parent_ids),
            "created_at": item.created_at.isoformat(),
            "modified_at": item.modified_at.isoformat(),
            "size_bytes": item.size_bytes,
            "md5_checksum": item.md5_checksum,
            "head_revision_id": item.head_revision_id,
            "can_download": item.can_download,
            "shortcut_target_id": item.shortcut_target_id,
            "shortcut_target_mime_type": item.shortcut_target_mime_type,
            "content_supported": content_supported,
            "content_transport_mode": content_transport_mode,
            "content_mime_type": content_mime_type,
        }

        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=item.remote_id,
                parent_remote_id=self._parent_remote_id(item.parent_ids),
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=str(item.version),
                etag=None,
                content_hash=None,
                acl_hash=None,
                updated_at=item.modified_at,
            ),
            title=item.name,
            item_type=_ITEM_TYPE_BY_KIND[item.kind],
            content_mode=content_mode,
            content_available=content_available,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=item.remote_id,
                web_url=item.web_view_link,
                safe_locator=None,
            ),
            metadata=metadata,
        )

    def _parse_metadata_datetime(self, raw: object) -> datetime:
        if not isinstance(raw, str):
            raise ValueError("timestamp must be ISO string")
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("timestamp must not be empty")
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return parsed

    def _reconstruct_descriptor(self, item: object) -> KnowledgeItemDescriptor:
        try:
            if type(item) is not KnowledgeItemDescriptor:
                raise ValueError("invalid descriptor type")
            metadata_raw = item.metadata
            if isinstance(metadata_raw, dict):
                parent_ids_raw = metadata_raw.get("parent_ids")
                if type(parent_ids_raw) is list:
                    for pid in parent_ids_raw:
                        if type(pid) is not str:
                            raise ValueError("invalid parent_ids element")
            snapshot = item.model_dump(mode="python")
            identity_data = snapshot.get("identity")
            if isinstance(identity_data, dict):
                snapshot["identity"] = KnowledgeItemIdentity(**identity_data)
            revision_data = snapshot.get("revision")
            if isinstance(revision_data, dict):
                snapshot["revision"] = KnowledgeItemRevision(**revision_data)
            provenance_data = snapshot.get("provenance")
            if isinstance(provenance_data, dict):
                snapshot["provenance"] = KnowledgeItemProvenance(**provenance_data)
            return KnowledgeItemDescriptor(**snapshot)
        except Exception:
            raise self._invalid_descriptor_error() from None

    def _descriptor_to_provider_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        scope: GoogleDriveScope,
    ) -> GoogleDriveItem:
        try:
            reconstructed = self._reconstruct_descriptor(item)
            self._validate_item_provenance(reconstructed, source=source)
            if reconstructed.identity.logical_key is not None:
                raise ValueError("logical_key must be None")
            if reconstructed.revision.etag is not None:
                raise ValueError("etag must be None")
            if reconstructed.revision.content_hash is not None:
                raise ValueError("content_hash must be None")
            if reconstructed.revision.acl_hash is not None:
                raise ValueError("acl_hash must be None")
            if reconstructed.provenance.safe_locator is not None:
                raise ValueError("safe_locator must be None")
            metadata = reconstructed.metadata
            if not isinstance(metadata, dict):
                raise ValueError("invalid metadata")
            if set(metadata.keys()) != _METADATA_KEYS:
                raise ValueError("invalid metadata keys")
            if metadata.get("schema_version") != GOOGLE_DRIVE_ITEM_METADATA_VERSION:
                raise ValueError("invalid metadata version")

            metadata_scope_kind = metadata.get("scope_kind")
            if metadata_scope_kind != scope.kind.value:
                raise ValueError("metadata scope mismatch")
            metadata_drive_id = metadata.get("drive_id")
            if scope.kind is GoogleDriveScopeKind.USER:
                if metadata_drive_id is not None:
                    raise ValueError("metadata drive_id mismatch")
            else:
                if metadata_drive_id != scope.drive_id:
                    raise ValueError("metadata drive_id mismatch")

            item_kind_raw = metadata.get("item_kind")
            if type(item_kind_raw) is not str:
                raise ValueError("invalid item_kind")
            try:
                item_kind = GoogleDriveItemKind(item_kind_raw)
            except ValueError:
                raise ValueError("invalid item_kind") from None

            expected_item_type = _ITEM_TYPE_BY_KIND[item_kind]
            if reconstructed.item_type != expected_item_type:
                raise ValueError("item type mismatch")

            parent_ids_raw = metadata.get("parent_ids")
            if type(parent_ids_raw) is not list:
                raise ValueError("invalid parent_ids")
            for pid in parent_ids_raw:
                if type(pid) is not str:
                    raise ValueError("invalid parent_ids element")
            parent_ids = tuple(parent_ids_raw)

            expected_parent = self._parent_remote_id(parent_ids)
            if reconstructed.identity.parent_remote_id != expected_parent:
                raise ValueError("parent projection mismatch")

            created_at = self._parse_metadata_datetime(metadata.get("created_at"))
            modified_at = self._parse_metadata_datetime(metadata.get("modified_at"))
            version_raw = reconstructed.revision.version
            if type(version_raw) is not str or not version_raw.strip():
                raise ValueError("invalid version")
            if version_raw != version_raw.strip():
                raise ValueError("invalid version")
            try:
                version = int(version_raw)
            except ValueError:
                raise ValueError("invalid version") from None

            size_bytes = metadata.get("size_bytes")
            if size_bytes is not None and type(size_bytes) is not int:
                raise ValueError("invalid size_bytes")
            md5_checksum = metadata.get("md5_checksum")
            if md5_checksum is not None and type(md5_checksum) is not str:
                raise ValueError("invalid md5_checksum")
            head_revision_id = metadata.get("head_revision_id")
            if head_revision_id is not None and type(head_revision_id) is not str:
                raise ValueError("invalid head_revision_id")
            can_download = metadata.get("can_download")
            if type(can_download) is not bool:
                raise ValueError("invalid can_download")
            shortcut_target_id = metadata.get("shortcut_target_id")
            if shortcut_target_id is not None and type(shortcut_target_id) is not str:
                raise ValueError("invalid shortcut_target_id")
            shortcut_target_mime_type = metadata.get("shortcut_target_mime_type")
            if shortcut_target_mime_type is not None and type(shortcut_target_mime_type) is not str:
                raise ValueError("invalid shortcut_target_mime_type")
            mime_type = metadata.get("mime_type")
            if type(mime_type) is not str:
                raise ValueError("invalid mime_type")

            content_supported = metadata.get("content_supported")
            if type(content_supported) is not bool:
                raise ValueError("invalid content_supported")
            content_transport_mode = metadata.get("content_transport_mode")
            if content_transport_mode is not None and type(content_transport_mode) is not str:
                raise ValueError("invalid content_transport_mode")
            content_mime_type_meta = metadata.get("content_mime_type")
            if content_mime_type_meta is not None and type(content_mime_type_meta) is not str:
                raise ValueError("invalid content_mime_type")

            provider_item = GoogleDriveItem(
                remote_id=reconstructed.identity.remote_id,
                scope=scope,
                kind=item_kind,
                name=reconstructed.title,
                mime_type=mime_type,
                parent_ids=parent_ids,
                drive_id=metadata_drive_id,
                created_at=created_at,
                modified_at=modified_at,
                size_bytes=size_bytes,
                md5_checksum=md5_checksum,
                version=version,
                head_revision_id=head_revision_id,
                web_view_link=reconstructed.provenance.web_url,
                can_download=can_download,
                shortcut_target_id=shortcut_target_id,
                shortcut_target_mime_type=shortcut_target_mime_type,
            )

            profile = self._resolve_content_profile(provider_item)
            if content_supported:
                if profile is None:
                    raise ValueError("content profile mismatch")
                if reconstructed.content_mode is not KnowledgeContentMode.BINARY:
                    raise ValueError("content mode mismatch")
                if reconstructed.content_available != can_download:
                    raise ValueError("content availability mismatch")
                if content_transport_mode != profile.mode.value:
                    raise ValueError("content transport mode mismatch")
                if content_mime_type_meta != profile.content_mime_type:
                    raise ValueError("content mime mismatch")
            else:
                if profile is not None:
                    raise ValueError("content profile mismatch")
                if reconstructed.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
                    raise ValueError("content mode mismatch")
                if reconstructed.content_available:
                    raise ValueError("content availability mismatch")
                if content_transport_mode is not None or content_mime_type_meta is not None:
                    raise ValueError("content profile metadata mismatch")

            if not reconstructed.content_available:
                raise ValueError("content not available")

            if reconstructed.identity.remote_id != provider_item.remote_id:
                raise ValueError("remote id mismatch")
            if reconstructed.revision.version != str(provider_item.version):
                raise ValueError("version mismatch")
            if reconstructed.revision.updated_at != provider_item.modified_at:
                raise ValueError("timestamp mismatch")
            if reconstructed.provenance.remote_id != provider_item.remote_id:
                raise ValueError("provenance remote id mismatch")
            if reconstructed.provenance.web_url != provider_item.web_view_link:
                raise ValueError("web url mismatch")

            return provider_item
        except VendorKnowledgeError:
            raise
        except Exception:
            raise self._invalid_descriptor_error() from None

    def _validate_fetched_content(
        self,
        result: object,
        *,
        provider_item: GoogleDriveItem,
    ) -> None:
        try:
            if type(result) is not GoogleDriveFileContent:
                raise ValueError("content must be GoogleDriveFileContent")
            snapshot = result.model_dump(mode="python")
            item_data = snapshot.get("item")
            if isinstance(item_data, dict):
                scope_data = item_data.get("scope")
                if isinstance(scope_data, dict):
                    item_data = dict(item_data)
                    item_data["scope"] = GoogleDriveScope(**scope_data)
                snapshot["item"] = GoogleDriveItem(**item_data)
            validated = GoogleDriveFileContent(**snapshot)
            result_item = validated.item
            if result_item.remote_id != provider_item.remote_id:
                raise ValueError("remote id mismatch")
            if result_item.scope != provider_item.scope:
                raise ValueError("scope mismatch")
            if result_item.kind != provider_item.kind:
                raise ValueError("kind mismatch")
            if result_item.mime_type != provider_item.mime_type:
                raise ValueError("mime_type mismatch")
            if result_item.version != provider_item.version:
                raise ValueError("version mismatch")
            if result_item.modified_at != provider_item.modified_at:
                raise ValueError("modified_at mismatch")
            if result_item.size_bytes != provider_item.size_bytes:
                raise ValueError("size_bytes mismatch")
            if result_item.md5_checksum != provider_item.md5_checksum:
                raise ValueError("md5_checksum mismatch")
            if result_item.head_revision_id != provider_item.head_revision_id:
                raise ValueError("head_revision_id mismatch")
            if result_item.can_download != provider_item.can_download:
                raise ValueError("can_download mismatch")
            profile = resolve_google_drive_content_profile(provider_item)
            if validated.mode != profile.mode:
                raise ValueError("mode mismatch")
            if validated.content_mime_type != profile.content_mime_type:
                raise ValueError("mime mismatch")
            if validated.size_bytes != len(validated.data):
                raise ValueError("size mismatch")
            expected_hash = hashlib.sha256(validated.data).hexdigest()
            if validated.content_hash != expected_hash:
                raise ValueError("hash mismatch")
        except Exception:
            raise self._invalid_provider_response_error() from None

    def _encode_cursor(self, cursor: _GoogleDriveCursor) -> KnowledgeCursor:
        payload = cursor.model_dump()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return KnowledgeCursor(value=encoded, version=GOOGLE_DRIVE_CURSOR_VERSION)

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        scope: GoogleDriveScope,
    ) -> _GoogleDriveCursor | None:
        if cursor is None:
            return None
        try:
            if type(cursor) is not KnowledgeCursor:
                raise ValueError("invalid cursor type")
            snapshot = cursor.model_dump(mode="python")
            reconstructed_cursor = KnowledgeCursor(**snapshot)
            outer_value = reconstructed_cursor.value
            if type(outer_value) is not str:
                raise ValueError("invalid cursor value type")
            if not outer_value:
                raise ValueError("blank cursor value")
            if outer_value != outer_value.strip():
                raise ValueError("cursor whitespace")
            if len(outer_value) > _MAX_ENCODED_CURSOR_LENGTH:
                raise ValueError("cursor too long")
            if "=" in outer_value:
                raise ValueError("cursor padding")
            if _CURSOR_ALPHABET.fullmatch(outer_value) is None:
                raise ValueError("cursor alphabet")
            if reconstructed_cursor.version != GOOGLE_DRIVE_CURSOR_VERSION:
                raise ValueError("cursor version mismatch")
            padding = "=" * (-len(outer_value) % 4)
            raw = base64.b64decode(outer_value + padding, altchars=b"-_", validate=True)
            data = json.loads(raw.decode("utf-8"))
            if type(data) is not dict:
                raise ValueError("cursor must be object")
            decoded = _GoogleDriveCursor.model_validate(data)
            canonical = self._encode_cursor(decoded)
            if (
                canonical.value != reconstructed_cursor.value
                or canonical.version != reconstructed_cursor.version
            ):
                raise ValueError("noncanonical cursor")
        except Exception:
            raise self._invalid_cursor_error() from None
        if decoded.scope_kind == "user":
            if scope.kind is not GoogleDriveScopeKind.USER:
                raise self._invalid_cursor_error()
        elif decoded.scope_kind == "shared_drive":
            if (
                scope.kind is not GoogleDriveScopeKind.SHARED_DRIVE
                or decoded.drive_id != scope.drive_id
            ):
                raise self._invalid_cursor_error()
        return decoded

    def _safe_message_for_code(self, code: VendorKnowledgeErrorCode) -> str:
        if code is VendorKnowledgeErrorCode.INVALID_CURSOR:
            return _INVALID_CURSOR_MESSAGE
        if code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR:
            return _CONFIGURATION_ERROR_MESSAGE
        if code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE:
            return _DEPENDENCY_UNAVAILABLE_MESSAGE
        if code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY:
            return _UNSUPPORTED_PERMISSIONS_MESSAGE
        return _INVALID_PROVIDER_RESPONSE_MESSAGE

    def _map_google_api_error(
        self,
        exc: GoogleWorkspaceApiError,
        *,
        cursor_in_use: bool,
    ) -> VendorKnowledgeError:
        if exc.kind is GoogleWorkspaceErrorKind.AUTHENTICATION:
            code = VendorKnowledgeErrorCode.AUTHENTICATION_FAILED
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.AUTHORIZATION:
            code = VendorKnowledgeErrorCode.AUTHORIZATION_DENIED
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.NOT_FOUND:
            code = VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.RATE_LIMITED:
            code = VendorKnowledgeErrorCode.RATE_LIMITED
            retryable = True
        elif exc.kind is GoogleWorkspaceErrorKind.TEMPORARY:
            code = VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
            retryable = True
        elif exc.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE:
            code = VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT:
            code = VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE:
            code = VendorKnowledgeErrorCode.CONFIGURATION_ERROR
            retryable = False
        elif exc.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST:
            if cursor_in_use and exc.status_code in {400, 410}:
                code = VendorKnowledgeErrorCode.INVALID_CURSOR
            else:
                code = VendorKnowledgeErrorCode.CONFIGURATION_ERROR
            retryable = False
        else:
            code = VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
            retryable = True
        return VendorKnowledgeError(
            code=code,
            safe_message=self._safe_message_for_code(code),
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=retryable,
        )

    async def _invoke_integration(
        self,
        operation: Callable[[], _T],
        *,
        cursor_in_use: bool,
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except GoogleDriveContentChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_DEPENDENCY_UNAVAILABLE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except GoogleDriveContentTooLarge:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except GoogleDriveContentUnavailable:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except GoogleDriveUnsupportedContent:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except GoogleWorkspaceApiError as exc:
            raise self._map_google_api_error(exc, cursor_in_use=cursor_in_use) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_DEPENDENCY_UNAVAILABLE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_DEPENDENCY_UNAVAILABLE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None


def register_google_workspace_drive_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> GoogleWorkspaceDriveKnowledgeAdapter:
    adapter = GoogleWorkspaceDriveKnowledgeAdapter()
    registry.register(adapter)
    return adapter
