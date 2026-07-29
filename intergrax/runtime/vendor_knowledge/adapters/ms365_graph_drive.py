# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Drive knowledge source adapter (MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MSGRAPH_DRIVE_SOURCE_KIND,
    MsGraphDriveDeltaPage,
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    validate_msgraph_drive_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive_content import (
    DEFAULT_DRIVE_CONTENT_MAX_BYTES,
    MsGraphDriveFileContent,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
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
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

MSGRAPH_DRIVE_SCOPE_TYPE = "msgraph_drive"
MSGRAPH_DRIVE_CURSOR_VERSION = "msgraph.drive.cursor.v1"

_MAX_CONTINUATION_URL_LEN = 32_768
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PROVIDER_PAGE_LIMIT = 200

_T = TypeVar("_T")


class _MsGraphDriveCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.drive.cursor.v1"]
    drive_id: str
    continuation_kind: Literal["next_page", "delta"]
    continuation_url: str = Field(repr=False)

    @field_validator("drive_id", mode="before")
    @classmethod
    def _validate_drive_id(cls, value: object) -> str:
        return validate_msgraph_drive_id(value)

    @field_validator("continuation_url", mode="before")
    @classmethod
    def _validate_continuation_url(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("continuation_url must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("continuation_url must not be empty")
        if _ASCII_CONTROL.search(cleaned):
            raise ValueError("continuation_url must not contain control characters")
        if len(cleaned) > _MAX_CONTINUATION_URL_LEN:
            raise ValueError("continuation_url exceeds maximum length")
        return cleaned


class MsGraphDriveKnowledgeAdapter:
    """Thin mapping from Microsoft Graph Drive integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return MSGRAPH_DRIVE_SOURCE_KIND

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
        self._require_graph_integration(integration=integration, source=source)
        self._validate_source(source)
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self.capabilities,
            safe_display_name=source.scope.safe_display_name,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        graph_integration = self._require_graph_integration(integration=integration, source=source)
        drive_id = self._validate_source(source)
        provider_limit = self._validate_limit(limit)
        decoded = self._decode_cursor(cursor, drive_id=drive_id)
        provider_continuation = self._to_provider_continuation(decoded)
        page = await self._invoke_integration(
            lambda: graph_integration.read_drive_delta_page(
                drive_id=drive_id,
                continuation=provider_continuation,
                limit=provider_limit,
            )
        )
        try:
            self._validate_page_for_source(page, drive_id=drive_id)
            changes = tuple(self._item_to_change(item) for item in page.items)
            encoded_checkpoint = self._encode_cursor_from_continuation(
                drive_id=drive_id,
                continuation=page.continuation,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Drive knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if page.continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            return KnowledgePage(
                changes=changes,
                next_cursor=encoded_checkpoint,
                proposed_checkpoint=encoded_checkpoint,
                has_more=True,
            )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=encoded_checkpoint,
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        graph_integration = self._require_graph_integration(integration=integration, source=source)
        drive_id = self._validate_source(source)
        try:
            self._validate_file_item(item, source=source, drive_id=drive_id)
            provider_item = self._descriptor_to_provider_item(item, drive_id=drive_id)
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message=(
                    "Microsoft Graph Drive file descriptor "
                    "is invalid"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        result = await self._invoke_integration(
            lambda: graph_integration.read_drive_file_content(
                item=provider_item,
                max_bytes=DEFAULT_DRIVE_CONTENT_MAX_BYTES,
            )
        )
        try:
            self._validate_fetched_content(result, item=item, drive_id=drive_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=(
                    "Microsoft Graph Drive content identity "
                    "does not match requested item"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        return KnowledgeContent(
            mode=KnowledgeContentMode.BINARY,
            binary=result.data,
            mime_type=result.mime_type,
            content_hash=result.content_hash,
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self._require_graph_integration(integration=integration, source=source)
        self._validate_source(source)
        self._validate_item_provenance(item, source=source)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=(
                "Microsoft Graph Drive authoritative permission "
                "projection is not implemented"
            ),
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _require_graph_integration(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> Ms365GraphCollaborationSuiteIntegration:
        if not isinstance(integration, Ms365GraphCollaborationSuiteIntegration):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=(
                    "Microsoft Graph Drive knowledge adapter requires "
                    "Microsoft Graph collaboration-suite integration"
                ),
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return integration

    def _validate_source(self, source: KnowledgeSourceRef) -> str:
        if (
            source.provider_id != self.provider_id
            or source.integration_kind != self.integration_kind
            or source.source_kind != self.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        scope = source.scope
        if scope.remote_scope_type != MSGRAPH_DRIVE_SCOPE_TYPE:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if scope.parameters:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        try:
            return validate_msgraph_drive_id(scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Drive knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Drive knowledge page limit is invalid",
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
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Item provenance does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_file_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        drive_id: str,
    ) -> None:
        self._validate_item_provenance(item, source=source)
        if item.item_type != "msgraph_drive_file":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive knowledge item type is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.BINARY:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file content mode must be binary",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file content is not available",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        remote_id = item.identity.remote_id
        if not isinstance(remote_id, str) or not remote_id.strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file remote id is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        version = item.revision.version
        if not isinstance(version, str) or not version.strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file revision is missing",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.revision.updated_at is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file revision timestamp is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file metadata is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        metadata_drive_id = metadata.get("drive_id")
        if not isinstance(metadata_drive_id, str):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file metadata is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        try:
            if validate_msgraph_drive_id(metadata_drive_id) != drive_id:
                raise ValueError("drive id mismatch")
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file metadata is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        drive_item_kind = metadata.get("drive_item_kind")
        if drive_item_kind != "file":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file metadata is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        size_bytes = metadata.get("size_bytes")
        if size_bytes is not None and (type(size_bytes) is not int or size_bytes < 0):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file metadata is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        mime_type = metadata.get("mime_type")
        if mime_type is not None and (
            not isinstance(mime_type, str) or not mime_type.strip()
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Drive file metadata is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_page_for_source(
        self,
        page: object,
        *,
        drive_id: str,
    ) -> None:
        if not isinstance(page, MsGraphDriveDeltaPage):
            raise ValueError("page must be a MsGraphDriveDeltaPage")
        if page.continuation.kind not in {
            MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            MsGraphKnowledgeContinuationKind.DELTA,
        }:
            raise ValueError("invalid continuation kind")
        seen_remote_ids: set[str] = set()
        for item in page.items:
            if not isinstance(item, MsGraphDriveItem):
                raise ValueError("item must be a MsGraphDriveItem")
            if item.remote_id in seen_remote_ids:
                raise ValueError("duplicate remote id on page")
            seen_remote_ids.add(item.remote_id)
            if item.drive_id != drive_id:
                raise ValueError("item drive id does not match source")
            if item.kind == MsGraphDriveItemKind.DELETED:
                continue
            if item.name is None or not item.name.strip():
                raise ValueError("active item name is required")
            if item.last_modified_at is None:
                raise ValueError("active item last_modified_at is required")
            if item.kind == MsGraphDriveItemKind.FILE and not item.c_tag:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Microsoft Graph Drive file revision is missing",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                )

    def _item_to_change(self, item: MsGraphDriveItem) -> KnowledgeChange:
        if item.kind == MsGraphDriveItemKind.DELETED:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=item.remote_id,
                descriptor=None,
            )
        descriptor = self._item_to_descriptor(item)
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=item.remote_id,
            descriptor=descriptor,
        )

    def _item_to_descriptor(self, item: MsGraphDriveItem) -> KnowledgeItemDescriptor:
        if item.kind == MsGraphDriveItemKind.FILE:
            item_type = "msgraph_drive_file"
            content_mode = KnowledgeContentMode.BINARY
            content_available = True
        elif item.kind == MsGraphDriveItemKind.FOLDER:
            item_type = "msgraph_drive_folder"
            content_mode = KnowledgeContentMode.STRUCTURED_RECORD
            content_available = False
        elif item.kind == MsGraphDriveItemKind.PACKAGE:
            item_type = "msgraph_drive_package"
            content_mode = KnowledgeContentMode.STRUCTURED_RECORD
            content_available = False
        else:
            item_type = "msgraph_drive_other"
            content_mode = KnowledgeContentMode.STRUCTURED_RECORD
            content_available = False

        metadata: dict[str, object] = {
            "drive_id": item.drive_id,
            "drive_item_kind": item.kind.value,
            "is_root": item.is_root,
        }
        if item.size_bytes is not None:
            metadata["size_bytes"] = item.size_bytes
        if item.mime_type is not None:
            metadata["mime_type"] = item.mime_type
        if item.created_at is not None:
            metadata["created_at"] = item.created_at.isoformat()
        if item.last_modified_at is not None:
            metadata["last_modified_at"] = item.last_modified_at.isoformat()

        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=item.remote_id,
                parent_remote_id=item.parent_remote_id,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=item.c_tag,
                etag=item.e_tag,
                updated_at=item.last_modified_at,
            ),
            title=item.name or "",
            item_type=item_type,
            content_mode=content_mode,
            content_available=content_available,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=item.remote_id,
                web_url=item.web_url,
                safe_locator=None,
            ),
            metadata=metadata,
        )

    def _parse_auxiliary_created_at(self, raw: object) -> datetime | None:
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise ValueError("created_at must be an ISO string")
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("created_at must not be empty")
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("created_at must be valid ISO-8601") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return parsed

    def _parse_auxiliary_is_root(self, raw: object) -> bool:
        if raw is None:
            return False
        if type(raw) is not bool:
            raise ValueError("is_root must be a bool")
        return raw

    def _descriptor_to_provider_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        drive_id: str,
    ) -> MsGraphDriveItem:
        metadata = item.metadata or {}
        size_bytes = metadata.get("size_bytes")
        if size_bytes is not None and type(size_bytes) is not int:
            raise ValueError("size_bytes must be an int")
        mime_type = metadata.get("mime_type")
        if mime_type is not None and not isinstance(mime_type, str):
            raise ValueError("mime_type must be a string")
        created_at = self._parse_auxiliary_created_at(metadata.get("created_at"))
        is_root = self._parse_auxiliary_is_root(metadata.get("is_root"))
        return MsGraphDriveItem(
            remote_id=item.identity.remote_id,
            drive_id=drive_id,
            parent_remote_id=item.identity.parent_remote_id,
            kind=MsGraphDriveItemKind.FILE,
            name=item.title,
            e_tag=item.revision.etag,
            c_tag=item.revision.version,
            size_bytes=size_bytes,
            mime_type=mime_type,
            created_at=created_at,
            last_modified_at=item.revision.updated_at,
            web_url=item.provenance.web_url,
            is_root=is_root,
            deleted_state=None,
        )

    def _validate_fetched_content(
        self,
        result: object,
        *,
        item: KnowledgeItemDescriptor,
        drive_id: str,
    ) -> None:
        if not isinstance(result, MsGraphDriveFileContent):
            raise ValueError("content must be MsGraphDriveFileContent")
        if result.drive_id != drive_id:
            raise ValueError("drive id mismatch")
        if result.remote_id != item.identity.remote_id:
            raise ValueError("remote id mismatch")
        if result.content_revision != item.revision.version:
            raise ValueError("content revision mismatch")
        if result.size_bytes != len(result.data):
            raise ValueError("size mismatch")
        expected_hash = hashlib.sha256(result.data).hexdigest()
        if result.content_hash != expected_hash:
            raise ValueError("hash mismatch")

    def _encode_cursor(self, cursor: _MsGraphDriveCursor) -> KnowledgeCursor:
        payload = cursor.model_dump()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return KnowledgeCursor(value=encoded, version=MSGRAPH_DRIVE_CURSOR_VERSION)

    def _encode_cursor_from_continuation(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation,
    ) -> KnowledgeCursor:
        if continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            continuation_kind: Literal["next_page", "delta"] = "next_page"
        else:
            continuation_kind = "delta"
        return self._encode_cursor(
            _MsGraphDriveCursor(
                schema_version=MSGRAPH_DRIVE_CURSOR_VERSION,
                drive_id=drive_id,
                continuation_kind=continuation_kind,
                continuation_url=continuation.url,
            )
        )

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        drive_id: str,
    ) -> _MsGraphDriveCursor | None:
        if cursor is None:
            return None
        if cursor.version != MSGRAPH_DRIVE_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Drive knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            padding = "=" * (-len(cursor.value) % 4)
            raw = base64.urlsafe_b64decode(cursor.value + padding)
            data = json.loads(raw.decode("utf-8"))
            decoded = _MsGraphDriveCursor.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Drive knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if decoded.drive_id != drive_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Drive knowledge cursor scope "
                    "does not match source"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded

    def _to_provider_continuation(
        self,
        decoded: _MsGraphDriveCursor | None,
    ) -> MsGraphKnowledgeContinuation | None:
        if decoded is None:
            return None
        if decoded.continuation_kind == "next_page":
            kind = MsGraphKnowledgeContinuationKind.NEXT_PAGE
        else:
            kind = MsGraphKnowledgeContinuationKind.DELTA
        return MsGraphKnowledgeContinuation(kind=kind, url=decoded.continuation_url)

    async def _invoke_integration(
        self,
        operation: Callable[[], _T],
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Drive knowledge adapter configuration is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Drive knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Drive knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Drive knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None


def register_msgraph_drive_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> MsGraphDriveKnowledgeAdapter:
    adapter = MsGraphDriveKnowledgeAdapter()
    registry.register(adapter)
    return adapter
