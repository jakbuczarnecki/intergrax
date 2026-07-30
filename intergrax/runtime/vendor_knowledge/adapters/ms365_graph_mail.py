# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Mail knowledge source adapter (MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL).

REMOVED delta tombstones mean the message is no longer present in the synchronized
mailbox-folder source view. They do not claim global mailbox deletion or permanent
removal from Microsoft 365.
"""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_content import (
    DEFAULT_MAIL_CONTENT_MAX_CHARS,
    MsGraphMailContentTooLarge,
    MsGraphMailMessageChanged,
    MsGraphMailMessageContent,
    MsGraphMailParticipant,
    validate_msgraph_mail_message_content,
    validate_msgraph_mail_participant,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
    validate_msgraph_mail_folder_id,
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_messages import (
    MsGraphMailImportance,
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    MsGraphMailMessageDeltaPage,
    validate_msgraph_mail_message_change,
    validate_msgraph_mail_message_id,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    JsonObject,
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

MSGRAPH_MAIL_SCOPE_TYPE = "msgraph_mail_folder"
MSGRAPH_MAIL_CURSOR_VERSION = "msgraph.mail.cursor.v1"
_MSGRAPH_MAIL_SCOPE_SCHEMA_VERSION = "msgraph.mail.scope.v1"
_MSGRAPH_MAIL_MESSAGE_ID_SCHEMA_VERSION = "msgraph.mail.message-id.v1"
_MSGRAPH_MAIL_REVISION_SCHEMA_VERSION = "msgraph.mail.revision.v1"

_STRUCTURED_RECORD_SCHEMA = "msgraph.mail.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-mail-message+json"
_SAFE_TITLE_FALLBACK = "Mail message"
_REMOVAL_SEMANTICS = "removed_from_synchronized_folder_view"

_MAX_CONTINUATION_URL_LEN = 32_768
_MAX_CHANGE_KEY_LEN = 2048
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PROVIDER_PAGE_LIMIT = 200

_METADATA_REQUIRED_KEYS = frozenset(
    {
        "message_state",
        "is_read",
        "is_draft",
        "has_attachments",
        "importance",
        "attachment_inventory_included",
        "attachment_content_included",
        "removal_semantics",
    }
)
_METADATA_OPTIONAL_TIMESTAMP_KEYS = frozenset(
    {"created_at", "received_at", "sent_at", "last_modified_at"}
)
_METADATA_ALLOWED_KEYS = _METADATA_REQUIRED_KEYS | _METADATA_OPTIONAL_TIMESTAMP_KEYS

_T = TypeVar("_T")


class _MsGraphMailFolderScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.mail.scope.v1"]
    mailbox_user_id: str = Field(repr=False)
    folder_id: str = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("folder_id", mode="before")
    @classmethod
    def _validate_folder_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)


class _MsGraphMailMessageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.mail.message-id.v1"]
    mailbox_user_id: str = Field(repr=False)
    folder_id: str = Field(repr=False)
    message_id: str = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("folder_id", mode="before")
    @classmethod
    def _validate_folder_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)

    @field_validator("message_id", mode="before")
    @classmethod
    def _validate_message_id(cls, value: object) -> str:
        return validate_msgraph_mail_message_id(value)


class _MsGraphMailRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.mail.revision.v1"]
    change_key: str = Field(repr=False)

    @field_validator("change_key", mode="before")
    @classmethod
    def _validate_change_key(cls, value: object) -> str:
        return _validate_opaque_change_key(value)


class _MsGraphMailCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.mail.cursor.v1"]
    mailbox_user_id: str = Field(repr=False)
    folder_id: str = Field(repr=False)
    continuation_kind: Literal["next_page", "delta"]
    continuation_url: str = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("folder_id", mode="before")
    @classmethod
    def _validate_folder_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)

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


def encode_msgraph_mail_folder_scope_id(
    *,
    mailbox_user_id: str,
    folder_id: str,
) -> str:
    """Return the canonical opaque remote_scope_id for one mailbox-folder source."""
    scope = _MsGraphMailFolderScope(
        schema_version=_MSGRAPH_MAIL_SCOPE_SCHEMA_VERSION,
        mailbox_user_id=mailbox_user_id,
        folder_id=folder_id,
    )
    return _encode_canonical_payload(scope.model_dump())


class MsGraphMailKnowledgeAdapter:
    """Thin mapping from Microsoft Graph Mail integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return MSGRAPH_MAIL_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=True,
            content_fetch=True,
            binary_content=False,
            rich_text_content=False,
            structured_content=True,
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
        mailbox_user_id, folder_id = self._validate_source(source)
        provider_limit = self._validate_limit(limit)
        decoded = self._decode_cursor(
            cursor,
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
        )
        provider_continuation = self._to_provider_continuation(decoded)
        page = await self._invoke_integration(
            lambda: graph_integration.read_mail_messages_delta_page(
                mailbox_user_id=mailbox_user_id,
                folder_id=folder_id,
                continuation=provider_continuation,
                limit=provider_limit,
            )
        )
        try:
            self._validate_page_for_source(
                page,
                mailbox_user_id=mailbox_user_id,
                folder_id=folder_id,
            )
            changes = tuple(self._change_to_knowledge_change(item) for item in page.items)
            encoded_checkpoint = self._encode_cursor_from_continuation(
                mailbox_user_id=mailbox_user_id,
                folder_id=folder_id,
                continuation=page.continuation,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Mail knowledge provider response is invalid",
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
        mailbox_user_id, folder_id = self._validate_source(source)
        try:
            message_identity, revision = self._validate_message_item(
                item,
                source=source,
                mailbox_user_id=mailbox_user_id,
                folder_id=folder_id,
            )
            metadata = self._validate_descriptor_metadata(item.metadata)
            provider_message = self._descriptor_to_provider_message(
                message_identity=message_identity,
                revision=revision,
                metadata=metadata,
                updated_at=item.revision.updated_at,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        result = await self._invoke_integration(
            lambda: graph_integration.read_mail_message_content(
                message=provider_message,
                max_chars=DEFAULT_MAIL_CONTENT_MAX_CHARS,
            ),
            content_errors=True,
        )
        try:
            validated_content = validate_msgraph_mail_message_content(
                result,
                message=provider_message,
                max_chars=DEFAULT_MAIL_CONTENT_MAX_CHARS,
            )
            self._validate_fetched_content_identity(
                validated_content,
                message_identity=message_identity,
                revision=revision,
            )
        except MsGraphMailMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Mail knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Mail knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        structured_record = self._build_structured_record(
            content=validated_content,
            metadata=metadata,
            updated_at=item.revision.updated_at,
        )
        canonical = json.dumps(
            structured_record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        content_hash = hashlib.sha256(canonical).hexdigest()
        return KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record=structured_record,
            mime_type=_STRUCTURED_RECORD_MIME,
            content_hash=content_hash,
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
                "Microsoft Graph Mail authoritative permission projection is not implemented"
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
                    "Microsoft Graph Mail knowledge adapter requires "
                    "Microsoft Graph collaboration-suite integration"
                ),
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return integration

    def _validate_source(self, source: KnowledgeSourceRef) -> tuple[str, str]:
        if (
            source.provider_id != self.provider_id
            or source.integration_kind != self.integration_kind
            or source.source_kind != self.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        scope = source.scope
        if scope.remote_scope_type != MSGRAPH_MAIL_SCOPE_TYPE:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if scope.parameters:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        try:
            decoded = _decode_scope_payload(scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail knowledge source scope is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        return decoded.mailbox_user_id, decoded.folder_id

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Mail knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Mail knowledge page limit is invalid",
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

    def _validate_page_for_source(
        self,
        page: object,
        *,
        mailbox_user_id: str,
        folder_id: str,
    ) -> None:
        if not isinstance(page, MsGraphMailMessageDeltaPage):
            raise ValueError("page must be a MsGraphMailMessageDeltaPage")
        if page.continuation.kind not in {
            MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            MsGraphKnowledgeContinuationKind.DELTA,
        }:
            raise ValueError("invalid continuation kind")
        seen_remote_ids: set[str] = set()
        for item in page.items:
            if not isinstance(item, MsGraphMailMessageChange):
                raise ValueError("item must be a MsGraphMailMessageChange")
            validated = validate_msgraph_mail_message_change(item)
            if (
                validated.mailbox_user_id != mailbox_user_id
                or validated.scope_folder_id != folder_id
            ):
                raise ValueError("item scope does not match source")
            opaque_id = self._encode_message_identity(
                mailbox_user_id=mailbox_user_id,
                folder_id=folder_id,
                message_id=validated.remote_id,
            )
            if opaque_id in seen_remote_ids:
                raise ValueError("duplicate remote id on page")
            seen_remote_ids.add(opaque_id)
            if validated.kind == MsGraphMailMessageChangeKind.ACTIVE:
                if validated.change_key is None or validated.last_modified_at is None:
                    raise ValueError("active item missing revision")
            elif validated.kind == MsGraphMailMessageChangeKind.REMOVED:
                if validated.removed_reason is None:
                    raise ValueError("removed item missing reason")

    def _change_to_knowledge_change(self, item: MsGraphMailMessageChange) -> KnowledgeChange:
        opaque_id = self._encode_message_identity(
            mailbox_user_id=item.mailbox_user_id,
            folder_id=item.scope_folder_id,
            message_id=item.remote_id,
        )
        if item.kind == MsGraphMailMessageChangeKind.REMOVED:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=opaque_id,
                descriptor=None,
            )
        descriptor = self._active_message_to_descriptor(item, opaque_remote_id=opaque_id)
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=opaque_id,
            descriptor=descriptor,
        )

    def _active_message_to_descriptor(
        self,
        message: MsGraphMailMessageChange,
        *,
        opaque_remote_id: str,
    ) -> KnowledgeItemDescriptor:
        title = _resolve_message_title(message.subject)
        assert message.change_key is not None
        assert message.last_modified_at is not None
        assert message.is_read is not None
        assert message.is_draft is not None
        assert message.has_attachments is not None
        assert message.importance is not None

        metadata: dict[str, object] = {
            "message_state": "active",
            "is_read": message.is_read,
            "is_draft": message.is_draft,
            "has_attachments": message.has_attachments,
            "importance": message.importance.value,
            "attachment_inventory_included": False,
            "attachment_content_included": False,
            "removal_semantics": _REMOVAL_SEMANTICS,
        }
        if message.created_at is not None:
            metadata["created_at"] = message.created_at.isoformat()
        if message.received_at is not None:
            metadata["received_at"] = message.received_at.isoformat()
        if message.sent_at is not None:
            metadata["sent_at"] = message.sent_at.isoformat()
        metadata["last_modified_at"] = message.last_modified_at.isoformat()

        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=opaque_remote_id,
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=self._encode_revision(message.change_key),
                etag=None,
                updated_at=message.last_modified_at,
            ),
            title=title,
            item_type="msgraph_mail_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=opaque_remote_id,
                web_url=None,
                safe_locator=None,
            ),
            metadata=metadata,
        )

    def _validate_message_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        mailbox_user_id: str,
        folder_id: str,
    ) -> tuple[_MsGraphMailMessageIdentity, _MsGraphMailRevision]:
        self._validate_item_provenance(item, source=source)
        if item.item_type != "msgraph_mail_message":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        message_identity = self._decode_message_identity(item.identity.remote_id)
        if (
            message_identity.mailbox_user_id != mailbox_user_id
            or message_identity.folder_id != folder_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        version = item.revision.version
        if not isinstance(version, str) or not version.strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.revision.updated_at is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if (
            item.revision.updated_at.tzinfo is None
            or item.revision.updated_at.utcoffset() is None
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        revision = self._decode_revision(version)
        return message_identity, revision

    def _validate_descriptor_metadata(self, metadata: object) -> dict[str, object]:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        keys = set(metadata.keys())
        if not _METADATA_REQUIRED_KEYS.issubset(keys):
            raise ValueError("metadata missing required key")
        unknown_keys = keys - _METADATA_ALLOWED_KEYS
        if unknown_keys:
            raise ValueError("metadata keys are invalid")
        if metadata["message_state"] != "active":
            raise ValueError("message_state must be active")
        for bool_key in ("is_read", "is_draft", "has_attachments"):
            if type(metadata[bool_key]) is not bool:
                raise ValueError(f"{bool_key} must be bool")
        if metadata["attachment_inventory_included"] is not False:
            raise ValueError("attachment_inventory_included must be False")
        if metadata["attachment_content_included"] is not False:
            raise ValueError("attachment_content_included must be False")
        if metadata["removal_semantics"] != _REMOVAL_SEMANTICS:
            raise ValueError("removal_semantics is invalid")
        importance_raw = metadata["importance"]
        if not isinstance(importance_raw, str):
            raise ValueError("importance must be a string")
        try:
            MsGraphMailImportance(importance_raw)
        except ValueError:
            raise ValueError("importance is invalid") from None
        for timestamp_key in _METADATA_OPTIONAL_TIMESTAMP_KEYS:
            if timestamp_key not in metadata:
                continue
            raw_value = metadata[timestamp_key]
            if raw_value is None:
                continue
            self._parse_timezone_aware_iso(raw_value)
        return metadata

    def _descriptor_to_provider_message(
        self,
        *,
        message_identity: _MsGraphMailMessageIdentity,
        revision: _MsGraphMailRevision,
        metadata: dict[str, object],
        updated_at: datetime | None,
    ) -> MsGraphMailMessageChange:
        if updated_at is None:
            raise ValueError("updated_at is required")
        metadata_last_modified = metadata.get("last_modified_at")
        if metadata_last_modified is not None:
            parsed_last_modified = self._parse_timezone_aware_iso(metadata_last_modified)
            if parsed_last_modified != updated_at:
                raise ValueError("last_modified_at mismatch")
        importance = MsGraphMailImportance(str(metadata["importance"]))
        return MsGraphMailMessageChange(
            mailbox_user_id=message_identity.mailbox_user_id,
            scope_folder_id=message_identity.folder_id,
            remote_id=message_identity.message_id,
            kind=MsGraphMailMessageChangeKind.ACTIVE,
            parent_folder_id=message_identity.folder_id,
            change_key=revision.change_key,
            last_modified_at=updated_at,
            is_read=metadata["is_read"],
            is_draft=metadata["is_draft"],
            has_attachments=metadata["has_attachments"],
            importance=importance,
        )

    def _validate_fetched_content_identity(
        self,
        content: MsGraphMailMessageContent,
        *,
        message_identity: _MsGraphMailMessageIdentity,
        revision: _MsGraphMailRevision,
    ) -> None:
        if content.mailbox_user_id != message_identity.mailbox_user_id:
            raise ValueError("mailbox mismatch")
        if content.parent_folder_id != message_identity.folder_id:
            raise ValueError("folder mismatch")
        if content.remote_id != message_identity.message_id:
            raise ValueError("message id mismatch")
        if content.content_revision != revision.change_key:
            raise ValueError("revision mismatch")
        for participant in (
            content.from_participant,
            content.sender_participant,
            *content.reply_to,
            *content.to_recipients,
            *content.cc_recipients,
            *content.bcc_recipients,
        ):
            if participant is not None:
                validate_msgraph_mail_participant(participant)

    def _build_structured_record(
        self,
        *,
        content: MsGraphMailMessageContent,
        metadata: dict[str, object],
        updated_at: datetime | None,
    ) -> JsonObject:
        if updated_at is None:
            raise ValueError("updated_at is required")
        return {
            "schema": _STRUCTURED_RECORD_SCHEMA,
            "subject": content.subject,
            "conversation_id": content.conversation_id,
            "internet_message_id": content.internet_message_id,
            "body_text": content.body_text,
            "unique_body_text": content.unique_body_text,
            "from": _participant_to_record(content.from_participant),
            "sender": _participant_to_record(content.sender_participant),
            "reply_to": [_participant_to_record(item) for item in content.reply_to],
            "to_recipients": [_participant_to_record(item) for item in content.to_recipients],
            "cc_recipients": [_participant_to_record(item) for item in content.cc_recipients],
            "bcc_recipients": [_participant_to_record(item) for item in content.bcc_recipients],
            "created_at": metadata.get("created_at"),
            "last_modified_at": updated_at.isoformat(),
            "received_at": metadata.get("received_at"),
            "sent_at": metadata.get("sent_at"),
            "is_read": metadata["is_read"],
            "is_draft": metadata["is_draft"],
            "importance": str(metadata["importance"]),
            "attachments": {
                "has_attachments": metadata["has_attachments"],
                "inventory_included": False,
                "binary_content_included": False,
            },
        }

    def _encode_message_identity(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        message_id: str,
    ) -> str:
        identity = _MsGraphMailMessageIdentity(
            schema_version=_MSGRAPH_MAIL_MESSAGE_ID_SCHEMA_VERSION,
            mailbox_user_id=mailbox_user_id,
            folder_id=folder_id,
            message_id=message_id,
        )
        return _encode_canonical_payload(identity.model_dump())

    def _decode_message_identity(self, value: str) -> _MsGraphMailMessageIdentity:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphMailMessageIdentity.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_revision(self, change_key: str) -> str:
        revision = _MsGraphMailRevision(
            schema_version=_MSGRAPH_MAIL_REVISION_SCHEMA_VERSION,
            change_key=change_key,
        )
        return _encode_canonical_payload(revision.model_dump())

    def _decode_revision(self, value: str) -> _MsGraphMailRevision:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphMailRevision.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Mail message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_cursor(self, cursor: _MsGraphMailCursor) -> KnowledgeCursor:
        return KnowledgeCursor(
            value=_encode_canonical_payload(cursor.model_dump()),
            version=MSGRAPH_MAIL_CURSOR_VERSION,
        )

    def _encode_cursor_from_continuation(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation,
    ) -> KnowledgeCursor:
        if continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            continuation_kind: Literal["next_page", "delta"] = "next_page"
        else:
            continuation_kind = "delta"
        return self._encode_cursor(
            _MsGraphMailCursor(
                schema_version=MSGRAPH_MAIL_CURSOR_VERSION,
                mailbox_user_id=mailbox_user_id,
                folder_id=folder_id,
                continuation_kind=continuation_kind,
                continuation_url=continuation.url,
            )
        )

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        mailbox_user_id: str,
        folder_id: str,
    ) -> _MsGraphMailCursor | None:
        if cursor is None:
            return None
        if cursor.version != MSGRAPH_MAIL_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Mail knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            data = _decode_canonical_payload(cursor.value)
            decoded = _MsGraphMailCursor.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Mail knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if (
            decoded.mailbox_user_id != mailbox_user_id
            or decoded.folder_id != folder_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Mail knowledge cursor scope does not match source"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded

    def _to_provider_continuation(
        self,
        decoded: _MsGraphMailCursor | None,
    ) -> MsGraphKnowledgeContinuation | None:
        if decoded is None:
            return None
        if decoded.continuation_kind == "next_page":
            kind = MsGraphKnowledgeContinuationKind.NEXT_PAGE
        else:
            kind = MsGraphKnowledgeContinuationKind.DELTA
        return MsGraphKnowledgeContinuation(kind=kind, url=decoded.continuation_url)

    def _parse_timezone_aware_iso(self, value: object) -> datetime:
        if not isinstance(value, str):
            raise ValueError("timestamp must be an ISO string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("timestamp must not be empty")
        normalized = cleaned.replace("Z", "+00:00") if cleaned.endswith("Z") else cleaned
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            raise ValueError("timestamp must be valid ISO-8601") from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return parsed

    async def _invoke_integration(
        self,
        operation: Callable[[], _T],
        *,
        content_errors: bool = False,
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=(
                    "Microsoft Graph Mail knowledge adapter configuration is invalid"
                    if not content_errors
                    else "Microsoft Graph Mail message exceeds the configured content limit"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except (IntegrationDependencyError, MsGraphMailMessageChanged):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Mail knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except MsGraphMailContentTooLarge:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Mail message exceeds the configured content limit",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Mail knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Mail knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None


def _encode_canonical_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_canonical_payload(value: str) -> dict[str, object]:
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(value + padding)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    return data


def _decode_scope_payload(value: str) -> _MsGraphMailFolderScope:
    data = _decode_canonical_payload(value)
    return _MsGraphMailFolderScope.model_validate(data)


def _validate_opaque_change_key(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("change_key must be a string")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("change_key must not be empty")
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("change_key must not contain control characters")
    if len(trimmed) > _MAX_CHANGE_KEY_LEN:
        raise ValueError("change_key exceeds maximum length")
    return trimmed


def _resolve_message_title(subject: str | None) -> str:
    if subject is None:
        return _SAFE_TITLE_FALLBACK
    if not subject.strip():
        return _SAFE_TITLE_FALLBACK
    return subject


def _participant_to_record(participant: MsGraphMailParticipant | None) -> dict[str, object] | None:
    if participant is None:
        return None
    return {
        "display_name": participant.display_name,
        "address": participant.address,
    }


def register_msgraph_mail_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> MsGraphMailKnowledgeAdapter:
    adapter = MsGraphMailKnowledgeAdapter()
    registry.register(adapter)
    return adapter
