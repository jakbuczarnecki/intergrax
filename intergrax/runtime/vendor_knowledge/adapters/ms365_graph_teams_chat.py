# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Chat knowledge source adapter (MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT)."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mailbox_user_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_content import (
    MsGraphTeamsChatContentTooLarge,
    MsGraphTeamsChatMessageReference,
    validate_msgraph_teams_chat_message_content,
    validate_msgraph_teams_chat_message_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphTeamsChatReference,
    validate_msgraph_teams_chat_id,
    validate_msgraph_teams_chat_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatImportance,
    MsGraphTeamsChatMention,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageChanged,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageState,
    MsGraphTeamsChatMessageType,
    MsGraphTeamsChatMessageWindow,
    MsGraphTeamsChatReaction,
    MsGraphTeamsForwardedMessageReference,
    MsGraphTeamsIdentity,
    validate_msgraph_teams_chat_message,
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

MSGRAPH_TEAMS_CHAT_SCOPE_TYPE = "msgraph_teams_chat"
MSGRAPH_TEAMS_CHAT_CURSOR_VERSION = "msgraph.teams-chat.cursor.v1"
_MSGRAPH_TEAMS_CHAT_SCOPE_SCHEMA_VERSION = "msgraph.teams-chat.scope.v1"
_MSGRAPH_TEAMS_CHAT_MESSAGE_ID_SCHEMA_VERSION = "msgraph.teams-chat.message-id.v1"
_MSGRAPH_TEAMS_CHAT_REVISION_SCHEMA_VERSION = "msgraph.teams-chat.revision.v1"

_STRUCTURED_RECORD_SCHEMA = "msgraph.teams-chat.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-teams-chat-message+json"
_TITLE_FALLBACK = "Teams chat message"

_MAX_CONTINUATION_URL_LEN = 32_768
_MAX_REVISION_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PROVIDER_PAGE_LIMIT = 50
_MAX_EVENT_DETAIL_TYPE_LEN = 1024
_MAX_LOCALE_LEN = 128

_METADATA_REQUIRED_KEYS = frozenset(
    {
        "message_state",
        "message_type",
        "importance",
        "body_kind",
        "has_attachments",
        "created_at",
        "last_modified_at",
        "last_edited_at",
        "event_detail_type",
        "locale",
        "attachment_inventory_in_content",
        "attachment_binary_content_included",
        "hosted_content_included",
        "reference_urls_included",
    }
)

_T = TypeVar("_T")


def _validate_exact_durable_id(value: object, *, validator: Callable[[object], str]) -> str:
    if not isinstance(value, str):
        raise ValueError("durable id must be a string")
    if value == "":
        raise ValueError("durable id must not be empty")
    if value != value.strip():
        raise ValueError("durable id must not have leading or trailing whitespace")
    canonical = validator(value)
    if canonical != value:
        raise ValueError("durable id is invalid")
    return value


class _MsGraphTeamsChatScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-chat.scope.v1"]
    mailbox_user_id: str = Field(repr=False)
    chat_remote_id: str = Field(repr=False)
    window_start_at: datetime = Field(repr=False)
    window_end_at: datetime = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_mailbox_user_id)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_teams_chat_id)

    @field_validator("window_start_at", "window_end_at", mode="before")
    @classmethod
    def _validate_window_datetimes(cls, value: object) -> datetime:
        return _parse_scope_window_datetime(value)

    @model_validator(mode="after")
    def _validate_scope_shape(self) -> _MsGraphTeamsChatScope:
        validate_msgraph_teams_chat_reference(
            MsGraphTeamsChatReference(
                mailbox_user_id=self.mailbox_user_id,
                chat_remote_id=self.chat_remote_id,
            )
        )
        MsGraphTeamsChatMessageWindow(
            start_at=self.window_start_at,
            end_at=self.window_end_at,
        )
        return self


class _MsGraphTeamsChatMessageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-chat.message-id.v1"]
    mailbox_user_id: str = Field(repr=False)
    chat_remote_id: str = Field(repr=False)
    message_remote_id: str = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_mailbox_user_id)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_teams_chat_id)

    @field_validator("message_remote_id", mode="before")
    @classmethod
    def _validate_message_remote_id(cls, value: object) -> str:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
            validate_msgraph_teams_chat_message_id,
        )

        return _validate_exact_durable_id(
            value,
            validator=validate_msgraph_teams_chat_message_id,
        )


class _MsGraphTeamsChatMessageRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-chat.revision.v1"]
    revision: str = Field(repr=False)

    @field_validator("revision", mode="before")
    @classmethod
    def _validate_revision_field(cls, value: object) -> str:
        return _validate_opaque_revision(value)


class _MsGraphTeamsChatCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-chat.cursor.v1"]
    mailbox_user_id: str = Field(repr=False)
    chat_remote_id: str = Field(repr=False)
    window_start_at: datetime = Field(repr=False)
    window_end_at: datetime = Field(repr=False)
    phase: Literal["messages", "complete"]
    continuation_url: str | None = Field(default=None, repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_mailbox_user_id)

    @field_validator("chat_remote_id", mode="before")
    @classmethod
    def _validate_chat_remote_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_teams_chat_id)

    @field_validator("window_start_at", "window_end_at", mode="before")
    @classmethod
    def _validate_window_datetimes(cls, value: object) -> datetime:
        return _parse_scope_window_datetime(value)

    @field_validator("continuation_url", mode="before")
    @classmethod
    def _validate_continuation_url_field(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_continuation_url(value)

    @model_validator(mode="after")
    def _validate_phase_shape(self) -> _MsGraphTeamsChatCursor:
        if self.phase == "messages":
            if self.continuation_url is None:
                raise ValueError("messages phase requires continuation_url")
        elif self.continuation_url is not None:
            raise ValueError("complete phase forbids continuation_url")
        MsGraphTeamsChatMessageWindow(
            start_at=self.window_start_at,
            end_at=self.window_end_at,
        )
        return self


def encode_msgraph_teams_chat_scope_id(
    *,
    mailbox_user_id: str,
    chat_remote_id: str,
    window: MsGraphTeamsChatMessageWindow,
) -> str:
    """Return the canonical opaque remote_scope_id for one Teams chat source."""
    scope = _MsGraphTeamsChatScope(
        schema_version=_MSGRAPH_TEAMS_CHAT_SCOPE_SCHEMA_VERSION,
        mailbox_user_id=mailbox_user_id,
        chat_remote_id=chat_remote_id,
        window_start_at=window.start_at,
        window_end_at=window.end_at,
    )
    return _encode_canonical_payload(scope.model_dump(mode="json"))


class MsGraphTeamsChatKnowledgeAdapter:
    """Thin mapping from Microsoft Graph Teams Chat integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return MSGRAPH_TEAMS_CHAT_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
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
        validated_source = self._validate_source_ref(source)
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
        graph_integration = self._require_graph_integration(integration=integration, source=source)
        validated_source = self._validate_source_ref(source)
        mailbox_user_id, chat_remote_id, window = self._decode_scope(validated_source)
        self._validate_limit(limit)
        decoded_cursor = self._decode_cursor(
            cursor,
            mailbox_user_id=mailbox_user_id,
            chat_remote_id=chat_remote_id,
            window_start_at=window.start_at,
            window_end_at=window.end_at,
        )
        if decoded_cursor is not None and decoded_cursor.phase == "complete":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Teams Chat reconciliation cursor is complete; "
                    "restart reconciliation"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        provider_limit = min(limit, _PROVIDER_PAGE_LIMIT)
        provider_continuation: MsGraphKnowledgeContinuation | None = None
        if decoded_cursor is not None and decoded_cursor.phase == "messages":
            provider_continuation = MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=decoded_cursor.continuation_url,
            )
        chat_reference = validate_msgraph_teams_chat_reference(
            MsGraphTeamsChatReference(
                mailbox_user_id=mailbox_user_id,
                chat_remote_id=chat_remote_id,
            )
        )
        page = await self._invoke_integration(
            lambda: graph_integration.read_teams_chat_messages_snapshot_page_by_reference(
                chat=chat_reference,
                window=window,
                continuation=provider_continuation,
                limit=provider_limit,
                max_chars_per_message=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
            )
        )
        try:
            validated_page = self._validate_snapshot_page_for_source(
                page,
                mailbox_user_id=mailbox_user_id,
                chat_remote_id=chat_remote_id,
                window=window,
                provider_limit=provider_limit,
            )
            changes = tuple(
                self._message_to_change(
                    item,
                    mailbox_user_id=mailbox_user_id,
                    chat_remote_id=chat_remote_id,
                )
                for item in validated_page.items
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Chat knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if validated_page.continuation is not None:
            next_cursor = self._encode_cursor(
                _MsGraphTeamsChatCursor(
                    schema_version=MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
                    mailbox_user_id=mailbox_user_id,
                    chat_remote_id=chat_remote_id,
                    window_start_at=window.start_at,
                    window_end_at=window.end_at,
                    phase="messages",
                    continuation_url=validated_page.continuation.url,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        complete = self._encode_complete_cursor(
            mailbox_user_id=mailbox_user_id,
            chat_remote_id=chat_remote_id,
            window_start_at=window.start_at,
            window_end_at=window.end_at,
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=complete,
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
        validated_source = self._validate_source_ref(source)
        mailbox_user_id, chat_remote_id, _window = self._decode_scope(validated_source)
        try:
            validated_item = self._deep_validate_item_descriptor(item)
            message_identity, revision = self._validate_message_item(
                validated_item,
                source=validated_source,
                mailbox_user_id=mailbox_user_id,
                chat_remote_id=chat_remote_id,
            )
            metadata = self._validate_descriptor_metadata(
                validated_item.metadata,
                updated_at=validated_item.revision.updated_at,
            )
            provider_reference = self._descriptor_to_provider_reference(
                message_identity=message_identity,
                revision=revision,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        result = await self._invoke_integration(
            lambda: graph_integration.read_teams_chat_message_content(
                message=provider_reference,
                max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
            ),
        )
        try:
            validated_content = validate_msgraph_teams_chat_message_content(
                result,
                reference=provider_reference,
                max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
            )
            self._validate_fetched_content_identity(
                validated_content,
                message_identity=message_identity,
                revision=revision,
                metadata=metadata,
                updated_at=validated_item.revision.updated_at,
            )
        except MsGraphTeamsChatMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Chat knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Chat knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        try:
            structured_record = self._build_structured_record(validated_content)
            canonical = json.dumps(
                structured_record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            content_hash = hashlib.sha256(canonical).hexdigest()
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Chat knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
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
        validated_source = self._validate_source_ref(source)
        validated_item = self._deep_validate_item_descriptor(item)
        self._validate_item_provenance(validated_item, source=validated_source)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=(
                "Microsoft Graph Teams Chat authoritative permission projection "
                "is not implemented"
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
                    "Microsoft Graph Teams Chat knowledge adapter requires "
                    "Microsoft Graph collaboration-suite integration"
                ),
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return integration

    def _validate_source_ref(self, source: KnowledgeSourceRef) -> KnowledgeSourceRef:
        if not isinstance(source, KnowledgeSourceRef):
            raise self._invalid_source_scope_error()
        try:
            validated_source = KnowledgeSourceRef.model_validate(
                source.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_source_scope_error() from None
        if (
            validated_source.provider_id != self.provider_id
            or validated_source.integration_kind != self.integration_kind
            or validated_source.source_kind != self.source_kind
        ):
            raise self._invalid_source_scope_error(
                provider_id=validated_source.provider_id,
                source_kind=validated_source.source_kind,
            )
        scope = validated_source.scope
        if scope.remote_scope_type != MSGRAPH_TEAMS_CHAT_SCOPE_TYPE:
            raise self._invalid_source_scope_error(
                provider_id=validated_source.provider_id,
                source_kind=validated_source.source_kind,
            )
        if scope.parameters:
            raise self._invalid_source_scope_error(
                provider_id=validated_source.provider_id,
                source_kind=validated_source.source_kind,
            )
        try:
            _decode_scope_payload(scope.remote_scope_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_source_scope_error(
                provider_id=validated_source.provider_id,
                source_kind=validated_source.source_kind,
            ) from None
        return validated_source

    def _decode_scope(
        self,
        source: KnowledgeSourceRef,
    ) -> tuple[str, str, MsGraphTeamsChatMessageWindow]:
        try:
            decoded = _decode_scope_payload(source.scope.remote_scope_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_source_scope_error(
                provider_id=source.provider_id,
                source_kind=source.source_kind,
            ) from None
        window = MsGraphTeamsChatMessageWindow(
            start_at=decoded.window_start_at,
            end_at=decoded.window_end_at,
        )
        return decoded.mailbox_user_id, decoded.chat_remote_id, window

    def _invalid_source_scope_error(
        self,
        *,
        provider_id: str | None = None,
        source_kind: str | None = None,
    ) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message="Microsoft Graph Teams Chat knowledge source scope is invalid",
            provider_id=provider_id or self.provider_id,
            source_kind=source_kind or self.source_kind,
            retryable=False,
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Chat knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Chat knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return limit

    def _deep_validate_item_descriptor(self, item: object) -> KnowledgeItemDescriptor:
        if not isinstance(item, KnowledgeItemDescriptor):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if type(item.content_mode) is not KnowledgeContentMode:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            return KnowledgeItemDescriptor.model_validate(item.model_dump(mode="python"))
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

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
        if provenance.remote_id != item.identity.remote_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_snapshot_page_for_source(
        self,
        page: object,
        *,
        mailbox_user_id: str,
        chat_remote_id: str,
        window: MsGraphTeamsChatMessageWindow,
        provider_limit: int,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
        if type(page) is not MsGraphTeamsChatMessageSnapshotPage:
            raise ValueError("page must be a MsGraphTeamsChatMessageSnapshotPage")
        if page.mailbox_user_id != mailbox_user_id or page.chat_remote_id != chat_remote_id:
            raise ValueError("page scope does not match source")
        if (
            page.window.start_at != window.start_at
            or page.window.end_at != window.end_at
        ):
            raise ValueError("page window does not match source")
        if type(page.items) is not tuple:
            raise ValueError("page items must be a tuple")
        if len(page.items) > provider_limit:
            raise ValueError("page items exceed provider limit")
        seen_remote_ids: set[str] = set()
        validated_items: list[MsGraphTeamsChatMessage] = []
        for item in page.items:
            validated = validate_msgraph_teams_chat_message(
                item,
                max_chars=DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS,
            )
            if (
                validated.mailbox_user_id != mailbox_user_id
                or validated.chat_remote_id != chat_remote_id
            ):
                raise ValueError("item scope does not match source")
            opaque_id = self._encode_message_identity(
                mailbox_user_id=mailbox_user_id,
                chat_remote_id=chat_remote_id,
                message_remote_id=validated.remote_id,
            )
            if opaque_id in seen_remote_ids:
                raise ValueError("duplicate neutral id on page")
            seen_remote_ids.add(opaque_id)
            validated_items.append(validated)
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if page.continuation is not None:
            validated_continuation = self._deep_validate_continuation(page.continuation)
        return MsGraphTeamsChatMessageSnapshotPage(
            mailbox_user_id=mailbox_user_id,
            chat_remote_id=chat_remote_id,
            window=window,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )

    def _deep_validate_continuation(
        self,
        continuation: MsGraphKnowledgeContinuation,
    ) -> MsGraphKnowledgeContinuation:
        validated = MsGraphKnowledgeContinuation.model_validate(
            continuation.model_dump(mode="python")
        )
        if validated.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError("invalid continuation kind")
        _validate_continuation_url(validated.url)
        return validated

    def _validate_descriptor_owned_fields(self, item: KnowledgeItemDescriptor) -> None:
        if item.revision.etag is not None:
            raise ValueError("revision etag must be None")
        if item.revision.content_hash is not None:
            raise ValueError("revision content_hash must be None")
        if item.revision.acl_hash is not None:
            raise ValueError("revision acl_hash must be None")
        if item.provenance.web_url is not None:
            raise ValueError("provenance web_url must be None")
        if item.provenance.safe_locator is not None:
            raise ValueError("provenance safe_locator must be None")

    def _message_to_change(
        self,
        message: MsGraphTeamsChatMessage,
        *,
        mailbox_user_id: str,
        chat_remote_id: str,
    ) -> KnowledgeChange:
        opaque_id = self._encode_message_identity(
            mailbox_user_id=mailbox_user_id,
            chat_remote_id=chat_remote_id,
            message_remote_id=message.remote_id,
        )
        if message.state is MsGraphTeamsChatMessageState.DELETED:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=opaque_id,
                descriptor=None,
            )
        if message.state is not MsGraphTeamsChatMessageState.ACTIVE:
            raise ValueError("unknown message state")
        if message.body_kind is None or message.body_content is None:
            raise ValueError("active message body is missing")
        descriptor = self._active_message_to_descriptor(
            message,
            opaque_remote_id=opaque_id,
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=opaque_id,
            descriptor=descriptor,
        )

    def _active_message_to_descriptor(
        self,
        message: MsGraphTeamsChatMessage,
        *,
        opaque_remote_id: str,
    ) -> KnowledgeItemDescriptor:
        title = _resolve_message_title(subject=message.subject)
        if message.state is not MsGraphTeamsChatMessageState.ACTIVE:
            raise ValueError("message is not active")
        if message.body_kind is None or message.body_content is None:
            raise ValueError("active message body is missing")
        metadata: dict[str, object] = {
            "message_state": "active",
            "message_type": message.message_type.value,
            "importance": message.importance.value,
            "body_kind": message.body_kind.value,
            "has_attachments": bool(message.attachments),
            "created_at": message.created_at.isoformat(),
            "last_modified_at": message.last_modified_at.isoformat(),
            "last_edited_at": (
                message.last_edited_at.isoformat() if message.last_edited_at is not None else None
            ),
            "event_detail_type": message.event_detail_type,
            "locale": message.locale,
            "attachment_inventory_in_content": True,
            "attachment_binary_content_included": False,
            "hosted_content_included": False,
            "reference_urls_included": False,
        }
        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=opaque_remote_id,
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=self._encode_revision(message.revision),
                etag=None,
                content_hash=None,
                acl_hash=None,
                updated_at=message.last_modified_at,
            ),
            title=title,
            item_type="msgraph_teams_chat_message",
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
        chat_remote_id: str,
    ) -> tuple[_MsGraphTeamsChatMessageIdentity, _MsGraphTeamsChatMessageRevision]:
        self._validate_item_provenance(item, source=source)
        self._validate_descriptor_owned_fields(item)
        if item.item_type != "msgraph_teams_chat_message":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.identity.parent_remote_id is not None or item.identity.logical_key is not None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        message_identity = self._decode_message_identity(item.identity.remote_id)
        if (
            message_identity.mailbox_user_id != mailbox_user_id
            or message_identity.chat_remote_id != chat_remote_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        version = item.revision.version
        if not isinstance(version, str) or not version.strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.revision.updated_at is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
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
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        revision = self._decode_revision(version)
        return message_identity, revision

    def _validate_descriptor_metadata(
        self,
        metadata: object,
        *,
        updated_at: datetime | None,
    ) -> dict[str, object]:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        keys = set(metadata.keys())
        if keys != _METADATA_REQUIRED_KEYS:
            raise ValueError("metadata keys are invalid")
        if metadata["message_state"] != "active":
            raise ValueError("message_state must be active")
        if type(metadata["has_attachments"]) is not bool:
            raise ValueError("has_attachments must be bool")
        for bool_key in (
            "attachment_inventory_in_content",
            "attachment_binary_content_included",
            "hosted_content_included",
            "reference_urls_included",
        ):
            if type(metadata[bool_key]) is not bool:
                raise ValueError(f"{bool_key} must be bool")
        if metadata["attachment_inventory_in_content"] is not True:
            raise ValueError("attachment_inventory_in_content must be True")
        for false_key in (
            "attachment_binary_content_included",
            "hosted_content_included",
            "reference_urls_included",
        ):
            if metadata[false_key] is not False:
                raise ValueError(f"{false_key} must be False")
        if not isinstance(metadata["message_type"], str):
            raise ValueError("message_type must be a string")
        try:
            MsGraphTeamsChatMessageType(metadata["message_type"])
        except ValueError:
            raise ValueError("message_type is invalid") from None
        if not isinstance(metadata["importance"], str):
            raise ValueError("importance must be a string")
        try:
            MsGraphTeamsChatImportance(metadata["importance"])
        except ValueError:
            raise ValueError("importance is invalid") from None
        if metadata["body_kind"] not in {"text", "html"}:
            raise ValueError("body_kind is invalid")
        created_at = self._parse_timezone_aware_iso(metadata["created_at"])
        last_modified_at = self._parse_timezone_aware_iso(metadata["last_modified_at"])
        if updated_at is None or last_modified_at != updated_at:
            raise ValueError("last_modified_at mismatch")
        last_edited_raw = metadata["last_edited_at"]
        last_edited_at: datetime | None = None
        if last_edited_raw is not None:
            last_edited_at = self._parse_timezone_aware_iso(last_edited_raw)
        _validate_optional_metadata_string(
            metadata["event_detail_type"],
            max_length=_MAX_EVENT_DETAIL_TYPE_LEN,
        )
        _validate_optional_metadata_string(
            metadata["locale"],
            max_length=_MAX_LOCALE_LEN,
        )
        if created_at > last_modified_at:
            raise ValueError("created_at after last_modified_at")
        if last_edited_at is not None and last_edited_at < created_at:
            raise ValueError("last_edited_at before created_at")
        return metadata

    def _descriptor_to_provider_reference(
        self,
        *,
        message_identity: _MsGraphTeamsChatMessageIdentity,
        revision: _MsGraphTeamsChatMessageRevision,
    ) -> MsGraphTeamsChatMessageReference:
        return validate_msgraph_teams_chat_message_reference(
            MsGraphTeamsChatMessageReference(
                mailbox_user_id=message_identity.mailbox_user_id,
                chat_remote_id=message_identity.chat_remote_id,
                remote_id=message_identity.message_remote_id,
                revision=revision.revision,
            )
        )

    def _validate_fetched_content_identity(
        self,
        content: MsGraphTeamsChatMessage,
        *,
        message_identity: _MsGraphTeamsChatMessageIdentity,
        revision: _MsGraphTeamsChatMessageRevision,
        metadata: dict[str, object],
        updated_at: datetime | None,
    ) -> None:
        if content.mailbox_user_id != message_identity.mailbox_user_id:
            raise ValueError("mailbox mismatch")
        if content.chat_remote_id != message_identity.chat_remote_id:
            raise ValueError("chat mismatch")
        if content.remote_id != message_identity.message_remote_id:
            raise ValueError("message id mismatch")
        if content.revision != revision.revision:
            raise ValueError("revision mismatch")
        if content.state is not MsGraphTeamsChatMessageState.ACTIVE:
            raise ValueError("message is not active")
        if content.body_kind is None or content.body_content is None:
            raise ValueError("message body missing")
        if updated_at is None or content.last_modified_at != updated_at:
            raise ValueError("updated_at mismatch")
        if content.message_type.value != metadata["message_type"]:
            raise ValueError("message_type mismatch")
        if content.importance.value != metadata["importance"]:
            raise ValueError("importance mismatch")
        if content.body_kind.value != metadata["body_kind"]:
            raise ValueError("body_kind mismatch")
        if bool(content.attachments) != metadata["has_attachments"]:
            raise ValueError("has_attachments mismatch")
        if content.created_at.isoformat() != metadata["created_at"]:
            raise ValueError("created_at mismatch")
        if content.last_modified_at.isoformat() != metadata["last_modified_at"]:
            raise ValueError("last_modified_at mismatch")
        expected_last_edited = metadata["last_edited_at"]
        actual_last_edited = (
            content.last_edited_at.isoformat() if content.last_edited_at is not None else None
        )
        if actual_last_edited != expected_last_edited:
            raise ValueError("last_edited_at mismatch")
        if content.event_detail_type != metadata["event_detail_type"]:
            raise ValueError("event_detail_type mismatch")
        if content.locale != metadata["locale"]:
            raise ValueError("locale mismatch")

    def _build_structured_record(self, message: MsGraphTeamsChatMessage) -> JsonObject:
        if message.state is not MsGraphTeamsChatMessageState.ACTIVE:
            raise ValueError("message is not active")
        if message.body_kind is None:
            raise ValueError("body_kind is missing")
        if message.body_content is None:
            raise ValueError("body_content is missing")
        return {
            "schema": _STRUCTURED_RECORD_SCHEMA,
            "state": "active",
            "subject": message.subject,
            "body": {
                "kind": message.body_kind.value,
                "content": message.body_content,
            },
            "sender": _identity_to_record(message.sender),
            "created_at": message.created_at.isoformat(),
            "last_modified_at": message.last_modified_at.isoformat(),
            "last_edited_at": (
                message.last_edited_at.isoformat() if message.last_edited_at is not None else None
            ),
            "message_type": message.message_type.value,
            "importance": message.importance.value,
            "locale": message.locale,
            "event_detail_type": message.event_detail_type,
            "mentions": [_mention_to_record(item) for item in message.mentions],
            "reactions": [_reaction_to_record(item) for item in message.reactions],
            "attachments": {
                "inventory_included": True,
                "binary_content_included": False,
                "hosted_content_included": False,
                "reference_urls_included": False,
                "items": [
                    _attachment_to_record(item) for item in message.attachments
                ],
            },
        }

    def _encode_message_identity(
        self,
        *,
        mailbox_user_id: str,
        chat_remote_id: str,
        message_remote_id: str,
    ) -> str:
        identity = _MsGraphTeamsChatMessageIdentity(
            schema_version=_MSGRAPH_TEAMS_CHAT_MESSAGE_ID_SCHEMA_VERSION,
            mailbox_user_id=mailbox_user_id,
            chat_remote_id=chat_remote_id,
            message_remote_id=message_remote_id,
        )
        return _encode_canonical_payload(identity.model_dump())

    def _decode_message_identity(self, value: str) -> _MsGraphTeamsChatMessageIdentity:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphTeamsChatMessageIdentity.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_revision(self, revision: str) -> str:
        encoded = _MsGraphTeamsChatMessageRevision(
            schema_version=_MSGRAPH_TEAMS_CHAT_REVISION_SCHEMA_VERSION,
            revision=revision,
        )
        return _encode_canonical_payload(encoded.model_dump())

    def _decode_revision(self, value: str) -> _MsGraphTeamsChatMessageRevision:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphTeamsChatMessageRevision.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Chat message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_cursor(self, cursor: _MsGraphTeamsChatCursor) -> KnowledgeCursor:
        return KnowledgeCursor(
            value=_encode_canonical_payload(cursor.model_dump(mode="json")),
            version=MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
        )

    def _encode_complete_cursor(
        self,
        *,
        mailbox_user_id: str,
        chat_remote_id: str,
        window_start_at: datetime,
        window_end_at: datetime,
    ) -> KnowledgeCursor:
        return self._encode_cursor(
            _MsGraphTeamsChatCursor(
                schema_version=MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
                mailbox_user_id=mailbox_user_id,
                chat_remote_id=chat_remote_id,
                window_start_at=window_start_at,
                window_end_at=window_end_at,
                phase="complete",
            )
        )

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        mailbox_user_id: str,
        chat_remote_id: str,
        window_start_at: datetime,
        window_end_at: datetime,
    ) -> _MsGraphTeamsChatCursor | None:
        if cursor is None:
            return None
        if cursor.version != MSGRAPH_TEAMS_CHAT_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Teams Chat knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            data = _decode_canonical_payload(cursor.value)
            decoded = _MsGraphTeamsChatCursor.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Teams Chat knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if (
            decoded.mailbox_user_id != mailbox_user_id
            or decoded.chat_remote_id != chat_remote_id
            or decoded.window_start_at != window_start_at
            or decoded.window_end_at != window_end_at
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Teams Chat knowledge cursor scope does not match source"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded

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
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except MsGraphTeamsChatContentTooLarge:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Chat message exceeds the configured content limit",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Chat knowledge adapter configuration is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except MsGraphTeamsChatMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Chat knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Chat knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Chat knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Chat knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None


def _parse_scope_window_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("window timestamp must be timezone-aware")
        return value
    if not isinstance(value, str):
        raise ValueError("window timestamp must be a datetime")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("window timestamp must not be empty")
    normalized = cleaned.replace("Z", "+00:00") if cleaned.endswith("Z") else cleaned
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError("window timestamp must be valid ISO-8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("window timestamp must be timezone-aware")
    return parsed


def _validate_optional_metadata_string(value: object, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("metadata string must be a string or None")
    if not value:
        raise ValueError("metadata string must not be empty")
    if value != value.strip():
        raise ValueError("metadata string must not have leading or trailing whitespace")
    if _ASCII_CONTROL.search(value):
        raise ValueError("metadata string must not contain control characters")
    if len(value) > max_length:
        raise ValueError("metadata string exceeds maximum length")
    return value


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


def _decode_scope_payload(value: str) -> _MsGraphTeamsChatScope:
    data = _decode_canonical_payload(value)
    return _MsGraphTeamsChatScope.model_validate(data)


def _validate_opaque_revision(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("revision must be a string")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("revision must not be empty")
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("revision must not contain control characters")
    if len(trimmed) > _MAX_REVISION_LEN:
        raise ValueError("revision exceeds maximum length")
    return trimmed


def _validate_continuation_url(value: object) -> str:
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


def _resolve_message_title(*, subject: str | None) -> str:
    if subject is not None and subject.strip():
        return subject
    return _TITLE_FALLBACK


def _identity_to_record(identity: MsGraphTeamsIdentity | None) -> dict[str, object] | None:
    if identity is None:
        return None
    return {
        "kind": identity.identity_kind.value,
        "remote_id": identity.remote_id,
        "display_name": identity.display_name,
        "tenant_id": identity.tenant_id,
        "identity_type": identity.identity_type,
    }


def _mention_to_record(mention: MsGraphTeamsChatMention) -> dict[str, object]:
    return {
        "id": mention.mention_id,
        "text": mention.mention_text,
        "mentioned": _identity_to_record(mention.mentioned),
    }


def _reaction_to_record(reaction: MsGraphTeamsChatReaction) -> dict[str, object]:
    return {
        "type": reaction.reaction_type,
        "display_name": reaction.display_name,
        "created_at": reaction.created_at.isoformat(),
        "user": _identity_to_record(reaction.user),
    }


def _forwarded_message_to_record(
    forwarded: MsGraphTeamsForwardedMessageReference | None,
) -> dict[str, object] | None:
    if forwarded is None:
        return None
    return {
        "original_message_id": forwarded.original_message_id,
        "original_chat_id": forwarded.original_chat_id,
        "original_sent_at": forwarded.original_sent_at.isoformat(),
        "original_sender": _identity_to_record(forwarded.original_sender),
    }


def _attachment_to_record(attachment: MsGraphTeamsChatAttachmentReference) -> dict[str, object]:
    return {
        "remote_id": attachment.remote_id,
        "kind": attachment.attachment_kind.value,
        "content_type": attachment.content_type,
        "name": attachment.name,
        "teams_app_id": attachment.teams_app_id,
        "has_thumbnail": attachment.has_thumbnail_url,
        "forwarded_message": _forwarded_message_to_record(attachment.forwarded_message),
    }


def register_msgraph_teams_chat_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> MsGraphTeamsChatKnowledgeAdapter:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    registry.register(adapter)
    return adapter
