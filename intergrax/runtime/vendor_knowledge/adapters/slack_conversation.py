# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack conversation knowledge source adapter (SLACK-KNOWLEDGE-FOUNDATION-1)."""

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
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    DEFAULT_MESSAGE_MAX_CHARS,
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationContentTooLarge,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessageChanged,
    SlackConversationMessageNotFound,
    SlackConversationMessagePage,
    SlackConversationReadConfigurationError,
    SlackConversationReadError,
    SlackConversationSourceWindow,
    compute_slack_conversation_message_revision,
    validate_slack_conversation_message,
    validate_slack_timestamp,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.timestamp import (
    slack_timestamp_in_window,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
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

SLACK_CONVERSATION_SCOPE_TYPE = "slack_conversation"
SLACK_CONVERSATION_CURSOR_VERSION = "slack.conversation.cursor.v1"
_SLACK_CONVERSATION_SCOPE_SCHEMA_VERSION = "slack.conversation.scope.v2"
_SLACK_CONVERSATION_MESSAGE_ID_SCHEMA_VERSION = "slack.conversation.message-id.v1"
_SLACK_CONVERSATION_REVISION_SCHEMA_VERSION = "slack.conversation.revision.v1"

_STRUCTURED_RECORD_SCHEMA = "slack.conversation.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.slack-conversation-message+json"
_TITLE_FALLBACK = "Slack message"

_MAX_REVISION_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_HISTORY_PROVIDER_PAGE_LIMIT = 1
_REPLY_PROVIDER_PAGE_LIMIT = 15

_METADATA_REQUIRED_KEYS = frozenset(
    {
        "subtype",
        "has_files",
        "reply_count",
        "created_at",
        "edited_at",
        "thread_root_ts",
        "attachment_inventory_in_content",
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


class _SlackConversationScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["slack.conversation.scope.v2"]
    conversation_id: str = Field(repr=False)
    conversation_kind: SlackConversationKind
    root_oldest: str = Field(repr=False)
    root_latest: str = Field(repr=False)

    @field_validator("conversation_kind", mode="before")
    @classmethod
    def _validate_conversation_kind(cls, value: object) -> SlackConversationKind:
        if isinstance(value, SlackConversationKind):
            return value
        if isinstance(value, str):
            return SlackConversationKind(value)
        raise ValueError("invalid conversation kind")

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str:
        from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
            validate_slack_conversation_id,
        )

        return _validate_exact_durable_id(value, validator=validate_slack_conversation_id)

    @field_validator("root_oldest", "root_latest", mode="before")
    @classmethod
    def _validate_boundaries(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_slack_timestamp)

    @model_validator(mode="after")
    def _validate_scope_shape(self) -> _SlackConversationScope:
        SlackConversationSourceWindow(oldest=self.root_oldest, latest=self.root_latest)
        return self


class _SlackConversationMessageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["slack.conversation.message-id.v1"]
    conversation_id: str = Field(repr=False)
    message_ts: str = Field(repr=False)

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str:
        from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
            validate_slack_conversation_id,
        )

        return _validate_exact_durable_id(value, validator=validate_slack_conversation_id)

    @field_validator("message_ts", mode="before")
    @classmethod
    def _validate_message_ts(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_slack_timestamp)


class _SlackConversationMessageRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["slack.conversation.revision.v1"]
    revision: str = Field(repr=False)

    @field_validator("revision", mode="before")
    @classmethod
    def _validate_revision_field(cls, value: object) -> str:
        return _validate_opaque_revision(value)


class _SlackConversationCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["slack.conversation.cursor.v1"]
    conversation_id: str = Field(repr=False)
    conversation_kind: SlackConversationKind
    root_oldest: str = Field(repr=False)
    root_latest: str = Field(repr=False)
    phase: Literal["history", "replies", "complete"]
    resume_history_cursor: str | None = Field(default=None, repr=False)
    history_cursor: str | None = Field(default=None, repr=False)
    root_message_ts: str | None = Field(default=None, repr=False)
    root_message_revision: str | None = Field(default=None, repr=False)
    reply_cursor: str | None = Field(default=None, repr=False)

    @field_validator("conversation_kind", mode="before")
    @classmethod
    def _validate_cursor_conversation_kind(cls, value: object) -> SlackConversationKind:
        if isinstance(value, SlackConversationKind):
            return value
        if isinstance(value, str):
            return SlackConversationKind(value)
        raise ValueError("invalid conversation kind")

    @field_validator("conversation_id", mode="before")
    @classmethod
    def _validate_conversation_id(cls, value: object) -> str:
        from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
            validate_slack_conversation_id,
        )

        return _validate_exact_durable_id(value, validator=validate_slack_conversation_id)

    @field_validator("root_oldest", "root_latest", "root_message_ts", mode="before")
    @classmethod
    def _validate_timestamps(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_exact_durable_id(value, validator=validate_slack_timestamp)

    @field_validator("resume_history_cursor", "history_cursor", "reply_cursor", mode="before")
    @classmethod
    def _validate_provider_cursors(cls, value: object) -> str | None:
        if value is None:
            return None
        from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
            validate_provider_cursor,
        )

        return validate_provider_cursor(value)

    @field_validator("root_message_revision", mode="before")
    @classmethod
    def _validate_root_revision(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_opaque_revision(value)

    @model_validator(mode="after")
    def _validate_phase_shape(self) -> _SlackConversationCursor:
        SlackConversationSourceWindow(oldest=self.root_oldest, latest=self.root_latest)
        if self.phase == "history":
            if self.root_message_ts is not None or self.root_message_revision is not None:
                raise ValueError("history phase has forbidden root fields")
            if self.reply_cursor is not None:
                raise ValueError("history phase has forbidden reply cursor")
        elif self.phase == "replies":
            if self.root_message_ts is None or self.root_message_revision is None:
                raise ValueError("replies phase missing root fields")
        elif self.phase == "complete":
            if (
                self.resume_history_cursor is not None
                or self.history_cursor is not None
                or self.root_message_ts is not None
                or self.root_message_revision is not None
                or self.reply_cursor is not None
            ):
                raise ValueError("complete phase has forbidden fields")
        return self


def encode_slack_conversation_scope_id(
    *,
    conversation_id: str,
    conversation_kind: SlackConversationKind,
    oldest: str,
    latest: str,
) -> str:
    scope = _SlackConversationScope(
        schema_version=_SLACK_CONVERSATION_SCOPE_SCHEMA_VERSION,
        conversation_id=conversation_id,
        conversation_kind=conversation_kind,
        root_oldest=oldest,
        root_latest=latest,
    )
    return _encode_canonical_payload(scope.model_dump(mode="json"))


class SlackConversationKnowledgeAdapter:
    """Thin mapping from Slack conversation integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return SLACK_CONVERSATION_CHANNEL_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.CONVERSATION_CHANNEL

    @property
    def source_kind(self) -> str:
        return SLACK_CONVERSATION_SOURCE_KIND

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
            tombstones=False,
            remote_versions=True,
            reconciliation=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self._require_slack_integration(integration=integration, source=source)
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
        slack_integration = self._require_slack_integration(integration=integration, source=source)
        validated_source = self._validate_source_ref(source)
        conversation_id, conversation_kind, window = self._decode_scope(validated_source)
        self._validate_limit(limit)
        decoded_cursor = self._decode_cursor(
            cursor,
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,
            oldest=window.oldest,
            latest=window.latest,
        )
        if decoded_cursor is not None and decoded_cursor.phase == "complete":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Slack conversation reconciliation cursor is complete; restart reconciliation"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if decoded_cursor is None or decoded_cursor.phase == "history":
            return await self._read_history_page(
                slack_integration=slack_integration,
                conversation_id=conversation_id,
                conversation_kind=conversation_kind,
                window=window,
                decoded_cursor=decoded_cursor,
            )
        return await self._read_replies_page(
            slack_integration=slack_integration,
            conversation_id=conversation_id,
            window=window,
            decoded_cursor=decoded_cursor,
            limit=limit,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        slack_integration = self._require_slack_integration(integration=integration, source=source)
        validated_source = self._validate_source_ref(source)
        conversation_id, conversation_kind, window = self._decode_scope(validated_source)
        try:
            message_identity, revision, metadata = self._validate_descriptor_for_fetch(
                item,
                source=validated_source,
                conversation_id=conversation_id,
                window=window,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Slack conversation message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        root_thread_ts = metadata.get("thread_root_ts")
        result = await self._invoke_integration(
            lambda: slack_integration.read_exact_message(
                conversation_id=message_identity.conversation_id,
                conversation_kind=conversation_kind,
                message_ts=message_identity.message_ts,
                root_thread_ts=root_thread_ts if isinstance(root_thread_ts, str) else None,
                window=window,
                expected_revision=revision,
                max_chars_per_message=DEFAULT_MESSAGE_MAX_CHARS,
            ),
        )
        if not result.found or result.message is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND,
                safe_message="Slack conversation message was not found",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            validated_content = validate_slack_conversation_message(result.message)
            self._validate_fetched_content_identity(
                validated_content,
                message_identity=message_identity,
                revision=revision,
                metadata=metadata,
                updated_at=item.revision.updated_at,
            )
        except SlackConversationMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Slack conversation message changed during content read",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Slack conversation knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        structured_record = self._build_structured_record(validated_content)
        canonical = json.dumps(
            structured_record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
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
        self._require_slack_integration(integration=integration, source=source)
        validated_source = self._validate_source_ref(source)
        conversation_id, _, window = self._decode_scope(validated_source)
        try:
            self._validate_descriptor_for_fetch(
                item,
                source=validated_source,
                conversation_id=conversation_id,
                window=window,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Slack conversation message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message="Slack conversation authoritative permission projection is not implemented",
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    async def _read_history_page(
        self,
        *,
        slack_integration: SlackConversationChannelIntegration,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        window: SlackConversationSourceWindow,
        decoded_cursor: _SlackConversationCursor | None,
    ) -> KnowledgePage:
        resolved_kind = (
            decoded_cursor.conversation_kind if decoded_cursor is not None else conversation_kind
        )
        history_cursor = decoded_cursor.history_cursor if decoded_cursor is not None else None
        page = await self._invoke_integration(
            lambda: slack_integration.read_conversation_history_page(
                conversation_id=conversation_id,
                conversation_kind=conversation_kind,
                window=window,
                cursor=history_cursor,
                limit=_HISTORY_PROVIDER_PAGE_LIMIT,
                max_chars_per_message=DEFAULT_MESSAGE_MAX_CHARS,
            )
        )
        validated_page = self._validate_message_page_for_source(
            page,
            conversation_id=conversation_id,
            window=window,
        )
        if not validated_page.items:
            if validated_page.next_cursor is not None:
                advanced = self._encode_cursor(
                    _SlackConversationCursor(
                        schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                        conversation_id=conversation_id,
                        conversation_kind=resolved_kind,
                        root_oldest=window.oldest,
                        root_latest=window.latest,
                        phase="history",
                        history_cursor=validated_page.next_cursor,
                        resume_history_cursor=decoded_cursor.resume_history_cursor
                        if decoded_cursor is not None
                        else None,
                    )
                )
                return KnowledgePage(
                    changes=(),
                    next_cursor=advanced,
                    proposed_checkpoint=advanced,
                    has_more=True,
                )
            complete = self._encode_complete_cursor(
                conversation_id=conversation_id,
                conversation_kind=resolved_kind,
                window=window,
            )
            return KnowledgePage(
                changes=(),
                next_cursor=None,
                proposed_checkpoint=complete,
                has_more=False,
            )
        message = validated_page.items[0]
        if message.root_thread_ts is not None:
            resume_cursor = (
                decoded_cursor.resume_history_cursor if decoded_cursor is not None else None
            ) or validated_page.next_cursor
            if validated_page.next_cursor is not None:
                next_cursor = self._encode_cursor(
                    _SlackConversationCursor(
                        schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                        conversation_id=conversation_id,
                        conversation_kind=resolved_kind,
                        root_oldest=window.oldest,
                        root_latest=window.latest,
                        phase="history",
                        history_cursor=validated_page.next_cursor,
                        resume_history_cursor=resume_cursor,
                    )
                )
                return KnowledgePage(
                    changes=(),
                    next_cursor=next_cursor,
                    proposed_checkpoint=next_cursor,
                    has_more=True,
                )
            complete = self._encode_complete_cursor(
                conversation_id=conversation_id,
                conversation_kind=resolved_kind,
                window=window,
            )
            return KnowledgePage(
                changes=(),
                next_cursor=None,
                proposed_checkpoint=complete,
                has_more=False,
            )
        change = self._message_to_change(message, conversation_id=conversation_id)
        resume_cursor = (
            decoded_cursor.resume_history_cursor if decoded_cursor is not None else None
        ) or validated_page.next_cursor
        if message.reply_count and message.reply_count > 0:
            reply_cursor = self._encode_cursor(
                _SlackConversationCursor(
                    schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                    conversation_id=conversation_id,
                    conversation_kind=resolved_kind,
                    root_oldest=window.oldest,
                    root_latest=window.latest,
                    phase="replies",
                    resume_history_cursor=resume_cursor,
                    root_message_ts=message.message_ts,
                    root_message_revision=compute_slack_conversation_message_revision(message),
                )
            )
            return KnowledgePage(
                changes=(change,),
                next_cursor=reply_cursor,
                proposed_checkpoint=reply_cursor,
                has_more=True,
            )
        if validated_page.next_cursor is not None:
            next_cursor = self._encode_cursor(
                _SlackConversationCursor(
                    schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                    conversation_id=conversation_id,
                    conversation_kind=resolved_kind,
                    root_oldest=window.oldest,
                    root_latest=window.latest,
                    phase="history",
                    history_cursor=validated_page.next_cursor,
                    resume_history_cursor=resume_cursor,
                )
            )
            return KnowledgePage(
                changes=(change,),
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        complete = self._encode_complete_cursor(
            conversation_id=conversation_id,
            conversation_kind=resolved_kind,
            window=window,
        )
        return KnowledgePage(
            changes=(change,),
            next_cursor=None,
            proposed_checkpoint=complete,
            has_more=False,
        )

    async def _read_replies_page(
        self,
        *,
        slack_integration: SlackConversationChannelIntegration,
        conversation_id: str,
        window: SlackConversationSourceWindow,
        decoded_cursor: _SlackConversationCursor,
        limit: int,
    ) -> KnowledgePage:
        root_message_ts, root_message_revision = self._require_replies_cursor_fields(decoded_cursor)
        provider_limit = min(limit, _REPLY_PROVIDER_PAGE_LIMIT)
        page = await self._invoke_integration(
            lambda: slack_integration.read_thread_replies_page(
                conversation_id=conversation_id,
                conversation_kind=decoded_cursor.conversation_kind,
                root_message_ts=root_message_ts,
                window=window,
                cursor=decoded_cursor.reply_cursor,
                limit=provider_limit,
                max_chars_per_message=DEFAULT_MESSAGE_MAX_CHARS,
            )
        )
        validated_page = self._validate_message_page_for_source(
            page,
            conversation_id=conversation_id,
            window=window,
        )
        changes = tuple(
            self._message_to_change(item, conversation_id=conversation_id)
            for item in validated_page.items
        )
        if validated_page.next_cursor is not None:
            next_cursor = self._encode_cursor(
                _SlackConversationCursor(
                    schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                    conversation_id=conversation_id,
                    conversation_kind=decoded_cursor.conversation_kind,
                    root_oldest=window.oldest,
                    root_latest=window.latest,
                    phase="replies",
                    resume_history_cursor=decoded_cursor.resume_history_cursor,
                    root_message_ts=root_message_ts,
                    root_message_revision=root_message_revision,
                    reply_cursor=validated_page.next_cursor,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        if decoded_cursor.resume_history_cursor is not None:
            history_cursor = self._encode_cursor(
                _SlackConversationCursor(
                    schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                    conversation_id=conversation_id,
                    conversation_kind=decoded_cursor.conversation_kind,
                    root_oldest=window.oldest,
                    root_latest=window.latest,
                    phase="history",
                    history_cursor=decoded_cursor.resume_history_cursor,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=history_cursor,
                proposed_checkpoint=history_cursor,
                has_more=True,
            )
        complete = self._encode_complete_cursor(
            conversation_id=conversation_id,
            conversation_kind=decoded_cursor.conversation_kind,
            window=window,
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=complete,
            has_more=False,
        )

    def _require_slack_integration(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> SlackConversationChannelIntegration:
        if not isinstance(integration, SlackConversationChannelIntegration):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message=(
                    "Slack conversation knowledge adapter requires "
                    "SlackConversationChannelIntegration"
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
            validated_source = KnowledgeSourceRef.model_validate(source.model_dump(mode="python"))
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
        if scope.remote_scope_type != SLACK_CONVERSATION_SCOPE_TYPE:
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
    ) -> tuple[str, SlackConversationKind, SlackConversationSourceWindow]:
        try:
            decoded = _decode_scope_payload(source.scope.remote_scope_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_source_scope_error(
                provider_id=source.provider_id,
                source_kind=source.source_kind,
            ) from None
        window = SlackConversationSourceWindow(
            oldest=decoded.root_oldest,
            latest=decoded.root_latest,
        )
        return decoded.conversation_id, decoded.conversation_kind, window

    def _invalid_source_scope_error(
        self,
        *,
        provider_id: str | None = None,
        source_kind: str | None = None,
    ) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message="Slack conversation knowledge source scope is invalid",
            provider_id=provider_id or self.provider_id,
            source_kind=source_kind or self.source_kind,
            retryable=False,
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Slack conversation knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Slack conversation knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return limit

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        oldest: str,
        latest: str,
    ) -> _SlackConversationCursor | None:
        if cursor is None:
            return None
        if not isinstance(cursor, KnowledgeCursor):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Slack conversation knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            decoded = _decode_cursor_payload(cursor.value)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Slack conversation knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if (
            decoded.conversation_id != conversation_id
            or decoded.conversation_kind != conversation_kind
            or decoded.root_oldest != oldest
            or decoded.root_latest != latest
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Slack conversation knowledge cursor does not match source scope",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded

    def _validate_message_page_for_source(
        self,
        page: object,
        *,
        conversation_id: str,
        window: SlackConversationSourceWindow,
    ) -> SlackConversationMessagePage:
        if type(page) is not SlackConversationMessagePage:
            raise ValueError("page must be a SlackConversationMessagePage")
        if page.conversation_id != conversation_id:
            raise ValueError("page scope does not match source")
        if page.oldest != window.oldest or page.latest != window.latest:
            raise ValueError("page window does not match source")
        seen: set[str] = set()
        validated_items: list[SlackConversationMessage] = []
        for item in page.items:
            validated = validate_slack_conversation_message(item)
            if validated.conversation_id != conversation_id:
                raise ValueError("item scope does not match source")
            if validated.message_ts in seen:
                raise ValueError("duplicate message on page")
            seen.add(validated.message_ts)
            validated_items.append(validated)
        return SlackConversationMessagePage(
            conversation_id=conversation_id,
            oldest=window.oldest,
            latest=window.latest,
            items=tuple(validated_items),
            next_cursor=page.next_cursor,
        )

    def _message_to_change(
        self,
        message: SlackConversationMessage,
        *,
        conversation_id: str,
    ) -> KnowledgeChange:
        opaque_id = self._encode_message_identity(
            conversation_id=conversation_id,
            message_ts=message.message_ts,
        )
        descriptor = self._active_message_to_descriptor(
            message,
            opaque_remote_id=opaque_id,
            conversation_id=conversation_id,
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=opaque_id,
            descriptor=descriptor,
        )

    def _active_message_to_descriptor(
        self,
        message: SlackConversationMessage,
        *,
        opaque_remote_id: str,
        conversation_id: str,
    ) -> KnowledgeItemDescriptor:
        revision = compute_slack_conversation_message_revision(message)
        metadata: dict[str, object] = {
            "subtype": message.subtype,
            "has_files": bool(message.files),
            "reply_count": message.reply_count,
            "created_at": message.created_at.isoformat(),
            "edited_at": message.edited_at.isoformat() if message.edited_at is not None else None,
            "thread_root_ts": message.root_thread_ts,
            "attachment_inventory_in_content": True,
        }
        parent_remote_id = None
        if message.root_thread_ts is not None:
            parent_remote_id = self._encode_message_identity(
                conversation_id=conversation_id,
                message_ts=message.root_thread_ts,
            )
        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=opaque_remote_id,
                parent_remote_id=parent_remote_id,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=self._encode_revision(revision),
                updated_at=message.edited_at or message.created_at,
            ),
            title=_resolve_message_title(message.text),
            item_type="slack_conversation_message",
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

    def _encode_message_identity(self, *, conversation_id: str, message_ts: str) -> str:
        identity = _SlackConversationMessageIdentity(
            schema_version=_SLACK_CONVERSATION_MESSAGE_ID_SCHEMA_VERSION,
            conversation_id=conversation_id,
            message_ts=message_ts,
        )
        return _encode_canonical_payload(identity.model_dump())

    def _encode_revision(self, revision: str) -> str:
        encoded = _SlackConversationMessageRevision(
            schema_version=_SLACK_CONVERSATION_REVISION_SCHEMA_VERSION,
            revision=revision,
        )
        return _encode_canonical_payload(encoded.model_dump())

    def _encode_cursor(self, cursor: _SlackConversationCursor) -> KnowledgeCursor:
        return KnowledgeCursor(
            value=_encode_canonical_payload(cursor.model_dump(mode="json")),
        )

    def _encode_complete_cursor(
        self,
        *,
        conversation_id: str,
        conversation_kind: SlackConversationKind,
        window: SlackConversationSourceWindow,
    ) -> KnowledgeCursor:
        return self._encode_cursor(
            _SlackConversationCursor(
                schema_version=SLACK_CONVERSATION_CURSOR_VERSION,
                conversation_id=conversation_id,
                conversation_kind=conversation_kind,
                root_oldest=window.oldest,
                root_latest=window.latest,
                phase="complete",
            )
        )

    def _require_replies_cursor_fields(
        self,
        decoded_cursor: _SlackConversationCursor,
    ) -> tuple[str, str]:
        if decoded_cursor.phase != "replies":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Slack conversation knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if decoded_cursor.root_message_ts is None or decoded_cursor.root_message_revision is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Slack conversation knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded_cursor.root_message_ts, decoded_cursor.root_message_revision

    def _validate_descriptor_for_fetch(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        conversation_id: str,
        window: SlackConversationSourceWindow,
    ) -> tuple[_SlackConversationMessageIdentity, str, dict[str, object]]:
        validated_item = self._deep_validate_item_descriptor(item)
        self._validate_item_provenance(validated_item, source=source)
        if validated_item.item_type != "slack_conversation_message":
            raise ValueError("invalid item type")
        if validated_item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise ValueError("invalid content mode")
        if not validated_item.content_available:
            raise ValueError("content unavailable")
        if validated_item.identity.logical_key is not None:
            raise ValueError("logical_key must be None")
        if validated_item.identity.remote_id != validated_item.provenance.remote_id:
            raise ValueError("identity provenance remote_id mismatch")
        if validated_item.revision.updated_at is None:
            raise ValueError("updated_at required")
        message_identity, revision = self._validate_message_item(
            validated_item,
            source=source,
            conversation_id=conversation_id,
        )
        metadata = self._validate_descriptor_metadata_strict(
            validated_item.metadata,
            message_identity=message_identity,
            parent_remote_id=validated_item.identity.parent_remote_id,
            updated_at=validated_item.revision.updated_at,
            window=window,
        )
        return message_identity, revision, metadata

    def _deep_validate_item_descriptor(self, item: KnowledgeItemDescriptor) -> KnowledgeItemDescriptor:
        validated = KnowledgeItemDescriptor.model_validate(item.model_dump(mode="python"))
        self._validate_descriptor_owned_fields(validated)
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

    def _validate_item_provenance(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        if item.provenance.provider_id != source.provider_id:
            raise ValueError("provenance provider mismatch")
        if item.provenance.source_kind != source.source_kind:
            raise ValueError("provenance source_kind mismatch")

    def _validate_message_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        conversation_id: str,
    ) -> tuple[_SlackConversationMessageIdentity, str]:
        self._validate_item_provenance(item, source=source)
        try:
            identity = _decode_message_identity_payload(item.identity.remote_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Slack conversation message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if identity.conversation_id != conversation_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Slack conversation message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            revision_payload = _decode_revision_payload(item.revision.version)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Slack conversation message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        return identity, revision_payload.revision

    def _validate_descriptor_metadata_strict(
        self,
        metadata: dict[str, object],
        *,
        message_identity: _SlackConversationMessageIdentity,
        parent_remote_id: str | None,
        updated_at: datetime,
        window: SlackConversationSourceWindow,
    ) -> dict[str, object]:
        if set(metadata.keys()) != _METADATA_REQUIRED_KEYS:
            raise ValueError("descriptor metadata keys mismatch")
        subtype = metadata.get("subtype")
        if subtype is not None and not isinstance(subtype, str):
            raise ValueError("invalid subtype")
        has_files = metadata.get("has_files")
        if type(has_files) is not bool:
            raise ValueError("invalid has_files")
        reply_count = metadata.get("reply_count")
        if reply_count is not None and (type(reply_count) is not int or reply_count < 0):
            raise ValueError("invalid reply_count")
        created_at_raw = metadata.get("created_at")
        if not isinstance(created_at_raw, str):
            raise ValueError("invalid created_at")
        edited_at_raw = metadata.get("edited_at")
        if edited_at_raw is not None and not isinstance(edited_at_raw, str):
            raise ValueError("invalid edited_at")
        thread_root_ts = metadata.get("thread_root_ts")
        if thread_root_ts is not None and not isinstance(thread_root_ts, str):
            raise ValueError("invalid thread_root_ts")
        if thread_root_ts is not None:
            thread_root_ts = validate_slack_timestamp(thread_root_ts)
        attachment_flag = metadata.get("attachment_inventory_in_content")
        if attachment_flag is not True:
            raise ValueError("attachment_inventory_in_content must be true")
        created_at = datetime.fromisoformat(created_at_raw)
        if created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        edited_at = None
        if edited_at_raw is not None:
            edited_at = datetime.fromisoformat(edited_at_raw)
            if edited_at.tzinfo is None:
                raise ValueError("edited_at must be timezone-aware")
            if edited_at < created_at:
                raise ValueError("edited_at before created_at")
        expected_created = parse_slack_ts(message_identity.message_ts)
        if expected_created is None or created_at != expected_created:
            raise ValueError("created_at does not match message_ts")
        if edited_at is not None:
            if updated_at != edited_at:
                raise ValueError("updated_at must equal edited_at when edited")
        elif updated_at != created_at:
            raise ValueError("updated_at must equal created_at when not edited")
        if parent_remote_id is None:
            if thread_root_ts is not None:
                raise ValueError("root metadata has thread_root_ts")
            if not slack_timestamp_in_window(
                value=message_identity.message_ts,
                oldest=window.oldest,
                latest=window.latest,
            ):
                raise ValueError("root message_ts outside root window")
        else:
            parent_identity = _decode_message_identity_payload(parent_remote_id)
            if parent_identity.conversation_id != message_identity.conversation_id:
                raise ValueError("parent conversation mismatch")
            if thread_root_ts is None:
                raise ValueError("reply missing thread_root_ts")
            if parent_identity.message_ts != thread_root_ts:
                raise ValueError("parent thread_root_ts mismatch")
            if parent_identity.message_ts == message_identity.message_ts:
                raise ValueError("reply message_ts equals root")
            if not slack_timestamp_in_window(
                value=thread_root_ts,
                oldest=window.oldest,
                latest=window.latest,
            ):
                raise ValueError("reply root outside root window")
            if not slack_timestamp_in_window(
                value=message_identity.message_ts,
                oldest=window.oldest,
                latest=window.latest,
            ):
                raise ValueError("reply message_ts outside reply window")
        return metadata

    def _validate_fetched_content_identity(
        self,
        message: SlackConversationMessage,
        *,
        message_identity: _SlackConversationMessageIdentity,
        revision: str,
        metadata: dict[str, object],
        updated_at: datetime | None,
    ) -> None:
        if message.conversation_id != message_identity.conversation_id:
            raise ValueError("content conversation mismatch")
        if message.message_ts != message_identity.message_ts:
            raise ValueError("content message_ts mismatch")
        actual_revision = compute_slack_conversation_message_revision(message)
        if actual_revision != revision:
            raise SlackConversationMessageChanged()

    def _build_structured_record(self, message: SlackConversationMessage) -> JsonObject:
        return {
            "schema": _STRUCTURED_RECORD_SCHEMA,
            "provider": self.provider_id,
            "source_kind": self.source_kind,
            "conversation": {
                "conversation_id": message.conversation_id,
            },
            "message": {
                "message_ts": message.message_ts,
                "subtype": message.subtype,
            },
            "thread": {
                "root_thread_ts": message.root_thread_ts,
                "reply_count": message.reply_count,
            },
            "actor": {
                "provider_id": message.actor_provider_id,
            },
            "text": message.text,
            "timestamps": {
                "created_at": message.created_at.isoformat(),
                "edited_at": message.edited_at.isoformat() if message.edited_at else None,
            },
            "edit_state": {
                "edited": message.edited_at is not None,
            },
            "safe_file_inventory": [
                {
                    "file_id": file_ref.file_id,
                    "safe_file_name": file_ref.safe_file_name,
                    "title": file_ref.title,
                    "mimetype": file_ref.mimetype,
                    "filetype": file_ref.filetype,
                    "size": file_ref.size,
                    "mode": file_ref.mode,
                    "created_at": (
                        file_ref.created_at.isoformat() if file_ref.created_at else None
                    ),
                    "is_external": file_ref.is_external,
                }
                for file_ref in message.files
            ],
            "safe_subtype_flags": {
                "subtype": message.subtype,
            },
        }

    async def _invoke_integration(self, operation: Callable[[], _T]) -> _T:
        try:
            result = operation()
            if asyncio.iscoroutine(result):
                return await result
            return result
        except VendorKnowledgeError:
            raise
        except SlackConversationContentTooLarge:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Slack conversation message exceeds the configured content limit",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except SlackConversationReadConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Slack conversation knowledge adapter configuration is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except SlackConversationMessageNotFound:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND,
                safe_message="Slack conversation message was not found",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except SlackConversationMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Slack conversation message changed during content read",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except SlackConversationReadError as exc:
            if exc.slack_error == "ratelimited":
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.RATE_LIMITED,
                    safe_message="Slack conversation knowledge dependency is rate limited",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=True,
                ) from None
            if exc.slack_error in {
                "invalid_auth",
                "token_revoked",
                "not_authed",
                "account_inactive",
                "token_expired",
                "not_allowed_token_type",
            }:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
                    safe_message="Slack conversation knowledge authentication failed",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                ) from None
            if exc.slack_error in {
                "missing_scope",
                "no_permission",
                "not_in_channel",
                "access_denied",
                "restricted_action",
                "team_access_not_granted",
                "accesslimited",
                "enterprise_is_restricted",
                "ekm_access_denied",
                "org_login_required",
                "two_factor_setup_required",
            }:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
                    safe_message="Slack conversation knowledge authorization failed",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                ) from None
            if exc.slack_error == "malformed_response":
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Slack conversation knowledge provider response is invalid",
                    provider_id=self.provider_id,
                    source_kind=self.source_kind,
                    retryable=False,
                ) from None
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Slack conversation knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Slack conversation knowledge adapter configuration is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Slack conversation knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Slack conversation knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
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


def _decode_scope_payload(value: str) -> _SlackConversationScope:
    data = _decode_canonical_payload(value)
    scope = _SlackConversationScope.model_validate(data)
    if _encode_canonical_payload(scope.model_dump(mode="json")) != value:
        raise ValueError("scope payload is not canonical")
    return scope


def _decode_cursor_payload(value: str) -> _SlackConversationCursor:
    data = _decode_canonical_payload(value)
    cursor = _SlackConversationCursor.model_validate(data)
    if _encode_canonical_payload(cursor.model_dump(mode="json")) != value:
        raise ValueError("cursor payload is not canonical")
    return cursor


def _decode_message_identity_payload(value: str) -> _SlackConversationMessageIdentity:
    data = _decode_canonical_payload(value)
    identity = _SlackConversationMessageIdentity.model_validate(data)
    if _encode_canonical_payload(identity.model_dump()) != value:
        raise ValueError("message identity payload is not canonical")
    return identity


def _decode_revision_payload(value: str) -> _SlackConversationMessageRevision:
    data = _decode_canonical_payload(value)
    revision = _SlackConversationMessageRevision.model_validate(data)
    if _encode_canonical_payload(revision.model_dump()) != value:
        raise ValueError("revision payload is not canonical")
    return revision


def _validate_opaque_revision(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("revision must be a string")
    if value == "":
        raise ValueError("revision must not be empty")
    if value != value.strip():
        raise ValueError("revision must not have leading or trailing whitespace")
    if _ASCII_CONTROL.search(value):
        raise ValueError("revision must not contain control characters")
    if len(value) > _MAX_REVISION_LEN:
        raise ValueError("revision exceeds maximum length")
    return value


def _resolve_message_title(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return _TITLE_FALLBACK
    if len(stripped) > 120:
        return f"{stripped[:117]}..."
    return stripped


def register_slack_conversation_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> SlackConversationKnowledgeAdapter:
    adapter = SlackConversationKnowledgeAdapter()
    registry.register(adapter)
    return adapter


__all__ = [
    "SLACK_CONVERSATION_CURSOR_VERSION",
    "SLACK_CONVERSATION_SCOPE_TYPE",
    "SlackConversationKnowledgeAdapter",
    "encode_slack_conversation_scope_id",
    "register_slack_conversation_knowledge_adapter",
]
