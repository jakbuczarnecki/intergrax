# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Teams Channel knowledge source adapter (MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL)."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_content import (
    MsGraphTeamsChannelContentTooLarge,
    MsGraphTeamsChannelMessageReference,
    validate_msgraph_teams_channel_message_content,
    validate_msgraph_teams_channel_message_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    MsGraphTeamsChannelReference,
    validate_msgraph_teams_channel_id,
    validate_msgraph_teams_channel_message_id,
    validate_msgraph_teams_channel_reference,
    validate_msgraph_teams_team_id,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
    DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChannelMessageType,
    MsGraphTeamsChannelReplyPage,
    MsGraphTeamsChannelRootMessagePage,
    MsGraphTeamsChannelRootMessageReference,
    validate_msgraph_teams_channel_message,
    validate_msgraph_teams_channel_root_message_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_messages import (
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatMention,
    MsGraphTeamsChatReaction,
    MsGraphTeamsForwardedMessageReference,
    MsGraphTeamsIdentity,
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

MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE = "msgraph_teams_channel"
MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION = "msgraph.teams-channel.cursor.v1"
_MSGRAPH_TEAMS_CHANNEL_SCOPE_SCHEMA_VERSION = "msgraph.teams-channel.scope.v1"
_MSGRAPH_TEAMS_CHANNEL_MESSAGE_ID_SCHEMA_VERSION = "msgraph.teams-channel.message-id.v1"
_MSGRAPH_TEAMS_CHANNEL_REVISION_SCHEMA_VERSION = "msgraph.teams-channel.revision.v1"

_STRUCTURED_RECORD_SCHEMA = "msgraph.teams-channel.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-teams-channel-message+json"
_ROOT_TITLE_FALLBACK = "Teams channel post"
_REPLY_TITLE_FALLBACK = "Teams channel reply"

_MAX_CONTINUATION_URL_LEN = 32_768
_MAX_REVISION_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_ROOT_PROVIDER_PAGE_LIMIT = 1
_REPLY_PROVIDER_PAGE_LIMIT = 50

_METADATA_REQUIRED_KEYS = frozenset(
    {
        "message_state",
        "message_kind",
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


class _MsGraphTeamsChannelScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-channel.scope.v1"]
    team_remote_id: str = Field(repr=False)
    channel_remote_id: str = Field(repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)


class _MsGraphTeamsChannelMessageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-channel.message-id.v1"]
    team_remote_id: str = Field(repr=False)
    channel_remote_id: str = Field(repr=False)
    thread_root_remote_id: str = Field(repr=False)
    message_kind: Literal["root", "reply"]
    message_remote_id: str = Field(repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("thread_root_remote_id", "message_remote_id", mode="before")
    @classmethod
    def _validate_message_ids(cls, value: object) -> str:
        return validate_msgraph_teams_channel_message_id(value)

    @model_validator(mode="after")
    def _validate_identity_shape(self) -> _MsGraphTeamsChannelMessageIdentity:
        if self.message_kind == "root":
            if self.thread_root_remote_id != self.message_remote_id:
                raise ValueError("root identity thread root mismatch")
        elif self.thread_root_remote_id == self.message_remote_id:
            raise ValueError("reply identity thread root equals message id")
        return self


class _MsGraphTeamsChannelMessageRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-channel.revision.v1"]
    revision: str = Field(repr=False)

    @field_validator("revision", mode="before")
    @classmethod
    def _validate_revision_field(cls, value: object) -> str:
        return _validate_opaque_revision(value)


class _MsGraphTeamsChannelCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.teams-channel.cursor.v1"]
    team_remote_id: str = Field(repr=False)
    channel_remote_id: str = Field(repr=False)
    phase: Literal["roots", "replies", "complete"]
    resume_root_continuation_url: str | None = Field(default=None, repr=False)
    root_message_remote_id: str | None = Field(default=None, repr=False)
    root_message_revision: str | None = Field(default=None, repr=False)
    root_message_state: Literal["active", "deleted"] | None = None
    reply_continuation_url: str | None = Field(default=None, repr=False)

    @field_validator("team_remote_id", mode="before")
    @classmethod
    def _validate_team_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_team_id(value)

    @field_validator("channel_remote_id", mode="before")
    @classmethod
    def _validate_channel_remote_id(cls, value: object) -> str:
        return validate_msgraph_teams_channel_id(value)

    @field_validator("root_message_remote_id", mode="before")
    @classmethod
    def _validate_root_message_remote_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_teams_channel_message_id(value)

    @field_validator("root_message_revision", mode="before")
    @classmethod
    def _validate_root_message_revision(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_opaque_revision(value)

    @field_validator("resume_root_continuation_url", "reply_continuation_url", mode="before")
    @classmethod
    def _validate_continuation_url(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_continuation_url(value)

    @model_validator(mode="after")
    def _validate_phase_shape(self) -> _MsGraphTeamsChannelCursor:
        if self.phase == "roots":
            if self.resume_root_continuation_url is None:
                raise ValueError("roots phase requires resume_root_continuation_url")
            if (
                self.root_message_remote_id is not None
                or self.root_message_revision is not None
                or self.root_message_state is not None
                or self.reply_continuation_url is not None
            ):
                raise ValueError("roots phase has forbidden fields")
        elif self.phase == "replies":
            if (
                self.root_message_remote_id is None
                or self.root_message_revision is None
                or self.root_message_state is None
            ):
                raise ValueError("replies phase missing root reference fields")
        elif self.phase == "complete":
            if (
                self.resume_root_continuation_url is not None
                or self.root_message_remote_id is not None
                or self.root_message_revision is not None
                or self.root_message_state is not None
                or self.reply_continuation_url is not None
            ):
                raise ValueError("complete phase has forbidden fields")
        return self


def encode_msgraph_teams_channel_scope_id(
    *,
    team_remote_id: str,
    channel_remote_id: str,
) -> str:
    """Return the canonical opaque remote_scope_id for one Teams channel source."""
    scope = _MsGraphTeamsChannelScope(
        schema_version=_MSGRAPH_TEAMS_CHANNEL_SCOPE_SCHEMA_VERSION,
        team_remote_id=team_remote_id,
        channel_remote_id=channel_remote_id,
    )
    return _encode_canonical_payload(scope.model_dump())


class MsGraphTeamsChannelKnowledgeAdapter:
    """Thin mapping from Microsoft Graph Teams Channel integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND

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
        team_remote_id, channel_remote_id = self._decode_scope(validated_source)
        self._validate_limit(limit)
        decoded_cursor = self._decode_cursor(
            cursor,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
        )
        if decoded_cursor is not None and decoded_cursor.phase == "complete":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Teams Channel reconciliation cursor is complete; "
                    "restart reconciliation"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if decoded_cursor is None or decoded_cursor.phase == "roots":
            return await self._read_roots_page(
                graph_integration=graph_integration,
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                decoded_cursor=decoded_cursor,
            )
        return await self._read_replies_page(
            graph_integration=graph_integration,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
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
        graph_integration = self._require_graph_integration(integration=integration, source=source)
        validated_source = self._validate_source_ref(source)
        team_remote_id, channel_remote_id = self._decode_scope(validated_source)
        try:
            validated_item = self._deep_validate_item_descriptor(item)
            message_identity, revision = self._validate_message_item(
                validated_item,
                source=validated_source,
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
            )
            metadata = self._validate_descriptor_metadata(
                validated_item.metadata,
                message_identity=message_identity,
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
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        result = await self._invoke_integration(
            lambda: graph_integration.read_teams_channel_message_content(
                message=provider_reference,
                max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
            ),
        )
        try:
            validated_content = validate_msgraph_teams_channel_message_content(
                result,
                reference=provider_reference,
                max_chars=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
            )
            self._validate_fetched_content_identity(
                validated_content,
                message_identity=message_identity,
                revision=revision,
                metadata=metadata,
                updated_at=validated_item.revision.updated_at,
            )
        except MsGraphTeamsChannelMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Channel knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Channel knowledge provider response is invalid",
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
        self._require_graph_integration(integration=integration, source=source)
        validated_source = self._validate_source_ref(source)
        validated_item = self._deep_validate_item_descriptor(item)
        self._validate_item_provenance(validated_item, source=validated_source)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=(
                "Microsoft Graph Teams Channel authoritative permission projection "
                "is not implemented"
            ),
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    async def _read_roots_page(
        self,
        *,
        graph_integration: Ms365GraphCollaborationSuiteIntegration,
        team_remote_id: str,
        channel_remote_id: str,
        decoded_cursor: _MsGraphTeamsChannelCursor | None,
    ) -> KnowledgePage:
        channel_reference = validate_msgraph_teams_channel_reference(
            MsGraphTeamsChannelReference(
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
            )
        )
        provider_continuation: MsGraphKnowledgeContinuation | None = None
        if decoded_cursor is not None:
            provider_continuation = MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=decoded_cursor.resume_root_continuation_url,
            )
        page = await self._invoke_integration(
            lambda: graph_integration.read_teams_channel_root_messages_page_by_reference(
                channel=channel_reference,
                continuation=provider_continuation,
                limit=_ROOT_PROVIDER_PAGE_LIMIT,
                max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
            )
        )
        try:
            validated_page = self._validate_root_page_for_source(
                page,
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Channel knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if not validated_page.items:
            if validated_page.continuation is not None:
                advanced = self._encode_cursor(
                    _MsGraphTeamsChannelCursor(
                        schema_version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                        team_remote_id=team_remote_id,
                        channel_remote_id=channel_remote_id,
                        phase="roots",
                        resume_root_continuation_url=validated_page.continuation.url,
                    )
                )
                return KnowledgePage(
                    changes=(),
                    next_cursor=advanced,
                    proposed_checkpoint=advanced,
                    has_more=True,
                )
            complete = self._encode_complete_cursor(
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
            )
            return KnowledgePage(
                changes=(),
                next_cursor=None,
                proposed_checkpoint=complete,
                has_more=False,
            )
        if len(validated_page.items) > 1:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Channel knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        root_message = validated_page.items[0]
        root_change = self._message_to_change(
            root_message,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
        )
        resume_url = (
            validated_page.continuation.url if validated_page.continuation is not None else None
        )
        reply_cursor = self._encode_cursor(
            _MsGraphTeamsChannelCursor(
                schema_version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                phase="replies",
                resume_root_continuation_url=resume_url,
                root_message_remote_id=root_message.remote_id,
                root_message_revision=root_message.revision,
                root_message_state=self._cursor_root_state(root_message.state),
            )
        )
        return KnowledgePage(
            changes=(root_change,),
            next_cursor=reply_cursor,
            proposed_checkpoint=reply_cursor,
            has_more=True,
        )

    async def _read_replies_page(
        self,
        *,
        graph_integration: Ms365GraphCollaborationSuiteIntegration,
        team_remote_id: str,
        channel_remote_id: str,
        decoded_cursor: _MsGraphTeamsChannelCursor,
        limit: int,
    ) -> KnowledgePage:
        assert decoded_cursor.root_message_remote_id is not None
        assert decoded_cursor.root_message_revision is not None
        assert decoded_cursor.root_message_state is not None
        root_reference = validate_msgraph_teams_channel_root_message_reference(
            MsGraphTeamsChannelRootMessageReference(
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                remote_id=decoded_cursor.root_message_remote_id,
                revision=decoded_cursor.root_message_revision,
                state=self._provider_root_state(decoded_cursor.root_message_state),
            )
        )
        reply_continuation: MsGraphKnowledgeContinuation | None = None
        if decoded_cursor.reply_continuation_url is not None:
            reply_continuation = MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=decoded_cursor.reply_continuation_url,
            )
        provider_limit = min(limit, _REPLY_PROVIDER_PAGE_LIMIT)
        page = await self._invoke_integration(
            lambda: graph_integration.read_teams_channel_replies_page_by_reference(
                root_message=root_reference,
                continuation=reply_continuation,
                limit=provider_limit,
                max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
            )
        )
        try:
            validated_page = self._validate_reply_page_for_source(
                page,
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                root_message_remote_id=decoded_cursor.root_message_remote_id,
                root_message_revision=decoded_cursor.root_message_revision,
            )
            changes = tuple(
                self._message_to_change(
                    item,
                    team_remote_id=team_remote_id,
                    channel_remote_id=channel_remote_id,
                )
                for item in validated_page.items
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Channel knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if validated_page.continuation is not None:
            next_cursor = self._encode_cursor(
                _MsGraphTeamsChannelCursor(
                    schema_version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                    team_remote_id=team_remote_id,
                    channel_remote_id=channel_remote_id,
                    phase="replies",
                    resume_root_continuation_url=decoded_cursor.resume_root_continuation_url,
                    root_message_remote_id=decoded_cursor.root_message_remote_id,
                    root_message_revision=decoded_cursor.root_message_revision,
                    root_message_state=decoded_cursor.root_message_state,
                    reply_continuation_url=validated_page.continuation.url,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=next_cursor,
                proposed_checkpoint=next_cursor,
                has_more=True,
            )
        if decoded_cursor.resume_root_continuation_url is not None:
            roots_cursor = self._encode_cursor(
                _MsGraphTeamsChannelCursor(
                    schema_version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                    team_remote_id=team_remote_id,
                    channel_remote_id=channel_remote_id,
                    phase="roots",
                    resume_root_continuation_url=decoded_cursor.resume_root_continuation_url,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=roots_cursor,
                proposed_checkpoint=roots_cursor,
                has_more=True,
            )
        complete = self._encode_complete_cursor(
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=complete,
            has_more=False,
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
                    "Microsoft Graph Teams Channel knowledge adapter requires "
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
        if scope.remote_scope_type != MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE:
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

    def _decode_scope(self, source: KnowledgeSourceRef) -> tuple[str, str]:
        try:
            decoded = _decode_scope_payload(source.scope.remote_scope_id)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_source_scope_error(
                provider_id=source.provider_id,
                source_kind=source.source_kind,
            ) from None
        return decoded.team_remote_id, decoded.channel_remote_id

    def _invalid_source_scope_error(
        self,
        *,
        provider_id: str | None = None,
        source_kind: str | None = None,
    ) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message="Microsoft Graph Teams Channel knowledge source scope is invalid",
            provider_id=provider_id or self.provider_id,
            source_kind=source_kind or self.source_kind,
            retryable=False,
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Channel knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Channel knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return limit

    def _deep_validate_item_descriptor(self, item: object) -> KnowledgeItemDescriptor:
        if not isinstance(item, KnowledgeItemDescriptor):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if type(item.content_mode) is not KnowledgeContentMode:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            return KnowledgeItemDescriptor.model_validate(item.model_dump(mode="python"))
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
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
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _validate_root_page_for_source(
        self,
        page: object,
        *,
        team_remote_id: str,
        channel_remote_id: str,
    ) -> MsGraphTeamsChannelRootMessagePage:
        if not isinstance(page, MsGraphTeamsChannelRootMessagePage):
            raise ValueError("page must be a MsGraphTeamsChannelRootMessagePage")
        if page.team_remote_id != team_remote_id or page.channel_remote_id != channel_remote_id:
            raise ValueError("page scope does not match source")
        if len(page.items) > 1:
            raise ValueError("root page contains more than one item")
        for item in page.items:
            validated = validate_msgraph_teams_channel_message(item)
            if (
                validated.team_remote_id != team_remote_id
                or validated.channel_remote_id != channel_remote_id
            ):
                raise ValueError("item scope does not match source")
            if validated.message_kind is not MsGraphTeamsChannelMessageKind.ROOT:
                raise ValueError("item is not a root message")
            if validated.thread_root_remote_id != validated.remote_id:
                raise ValueError("root thread root mismatch")
        if page.continuation is not None:
            self._validate_page_continuation(page.continuation)
        return page

    def _validate_reply_page_for_source(
        self,
        page: object,
        *,
        team_remote_id: str,
        channel_remote_id: str,
        root_message_remote_id: str,
        root_message_revision: str,
    ) -> MsGraphTeamsChannelReplyPage:
        if not isinstance(page, MsGraphTeamsChannelReplyPage):
            raise ValueError("page must be a MsGraphTeamsChannelReplyPage")
        if page.team_remote_id != team_remote_id or page.channel_remote_id != channel_remote_id:
            raise ValueError("page scope does not match source")
        if page.root_message_remote_id != root_message_remote_id:
            raise ValueError("page root id mismatch")
        if page.root_message_revision != root_message_revision:
            raise ValueError("page root revision mismatch")
        seen_remote_ids: set[str] = set()
        for item in page.items:
            validated = validate_msgraph_teams_channel_message(item)
            if (
                validated.team_remote_id != team_remote_id
                or validated.channel_remote_id != channel_remote_id
            ):
                raise ValueError("item scope does not match source")
            if validated.message_kind is not MsGraphTeamsChannelMessageKind.REPLY:
                raise ValueError("item is not a reply message")
            if validated.thread_root_remote_id != root_message_remote_id:
                raise ValueError("reply thread root mismatch")
            opaque_id = self._encode_message_identity(
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                thread_root_remote_id=validated.thread_root_remote_id,
                message_kind="reply",
                message_remote_id=validated.remote_id,
            )
            if opaque_id in seen_remote_ids:
                raise ValueError("duplicate reply neutral id on page")
            seen_remote_ids.add(opaque_id)
        if page.continuation is not None:
            self._validate_page_continuation(page.continuation)
        return page

    def _validate_page_continuation(self, continuation: MsGraphKnowledgeContinuation) -> None:
        if continuation.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError("invalid continuation kind")
        _validate_continuation_url(continuation.url)

    def _message_to_change(
        self,
        message: MsGraphTeamsChannelMessage,
        *,
        team_remote_id: str,
        channel_remote_id: str,
    ) -> KnowledgeChange:
        message_kind = (
            "root"
            if message.message_kind is MsGraphTeamsChannelMessageKind.ROOT
            else "reply"
        )
        opaque_id = self._encode_message_identity(
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
            thread_root_remote_id=message.thread_root_remote_id,
            message_kind=message_kind,
            message_remote_id=message.remote_id,
        )
        if message.state is MsGraphTeamsChannelMessageState.DELETED:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=opaque_id,
                descriptor=None,
            )
        descriptor = self._active_message_to_descriptor(
            message,
            opaque_remote_id=opaque_id,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
            message_kind=message_kind,
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=opaque_id,
            descriptor=descriptor,
        )

    def _active_message_to_descriptor(
        self,
        message: MsGraphTeamsChannelMessage,
        *,
        opaque_remote_id: str,
        team_remote_id: str,
        channel_remote_id: str,
        message_kind: Literal["root", "reply"],
    ) -> KnowledgeItemDescriptor:
        parent_remote_id: str | None = None
        if message_kind == "reply":
            parent_remote_id = self._encode_message_identity(
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                thread_root_remote_id=message.thread_root_remote_id,
                message_kind="root",
                message_remote_id=message.thread_root_remote_id,
            )
        title = _resolve_message_title(
            subject=message.subject,
            message_kind=message_kind,
        )
        assert message.body_kind is not None
        metadata: dict[str, object] = {
            "message_state": "active",
            "message_kind": message_kind,
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
                parent_remote_id=parent_remote_id,
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
            item_type="msgraph_teams_channel_message",
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
        team_remote_id: str,
        channel_remote_id: str,
    ) -> tuple[_MsGraphTeamsChannelMessageIdentity, _MsGraphTeamsChannelMessageRevision]:
        self._validate_item_provenance(item, source=source)
        if item.item_type != "msgraph_teams_channel_message":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.identity.logical_key is not None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        message_identity = self._decode_message_identity(item.identity.remote_id)
        if (
            message_identity.team_remote_id != team_remote_id
            or message_identity.channel_remote_id != channel_remote_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        version = item.revision.version
        if not isinstance(version, str) or not version.strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.revision.updated_at is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
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
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        revision = self._decode_revision(version)
        expected_parent = None
        if message_identity.message_kind == "reply":
            expected_parent = self._encode_message_identity(
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                thread_root_remote_id=message_identity.thread_root_remote_id,
                message_kind="root",
                message_remote_id=message_identity.thread_root_remote_id,
            )
        if message_identity.message_kind == "root":
            if item.identity.parent_remote_id is not None:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                    safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
        elif item.identity.parent_remote_id != expected_parent:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return message_identity, revision

    def _validate_descriptor_metadata(
        self,
        metadata: object,
        *,
        message_identity: _MsGraphTeamsChannelMessageIdentity,
        updated_at: datetime | None,
    ) -> dict[str, object]:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        keys = set(metadata.keys())
        if keys != _METADATA_REQUIRED_KEYS:
            raise ValueError("metadata keys are invalid")
        if metadata["message_state"] != "active":
            raise ValueError("message_state must be active")
        if metadata["message_kind"] != message_identity.message_kind:
            raise ValueError("message_kind mismatch")
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
            MsGraphTeamsChannelMessageType(metadata["message_type"])
        except ValueError:
            raise ValueError("message_type is invalid") from None
        if not isinstance(metadata["importance"], str):
            raise ValueError("importance must be a string")
        try:
            MsGraphTeamsChannelImportance(metadata["importance"])
        except ValueError:
            raise ValueError("importance is invalid") from None
        if metadata["body_kind"] not in {"text", "html"}:
            raise ValueError("body_kind is invalid")
        created_at = self._parse_timezone_aware_iso(metadata["created_at"])
        last_modified_at = self._parse_timezone_aware_iso(metadata["last_modified_at"])
        if updated_at is None or last_modified_at != updated_at:
            raise ValueError("last_modified_at mismatch")
        last_edited_raw = metadata["last_edited_at"]
        if last_edited_raw is not None:
            self._parse_timezone_aware_iso(last_edited_raw)
        event_detail_type = metadata["event_detail_type"]
        if event_detail_type is not None and not isinstance(event_detail_type, str):
            raise ValueError("event_detail_type must be a string or None")
        locale = metadata["locale"]
        if locale is not None and not isinstance(locale, str):
            raise ValueError("locale must be a string or None")
        if created_at > last_modified_at:
            raise ValueError("created_at after last_modified_at")
        return metadata

    def _descriptor_to_provider_reference(
        self,
        *,
        message_identity: _MsGraphTeamsChannelMessageIdentity,
        revision: _MsGraphTeamsChannelMessageRevision,
    ) -> MsGraphTeamsChannelMessageReference:
        provider_kind = (
            MsGraphTeamsChannelMessageKind.ROOT
            if message_identity.message_kind == "root"
            else MsGraphTeamsChannelMessageKind.REPLY
        )
        return validate_msgraph_teams_channel_message_reference(
            MsGraphTeamsChannelMessageReference(
                team_remote_id=message_identity.team_remote_id,
                channel_remote_id=message_identity.channel_remote_id,
                thread_root_remote_id=message_identity.thread_root_remote_id,
                message_kind=provider_kind,
                remote_id=message_identity.message_remote_id,
                revision=revision.revision,
            )
        )

    def _validate_fetched_content_identity(
        self,
        content: MsGraphTeamsChannelMessage,
        *,
        message_identity: _MsGraphTeamsChannelMessageIdentity,
        revision: _MsGraphTeamsChannelMessageRevision,
        metadata: dict[str, object],
        updated_at: datetime | None,
    ) -> None:
        if content.team_remote_id != message_identity.team_remote_id:
            raise ValueError("team mismatch")
        if content.channel_remote_id != message_identity.channel_remote_id:
            raise ValueError("channel mismatch")
        if content.thread_root_remote_id != message_identity.thread_root_remote_id:
            raise ValueError("thread root mismatch")
        expected_kind = (
            MsGraphTeamsChannelMessageKind.ROOT
            if message_identity.message_kind == "root"
            else MsGraphTeamsChannelMessageKind.REPLY
        )
        if content.message_kind != expected_kind:
            raise ValueError("message kind mismatch")
        if content.remote_id != message_identity.message_remote_id:
            raise ValueError("message id mismatch")
        if content.revision != revision.revision:
            raise ValueError("revision mismatch")
        if content.state is not MsGraphTeamsChannelMessageState.ACTIVE:
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

    def _build_structured_record(self, message: MsGraphTeamsChannelMessage) -> JsonObject:
        assert message.body_kind is not None
        assert message.body_content is not None
        return {
            "schema": _STRUCTURED_RECORD_SCHEMA,
            "message_kind": (
                "root"
                if message.message_kind is MsGraphTeamsChannelMessageKind.ROOT
                else "reply"
            ),
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
        team_remote_id: str,
        channel_remote_id: str,
        thread_root_remote_id: str,
        message_kind: Literal["root", "reply"],
        message_remote_id: str,
    ) -> str:
        identity = _MsGraphTeamsChannelMessageIdentity(
            schema_version=_MSGRAPH_TEAMS_CHANNEL_MESSAGE_ID_SCHEMA_VERSION,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
            thread_root_remote_id=thread_root_remote_id,
            message_kind=message_kind,
            message_remote_id=message_remote_id,
        )
        return _encode_canonical_payload(identity.model_dump())

    def _decode_message_identity(self, value: str) -> _MsGraphTeamsChannelMessageIdentity:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphTeamsChannelMessageIdentity.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_revision(self, revision: str) -> str:
        encoded = _MsGraphTeamsChannelMessageRevision(
            schema_version=_MSGRAPH_TEAMS_CHANNEL_REVISION_SCHEMA_VERSION,
            revision=revision,
        )
        return _encode_canonical_payload(encoded.model_dump())

    def _decode_revision(self, value: str) -> _MsGraphTeamsChannelMessageRevision:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphTeamsChannelMessageRevision.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Teams Channel message descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_cursor(self, cursor: _MsGraphTeamsChannelCursor) -> KnowledgeCursor:
        return KnowledgeCursor(
            value=_encode_canonical_payload(cursor.model_dump()),
            version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
        )

    def _encode_complete_cursor(
        self,
        *,
        team_remote_id: str,
        channel_remote_id: str,
    ) -> KnowledgeCursor:
        return self._encode_cursor(
            _MsGraphTeamsChannelCursor(
                schema_version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                team_remote_id=team_remote_id,
                channel_remote_id=channel_remote_id,
                phase="complete",
            )
        )

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        team_remote_id: str,
        channel_remote_id: str,
    ) -> _MsGraphTeamsChannelCursor | None:
        if cursor is None:
            return None
        if cursor.version != MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Teams Channel knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            data = _decode_canonical_payload(cursor.value)
            decoded = _MsGraphTeamsChannelCursor.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Teams Channel knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if (
            decoded.team_remote_id != team_remote_id
            or decoded.channel_remote_id != channel_remote_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Teams Channel knowledge cursor scope does not match source"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded

    def _cursor_root_state(
        self,
        state: MsGraphTeamsChannelMessageState,
    ) -> Literal["active", "deleted"]:
        if state is MsGraphTeamsChannelMessageState.ACTIVE:
            return "active"
        return "deleted"

    def _provider_root_state(
        self,
        state: Literal["active", "deleted"],
    ) -> MsGraphTeamsChannelMessageState:
        if state == "active":
            return MsGraphTeamsChannelMessageState.ACTIVE
        return MsGraphTeamsChannelMessageState.DELETED

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
        except MsGraphTeamsChannelContentTooLarge:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Channel message exceeds the configured content limit",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Teams Channel knowledge adapter configuration is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except MsGraphTeamsChannelMessageChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Channel knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Channel knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Teams Channel knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Teams Channel knowledge dependency is unavailable",
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


def _decode_scope_payload(value: str) -> _MsGraphTeamsChannelScope:
    data = _decode_canonical_payload(value)
    return _MsGraphTeamsChannelScope.model_validate(data)


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


def _resolve_message_title(
    *,
    subject: str | None,
    message_kind: Literal["root", "reply"],
) -> str:
    if subject is not None and subject.strip():
        return subject
    if message_kind == "root":
        return _ROOT_TITLE_FALLBACK
    return _REPLY_TITLE_FALLBACK


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


def register_msgraph_teams_channel_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> MsGraphTeamsChannelKnowledgeAdapter:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    registry.register(adapter)
    return adapter
