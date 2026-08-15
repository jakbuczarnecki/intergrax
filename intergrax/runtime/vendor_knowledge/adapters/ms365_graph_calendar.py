# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Calendar knowledge source adapter (MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR).

DELETED delta tombstones mean the event is no longer present in the synchronized
calendar window view. They do not claim global event deletion, mailbox-wide deletion,
or permanent Microsoft 365 removal.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_attachments import (
    MsGraphCalendarAttachment,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_content import (
    DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
    MsGraphCalendarAttendee,
    MsGraphCalendarEventChanged,
    MsGraphCalendarEventContent,
    MsGraphCalendarEventContentTooLarge,
    MsGraphCalendarLocation,
    MsGraphCalendarParticipant,
    MsGraphCalendarRecurrence,
    MsGraphCalendarRecurrencePattern,
    MsGraphCalendarRecurrenceRange,
    MsGraphCalendarResponseStatus,
    validate_msgraph_calendar_event_content,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_events import (
    MsGraphCalendarEventChange,
    MsGraphCalendarEventChangeKind,
    MsGraphCalendarEventDeltaPage,
    MsGraphCalendarEventSnapshotPage,
    MsGraphCalendarEventType,
    MsGraphCalendarViewWindow,
    validate_msgraph_calendar_event_change,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
    MsGraphCalendar,
    MsGraphCalendarReference,
    validate_msgraph_calendar_reference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    validate_msgraph_mailbox_user_id,
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

MSGRAPH_CALENDAR_SCOPE_TYPE = "msgraph_calendar"
MSGRAPH_CALENDAR_CURSOR_VERSION = "msgraph.calendar.cursor.v1"
_MSGRAPH_CALENDAR_SCOPE_SCHEMA_VERSION = "msgraph.calendar.scope.v1"
_MSGRAPH_CALENDAR_EVENT_ID_SCHEMA_VERSION = "msgraph.calendar.event-id.v1"
_MSGRAPH_CALENDAR_REVISION_SCHEMA_VERSION = "msgraph.calendar.revision.v1"

_STRUCTURED_RECORD_SCHEMA = "msgraph.calendar.event.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-calendar-event+json"
_SAFE_TITLE_FALLBACK = "Calendar event"
_REMOVAL_SEMANTICS = "removed_from_synchronized_calendar_window_view"

_MAX_CONTINUATION_URL_LEN = 32_768
_MAX_CHANGE_KEY_LEN = 4096
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_PROVIDER_PAGE_LIMIT = 200
_ATTACHMENT_INVENTORY_LIMIT = 200

_METADATA_REQUIRED_KEYS = frozenset(
    {
        "event_state",
        "event_type",
        "start_at",
        "end_at",
        "original_start_at",
        "last_modified_at",
        "series_master_id",
        "i_cal_uid",
        "is_all_day",
        "is_cancelled",
        "is_draft",
        "has_attachments",
        "is_online_meeting",
        "removal_semantics",
    }
)

_T = TypeVar("_T")


class _MsGraphCalendarScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.calendar.scope.v1"]
    mailbox_user_id: str = Field(repr=False)
    calendar_remote_id: str = Field(repr=False)
    is_default_calendar: bool
    sync_strategy: Literal["primary_delta", "snapshot"]
    window_start_at: datetime = Field(repr=False)
    window_end_at: datetime = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
            validate_msgraph_calendar_id,
        )

        return _validate_exact_durable_id(value, validator=validate_msgraph_calendar_id)

    @field_validator("is_default_calendar", mode="before")
    @classmethod
    def _validate_is_default_calendar(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("is_default_calendar must be a bool")
        return value

    @field_validator("window_start_at", "window_end_at", mode="before")
    @classmethod
    def _validate_window_datetimes(cls, value: object) -> datetime:
        return _parse_scope_window_datetime(value)

    @model_validator(mode="after")
    def _validate_strategy_invariant(self) -> _MsGraphCalendarScope:
        if self.is_default_calendar and self.sync_strategy != "primary_delta":
            raise ValueError("default calendar requires primary_delta strategy")
        if not self.is_default_calendar and self.sync_strategy != "snapshot":
            raise ValueError("non-default calendar requires snapshot strategy")
        MsGraphCalendarViewWindow(
            start_at=self.window_start_at,
            end_at=self.window_end_at,
        )
        return self


class _MsGraphCalendarEventIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.calendar.event-id.v1"]
    mailbox_user_id: str = Field(repr=False)
    calendar_remote_id: str = Field(repr=False)
    event_remote_id: str = Field(repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_mailbox_user_id)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
            validate_msgraph_calendar_id,
        )

        return _validate_exact_durable_id(value, validator=validate_msgraph_calendar_id)

    @field_validator("event_remote_id", mode="before")
    @classmethod
    def _validate_event_remote_id(cls, value: object) -> str:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
            validate_msgraph_calendar_event_id,
        )

        return _validate_exact_durable_id(value, validator=validate_msgraph_calendar_event_id)


class _MsGraphCalendarRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.calendar.revision.v1"]
    change_key: str = Field(repr=False)

    @field_validator("change_key", mode="before")
    @classmethod
    def _validate_change_key(cls, value: object) -> str:
        return _validate_opaque_change_key(value)


class _MsGraphCalendarCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["msgraph.calendar.cursor.v1"]
    mailbox_user_id: str = Field(repr=False)
    calendar_remote_id: str = Field(repr=False)
    sync_strategy: Literal["primary_delta", "snapshot"]
    window_start_at: datetime = Field(repr=False)
    window_end_at: datetime = Field(repr=False)
    phase: Literal["next_page", "delta", "complete"]
    continuation_url: str | None = Field(default=None, repr=False)

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return _validate_exact_durable_id(value, validator=validate_msgraph_mailbox_user_id)

    @field_validator("calendar_remote_id", mode="before")
    @classmethod
    def _validate_calendar_remote_id(cls, value: object) -> str:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
            validate_msgraph_calendar_id,
        )

        return _validate_exact_durable_id(value, validator=validate_msgraph_calendar_id)

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
    def _validate_phase_shape(self) -> _MsGraphCalendarCursor:
        if self.sync_strategy == "primary_delta":
            if self.phase == "complete":
                raise ValueError("primary_delta forbids complete phase")
            if self.phase in {"next_page", "delta"} and self.continuation_url is None:
                raise ValueError("continuation_url required for next_page and delta phases")
        elif self.sync_strategy == "snapshot":
            if self.phase == "delta":
                raise ValueError("snapshot forbids delta phase")
            if self.phase == "next_page" and self.continuation_url is None:
                raise ValueError("next_page phase requires continuation_url")
            if self.phase == "complete" and self.continuation_url is not None:
                raise ValueError("complete phase forbids continuation_url")
        MsGraphCalendarViewWindow(
            start_at=self.window_start_at,
            end_at=self.window_end_at,
        )
        return self


def encode_msgraph_calendar_scope_id(
    *,
    calendar: MsGraphCalendar,
    window: MsGraphCalendarViewWindow,
) -> str:
    """Return the canonical opaque remote_scope_id for one Calendar source."""
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
        validate_msgraph_calendar,
    )

    validated_calendar = validate_msgraph_calendar(calendar)
    validated_window = MsGraphCalendarViewWindow.model_validate(
        window.model_dump(mode="python")
    )
    sync_strategy: Literal["primary_delta", "snapshot"] = (
        "primary_delta" if validated_calendar.is_default_calendar else "snapshot"
    )
    scope = _MsGraphCalendarScope(
        schema_version=_MSGRAPH_CALENDAR_SCOPE_SCHEMA_VERSION,
        mailbox_user_id=validated_calendar.mailbox_user_id,
        calendar_remote_id=validated_calendar.remote_id,
        is_default_calendar=validated_calendar.is_default_calendar,
        sync_strategy=sync_strategy,
        window_start_at=validated_window.start_at,
        window_end_at=validated_window.end_at,
    )
    return _encode_canonical_payload(scope.model_dump(mode="json"))


class MsGraphCalendarKnowledgeAdapter:
    """Thin mapping from Microsoft Graph Calendar integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return MSGRAPH_CALENDAR_SOURCE_KIND

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
        validated_source = self._validate_source_ref(source)
        sync_strategy = self._decode_scope_strategy(validated_source)
        return KnowledgeScopeInfo(
            source=validated_source,
            capabilities=self._capabilities_for_strategy(sync_strategy),
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
        scope = self._decode_scope_payload(validated_source.scope.remote_scope_id)
        window = MsGraphCalendarViewWindow(
            start_at=scope.window_start_at,
            end_at=scope.window_end_at,
        )
        calendar_reference = validate_msgraph_calendar_reference(
            MsGraphCalendarReference(
                mailbox_user_id=scope.mailbox_user_id,
                calendar_remote_id=scope.calendar_remote_id,
                is_default_calendar=scope.is_default_calendar,
            )
        )
        provider_limit = min(self._validate_limit(limit), _PROVIDER_PAGE_LIMIT)
        decoded_cursor = self._decode_cursor(
            cursor,
            mailbox_user_id=scope.mailbox_user_id,
            calendar_remote_id=scope.calendar_remote_id,
            sync_strategy=scope.sync_strategy,
            window_start_at=window.start_at,
            window_end_at=window.end_at,
        )
        if scope.sync_strategy == "primary_delta":
            return await self._read_primary_delta_page(
                graph_integration=graph_integration,
                calendar_reference=calendar_reference,
                window=window,
                decoded_cursor=decoded_cursor,
                provider_limit=provider_limit,
            )
        return await self._read_snapshot_page(
            graph_integration=graph_integration,
            calendar_reference=calendar_reference,
            window=window,
            decoded_cursor=decoded_cursor,
            provider_limit=provider_limit,
        )

    async def _read_primary_delta_page(
        self,
        *,
        graph_integration: Ms365GraphCollaborationSuiteIntegration,
        calendar_reference: MsGraphCalendarReference,
        window: MsGraphCalendarViewWindow,
        decoded_cursor: _MsGraphCalendarCursor | None,
        provider_limit: int,
    ) -> KnowledgePage:
        provider_continuation = self._to_provider_continuation(decoded_cursor)
        page = await self._invoke_integration(
            lambda: graph_integration.read_calendar_events_delta_page_by_reference(
                calendar=calendar_reference,
                window=window,
                continuation=provider_continuation,
                limit=provider_limit,
            )
        )
        try:
            self._validate_delta_page_for_source(
                page,
                mailbox_user_id=calendar_reference.mailbox_user_id,
                calendar_remote_id=calendar_reference.calendar_remote_id,
                window=window,
                provider_limit=provider_limit,
            )
            changes = tuple(
                self._event_change_to_knowledge_change(
                    item,
                    mailbox_user_id=calendar_reference.mailbox_user_id,
                    calendar_remote_id=calendar_reference.calendar_remote_id,
                )
                for item in page.items
            )
            encoded_checkpoint = self._encode_cursor_from_continuation(
                mailbox_user_id=calendar_reference.mailbox_user_id,
                calendar_remote_id=calendar_reference.calendar_remote_id,
                sync_strategy="primary_delta",
                window_start_at=window.start_at,
                window_end_at=window.end_at,
                continuation=page.continuation,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Calendar knowledge provider response is invalid",
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

    async def _read_snapshot_page(
        self,
        *,
        graph_integration: Ms365GraphCollaborationSuiteIntegration,
        calendar_reference: MsGraphCalendarReference,
        window: MsGraphCalendarViewWindow,
        decoded_cursor: _MsGraphCalendarCursor | None,
        provider_limit: int,
    ) -> KnowledgePage:
        if decoded_cursor is not None and decoded_cursor.phase == "complete":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Calendar reconciliation cursor is complete; "
                    "restart reconciliation"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        provider_continuation: MsGraphKnowledgeContinuation | None = None
        if decoded_cursor is not None and decoded_cursor.phase == "next_page":
            provider_continuation = MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=decoded_cursor.continuation_url,
            )
        page = await self._invoke_integration(
            lambda: graph_integration.read_calendar_events_snapshot_page_by_reference(
                calendar=calendar_reference,
                window=window,
                continuation=provider_continuation,
                limit=provider_limit,
            )
        )
        try:
            validated_page = self._validate_snapshot_page_for_source(
                page,
                mailbox_user_id=calendar_reference.mailbox_user_id,
                calendar_remote_id=calendar_reference.calendar_remote_id,
                window=window,
                provider_limit=provider_limit,
            )
            changes = tuple(
                self._event_change_to_knowledge_change(
                    item,
                    mailbox_user_id=calendar_reference.mailbox_user_id,
                    calendar_remote_id=calendar_reference.calendar_remote_id,
                )
                for item in validated_page.items
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Calendar knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if validated_page.continuation is not None:
            next_cursor = self._encode_cursor(
                _MsGraphCalendarCursor(
                    schema_version=MSGRAPH_CALENDAR_CURSOR_VERSION,
                    mailbox_user_id=calendar_reference.mailbox_user_id,
                    calendar_remote_id=calendar_reference.calendar_remote_id,
                    sync_strategy="snapshot",
                    window_start_at=window.start_at,
                    window_end_at=window.end_at,
                    phase="next_page",
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
            mailbox_user_id=calendar_reference.mailbox_user_id,
            calendar_remote_id=calendar_reference.calendar_remote_id,
            sync_strategy="snapshot",
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
        scope = self._decode_scope_payload(validated_source.scope.remote_scope_id)
        try:
            validated_item = self._deep_validate_item_descriptor(item)
            event_identity, revision = self._validate_event_item(
                validated_item,
                source=validated_source,
                mailbox_user_id=scope.mailbox_user_id,
                calendar_remote_id=scope.calendar_remote_id,
            )
            metadata = self._validate_descriptor_metadata(
                validated_item.metadata,
                updated_at=validated_item.revision.updated_at,
            )
            provider_event = self._descriptor_to_provider_event(
                event_identity=event_identity,
                revision=revision,
                metadata=metadata,
                updated_at=validated_item.revision.updated_at,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        result = await self._invoke_integration(
            lambda: graph_integration.read_calendar_event_content(
                event=provider_event,
                max_chars=DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
            ),
        )
        try:
            validated_content = validate_msgraph_calendar_event_content(
                result,
                event=provider_event,
                max_chars=DEFAULT_CALENDAR_EVENT_CONTENT_MAX_CHARS,
            )
            self._validate_fetched_content_identity(
                validated_content,
                event_identity=event_identity,
                revision=revision,
            )
            attachment_items: list[dict[str, object]] = []
            if validated_content.has_attachments:
                attachment_page = await self._invoke_integration(
                    lambda: graph_integration.read_calendar_attachments_page(
                        event=provider_event,
                        continuation=None,
                        limit=_ATTACHMENT_INVENTORY_LIMIT,
                    )
                )
                if attachment_page.continuation is not None:
                    raise VendorKnowledgeError(
                        code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                        safe_message=(
                            "Microsoft Graph Calendar attachment inventory exceeds "
                            "the supported limit"
                        ),
                        provider_id=self.provider_id,
                        source_kind=self.source_kind,
                        retryable=False,
                    )
                sorted_attachments = sorted(
                    attachment_page.items,
                    key=lambda attachment: attachment.remote_id,
                )
                attachment_items = [
                    _attachment_inventory_record(item) for item in sorted_attachments
                ]
        except MsGraphCalendarEventChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Calendar knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Calendar knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        try:
            structured_record = self._build_structured_record(
                content=validated_content,
                attachment_items=attachment_items,
            )
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
                safe_message="Microsoft Graph Calendar knowledge provider response is invalid",
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
        try:
            scope = self._decode_scope_payload(validated_source.scope.remote_scope_id)
            validated_item = self._deep_validate_item_descriptor(item)
            _event_identity, _revision = self._validate_event_item(
                validated_item,
                source=validated_source,
                mailbox_user_id=scope.mailbox_user_id,
                calendar_remote_id=scope.calendar_remote_id,
            )
            self._validate_descriptor_metadata(
                validated_item.metadata,
                updated_at=validated_item.revision.updated_at,
            )
        except VendorKnowledgeError:
            raise
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=(
                "Microsoft Graph Calendar authoritative permission projection "
                "is not implemented"
            ),
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _capabilities_for_strategy(
        self,
        sync_strategy: Literal["primary_delta", "snapshot"],
    ) -> KnowledgeAdapterCapabilities:
        base = self.capabilities
        if sync_strategy == "primary_delta":
            return KnowledgeAdapterCapabilities(
                full_inventory=True,
                incremental_changes=True,
                content_fetch=base.content_fetch,
                binary_content=base.binary_content,
                rich_text_content=base.rich_text_content,
                structured_content=base.structured_content,
                permissions=base.permissions,
                tombstones=True,
                remote_versions=base.remote_versions,
                reconciliation=True,
            )
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
            content_fetch=base.content_fetch,
            binary_content=base.binary_content,
            rich_text_content=base.rich_text_content,
            structured_content=base.structured_content,
            permissions=base.permissions,
            tombstones=False,
            remote_versions=base.remote_versions,
            reconciliation=True,
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
                    "Microsoft Graph Calendar knowledge adapter requires "
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
        if scope.remote_scope_type != MSGRAPH_CALENDAR_SCOPE_TYPE:
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
            decoded = self._decode_scope_payload(scope.remote_scope_id)
            canonical = _encode_canonical_payload(decoded.model_dump(mode="json"))
            if canonical != scope.remote_scope_id:
                raise ValueError("scope id is not canonical")
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise self._invalid_source_scope_error(
                provider_id=validated_source.provider_id,
                source_kind=validated_source.source_kind,
            ) from None
        return validated_source

    def _decode_scope_strategy(self, source: KnowledgeSourceRef) -> Literal["primary_delta", "snapshot"]:
        return self._decode_scope_payload(source.scope.remote_scope_id).sync_strategy

    def _decode_scope_payload(self, remote_scope_id: str) -> _MsGraphCalendarScope:
        try:
            data = _decode_canonical_payload(remote_scope_id)
            return _MsGraphCalendarScope.model_validate(data)
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise ValueError("scope payload is invalid") from None

    def _invalid_source_scope_error(
        self,
        *,
        provider_id: str | None = None,
        source_kind: str | None = None,
    ) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message="Microsoft Graph Calendar knowledge source scope is invalid",
            provider_id=provider_id or self.provider_id,
            source_kind=source_kind or self.source_kind,
            retryable=False,
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Calendar knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if limit < 1 or limit > 1000:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Calendar knowledge page limit is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return limit

    def _validate_delta_page_for_source(
        self,
        page: object,
        *,
        mailbox_user_id: str,
        calendar_remote_id: str,
        window: MsGraphCalendarViewWindow,
        provider_limit: int,
    ) -> None:
        if not isinstance(page, MsGraphCalendarEventDeltaPage):
            raise ValueError("page must be a MsGraphCalendarEventDeltaPage")
        if page.mailbox_user_id != mailbox_user_id or page.calendar_remote_id != calendar_remote_id:
            raise ValueError("page scope does not match source")
        if page.window.start_at != window.start_at or page.window.end_at != window.end_at:
            raise ValueError("page window does not match source")
        if page.continuation.kind not in {
            MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            MsGraphKnowledgeContinuationKind.DELTA,
        }:
            raise ValueError("invalid continuation kind")
        if len(page.items) > provider_limit:
            raise ValueError("page items exceed provider limit")
        seen_remote_ids: set[str] = set()
        for item in page.items:
            validated = validate_msgraph_calendar_event_change(item)
            if (
                validated.mailbox_user_id != mailbox_user_id
                or validated.calendar_remote_id != calendar_remote_id
            ):
                raise ValueError("item scope does not match source")
            opaque_id = self._encode_event_identity(
                mailbox_user_id=mailbox_user_id,
                calendar_remote_id=calendar_remote_id,
                event_remote_id=validated.remote_id,
            )
            if opaque_id in seen_remote_ids:
                raise ValueError("duplicate neutral id on page")
            seen_remote_ids.add(opaque_id)

    def _validate_snapshot_page_for_source(
        self,
        page: object,
        *,
        mailbox_user_id: str,
        calendar_remote_id: str,
        window: MsGraphCalendarViewWindow,
        provider_limit: int,
    ) -> MsGraphCalendarEventSnapshotPage:
        if not isinstance(page, MsGraphCalendarEventSnapshotPage):
            raise ValueError("page must be a MsGraphCalendarEventSnapshotPage")
        if page.mailbox_user_id != mailbox_user_id or page.calendar_remote_id != calendar_remote_id:
            raise ValueError("page scope does not match source")
        if page.window.start_at != window.start_at or page.window.end_at != window.end_at:
            raise ValueError("page window does not match source")
        if len(page.items) > provider_limit:
            raise ValueError("page items exceed provider limit")
        seen_remote_ids: set[str] = set()
        validated_items: list[MsGraphCalendarEventChange] = []
        for item in page.items:
            validated = validate_msgraph_calendar_event_change(item)
            if (
                validated.mailbox_user_id != mailbox_user_id
                or validated.calendar_remote_id != calendar_remote_id
            ):
                raise ValueError("item scope does not match source")
            if validated.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
                raise ValueError("snapshot item must be active")
            opaque_id = self._encode_event_identity(
                mailbox_user_id=mailbox_user_id,
                calendar_remote_id=calendar_remote_id,
                event_remote_id=validated.remote_id,
            )
            if opaque_id in seen_remote_ids:
                raise ValueError("duplicate neutral id on page")
            seen_remote_ids.add(opaque_id)
            validated_items.append(validated)
        validated_continuation: MsGraphKnowledgeContinuation | None = None
        if page.continuation is not None:
            validated_continuation = MsGraphKnowledgeContinuation.model_validate(
                page.continuation.model_dump(mode="python")
            )
        return MsGraphCalendarEventSnapshotPage(
            mailbox_user_id=mailbox_user_id,
            calendar_remote_id=calendar_remote_id,
            window=window,
            items=tuple(validated_items),
            continuation=validated_continuation,
        )

    def _event_change_to_knowledge_change(
        self,
        item: MsGraphCalendarEventChange,
        *,
        mailbox_user_id: str,
        calendar_remote_id: str,
    ) -> KnowledgeChange:
        opaque_id = self._encode_event_identity(
            mailbox_user_id=mailbox_user_id,
            calendar_remote_id=calendar_remote_id,
            event_remote_id=item.remote_id,
        )
        if item.kind == MsGraphCalendarEventChangeKind.REMOVED:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=opaque_id,
                descriptor=None,
            )
        descriptor = self._active_event_to_descriptor(
            item,
            opaque_remote_id=opaque_id,
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=opaque_id,
            descriptor=descriptor,
        )

    def _active_event_to_descriptor(
        self,
        event: MsGraphCalendarEventChange,
        *,
        opaque_remote_id: str,
    ) -> KnowledgeItemDescriptor:
        if event.kind is not MsGraphCalendarEventChangeKind.ACTIVE:
            raise ValueError("event is not active")
        if event.change_key is None or event.last_modified_at is None:
            raise ValueError("active event missing revision")
        if event.event_type is None or event.start_at is None or event.end_at is None:
            raise ValueError("active event missing required fields")
        if event.is_all_day is None or event.is_cancelled is None or event.is_draft is None:
            raise ValueError("active event missing flags")
        if event.has_attachments is None or event.is_online_meeting is None:
            raise ValueError("active event missing attachment flags")
        metadata: dict[str, object] = {
            "event_state": "active",
            "event_type": event.event_type.value,
            "start_at": event.start_at.isoformat(),
            "end_at": event.end_at.isoformat(),
            "original_start_at": (
                event.original_start_at.isoformat() if event.original_start_at is not None else None
            ),
            "last_modified_at": event.last_modified_at.isoformat(),
            "series_master_id": event.series_master_id,
            "i_cal_uid": event.i_cal_uid,
            "is_all_day": event.is_all_day,
            "is_cancelled": event.is_cancelled,
            "is_draft": event.is_draft,
            "has_attachments": event.has_attachments,
            "is_online_meeting": event.is_online_meeting,
            "removal_semantics": _REMOVAL_SEMANTICS,
        }
        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=opaque_remote_id,
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=self._encode_revision(event.change_key),
                etag=None,
                updated_at=event.last_modified_at,
            ),
            title=_SAFE_TITLE_FALLBACK,
            item_type="msgraph_calendar_event",
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

    def _deep_validate_item_descriptor(self, item: object) -> KnowledgeItemDescriptor:
        if not isinstance(item, KnowledgeItemDescriptor):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        if type(item.content_mode) is not KnowledgeContentMode:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            return KnowledgeItemDescriptor.model_validate(item.model_dump(mode="python"))
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _validate_event_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        mailbox_user_id: str,
        calendar_remote_id: str,
    ) -> tuple[_MsGraphCalendarEventIdentity, _MsGraphCalendarRevision]:
        self._validate_item_provenance(item, source=source)
        if item.item_type != "msgraph_calendar_event":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        event_identity = self._decode_event_identity(item.identity.remote_id)
        if (
            event_identity.mailbox_user_id != mailbox_user_id
            or event_identity.calendar_remote_id != calendar_remote_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        version = item.revision.version
        if not isinstance(version, str) or not version.strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.revision.updated_at is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
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
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        revision = self._decode_revision(version)
        return event_identity, revision

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
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

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
        if metadata["event_state"] != "active":
            raise ValueError("event_state must be active")
        event_type_raw = metadata["event_type"]
        if not isinstance(event_type_raw, str):
            raise ValueError("event_type must be a string")
        try:
            MsGraphCalendarEventType(event_type_raw)
        except ValueError:
            raise ValueError("event_type is invalid") from None
        for bool_key in (
            "is_all_day",
            "is_cancelled",
            "is_draft",
            "has_attachments",
            "is_online_meeting",
        ):
            if type(metadata[bool_key]) is not bool:
                raise ValueError(f"{bool_key} must be bool")
        if metadata["removal_semantics"] != _REMOVAL_SEMANTICS:
            raise ValueError("removal_semantics is invalid")
        start_at = self._parse_timezone_aware_iso(metadata["start_at"])
        end_at = self._parse_timezone_aware_iso(metadata["end_at"])
        last_modified_at = self._parse_timezone_aware_iso(metadata["last_modified_at"])
        if updated_at is not None and last_modified_at != updated_at:
            raise ValueError("last_modified_at mismatch")
        original_start_raw = metadata["original_start_at"]
        if original_start_raw is not None:
            self._parse_timezone_aware_iso(original_start_raw)
        series_master_id = metadata["series_master_id"]
        if series_master_id is not None and not isinstance(series_master_id, str):
            raise ValueError("series_master_id must be string or null")
        i_cal_uid = metadata["i_cal_uid"]
        if i_cal_uid is not None and not isinstance(i_cal_uid, str):
            raise ValueError("i_cal_uid must be string or null")
        if end_at <= start_at:
            raise ValueError("end_at must be after start_at")
        return metadata

    def _descriptor_to_provider_event(
        self,
        *,
        event_identity: _MsGraphCalendarEventIdentity,
        revision: _MsGraphCalendarRevision,
        metadata: dict[str, object],
        updated_at: datetime | None,
    ) -> MsGraphCalendarEventChange:
        if updated_at is None:
            raise ValueError("updated_at is required")
        event_type = MsGraphCalendarEventType(str(metadata["event_type"]))
        start_at = self._parse_timezone_aware_iso(metadata["start_at"])
        end_at = self._parse_timezone_aware_iso(metadata["end_at"])
        last_modified_at = self._parse_timezone_aware_iso(metadata["last_modified_at"])
        if last_modified_at != updated_at:
            raise ValueError("last_modified_at mismatch")
        original_start_raw = metadata["original_start_at"]
        original_start_at = (
            self._parse_timezone_aware_iso(original_start_raw)
            if original_start_raw is not None
            else None
        )
        return MsGraphCalendarEventChange(
            mailbox_user_id=event_identity.mailbox_user_id,
            calendar_remote_id=event_identity.calendar_remote_id,
            remote_id=event_identity.event_remote_id,
            kind=MsGraphCalendarEventChangeKind.ACTIVE,
            change_key=revision.change_key,
            event_type=event_type,
            start_at=start_at,
            end_at=end_at,
            original_start_at=original_start_at,
            last_modified_at=last_modified_at,
            series_master_id=metadata["series_master_id"],
            i_cal_uid=metadata["i_cal_uid"],
            is_all_day=metadata["is_all_day"],
            is_cancelled=metadata["is_cancelled"],
            is_draft=metadata["is_draft"],
            has_attachments=metadata["has_attachments"],
            is_online_meeting=metadata["is_online_meeting"],
        )

    def _validate_fetched_content_identity(
        self,
        content: MsGraphCalendarEventContent,
        *,
        event_identity: _MsGraphCalendarEventIdentity,
        revision: _MsGraphCalendarRevision,
    ) -> None:
        if content.mailbox_user_id != event_identity.mailbox_user_id:
            raise ValueError("mailbox mismatch")
        if content.calendar_remote_id != event_identity.calendar_remote_id:
            raise ValueError("calendar mismatch")
        if content.remote_id != event_identity.event_remote_id:
            raise ValueError("event id mismatch")
        if content.content_revision != revision.change_key:
            raise ValueError("revision mismatch")

    def _build_structured_record(
        self,
        *,
        content: MsGraphCalendarEventContent,
        attachment_items: list[dict[str, object]],
    ) -> JsonObject:
        return {
            "schema": _STRUCTURED_RECORD_SCHEMA,
            "event_type": content.event_type.value,
            "subject": content.subject,
            "body": {
                "kind": content.body_kind.value,
                "content": content.body_content,
                "preview": content.body_preview,
            },
            "start_at": content.start_at.isoformat(),
            "end_at": content.end_at.isoformat(),
            "original_start_at": (
                content.original_start_at.isoformat() if content.original_start_at is not None else None
            ),
            "original_start_time_zone": content.original_start_time_zone,
            "original_end_time_zone": content.original_end_time_zone,
            "created_at": content.created_at.isoformat(),
            "last_modified_at": content.last_modified_at.isoformat(),
            "organizer": _participant_to_record(content.organizer),
            "attendees": [_attendee_to_record(item) for item in content.attendees],
            "location": _location_to_record(content.location),
            "locations": [_location_to_record(item) for item in content.locations],
            "recurrence": _recurrence_to_record(content.recurrence),
            "series_master_id": content.series_master_id,
            "cancelled_occurrence_ids": list(content.cancelled_occurrence_ids),
            "categories": list(content.categories),
            "i_cal_uid": content.i_cal_uid,
            "importance": content.importance.value,
            "sensitivity": content.sensitivity.value,
            "show_as": content.show_as.value,
            "response_status": _response_status_to_record(content.response_status),
            "is_all_day": content.is_all_day,
            "is_cancelled": content.is_cancelled,
            "is_draft": content.is_draft,
            "is_organizer": content.is_organizer,
            "is_online_meeting": content.is_online_meeting,
            "online_meeting_provider": content.online_meeting_provider.value,
            "has_attachments": content.has_attachments,
            "hide_attendees": content.hide_attendees,
            "allow_new_time_proposals": content.allow_new_time_proposals,
            "response_requested": content.response_requested,
            "is_reminder_on": content.is_reminder_on,
            "reminder_minutes_before_start": content.reminder_minutes_before_start,
            "attachments": {
                "attachment_inventory_included": content.has_attachments,
                "attachment_inventory_complete": content.has_attachments,
                "attachment_binary_content_included": False,
                "items": attachment_items,
            },
        }

    def _encode_event_identity(
        self,
        *,
        mailbox_user_id: str,
        calendar_remote_id: str,
        event_remote_id: str,
    ) -> str:
        identity = _MsGraphCalendarEventIdentity(
            schema_version=_MSGRAPH_CALENDAR_EVENT_ID_SCHEMA_VERSION,
            mailbox_user_id=mailbox_user_id,
            calendar_remote_id=calendar_remote_id,
            event_remote_id=event_remote_id,
        )
        return _encode_canonical_payload(identity.model_dump())

    def _decode_event_identity(self, value: str) -> _MsGraphCalendarEventIdentity:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphCalendarEventIdentity.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_revision(self, change_key: str) -> str:
        revision = _MsGraphCalendarRevision(
            schema_version=_MSGRAPH_CALENDAR_REVISION_SCHEMA_VERSION,
            change_key=change_key,
        )
        return _encode_canonical_payload(revision.model_dump())

    def _decode_revision(self, value: str) -> _MsGraphCalendarRevision:
        try:
            data = _decode_canonical_payload(value)
            return _MsGraphCalendarRevision.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Microsoft Graph Calendar event descriptor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None

    def _encode_cursor(self, cursor: _MsGraphCalendarCursor) -> KnowledgeCursor:
        return KnowledgeCursor(
            value=_encode_canonical_payload(cursor.model_dump(mode="json")),
            version=MSGRAPH_CALENDAR_CURSOR_VERSION,
        )

    def _encode_cursor_from_continuation(
        self,
        *,
        mailbox_user_id: str,
        calendar_remote_id: str,
        sync_strategy: Literal["primary_delta", "snapshot"],
        window_start_at: datetime,
        window_end_at: datetime,
        continuation: MsGraphKnowledgeContinuation,
    ) -> KnowledgeCursor:
        if continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            phase: Literal["next_page", "delta", "complete"] = "next_page"
        else:
            phase = "delta"
        return self._encode_cursor(
            _MsGraphCalendarCursor(
                schema_version=MSGRAPH_CALENDAR_CURSOR_VERSION,
                mailbox_user_id=mailbox_user_id,
                calendar_remote_id=calendar_remote_id,
                sync_strategy=sync_strategy,
                window_start_at=window_start_at,
                window_end_at=window_end_at,
                phase=phase,
                continuation_url=continuation.url,
            )
        )

    def _encode_complete_cursor(
        self,
        *,
        mailbox_user_id: str,
        calendar_remote_id: str,
        sync_strategy: Literal["primary_delta", "snapshot"],
        window_start_at: datetime,
        window_end_at: datetime,
    ) -> KnowledgeCursor:
        return self._encode_cursor(
            _MsGraphCalendarCursor(
                schema_version=MSGRAPH_CALENDAR_CURSOR_VERSION,
                mailbox_user_id=mailbox_user_id,
                calendar_remote_id=calendar_remote_id,
                sync_strategy=sync_strategy,
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
        calendar_remote_id: str,
        sync_strategy: Literal["primary_delta", "snapshot"],
        window_start_at: datetime,
        window_end_at: datetime,
    ) -> _MsGraphCalendarCursor | None:
        if cursor is None:
            return None
        if cursor.version != MSGRAPH_CALENDAR_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Calendar knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            data = _decode_canonical_payload(cursor.value)
            decoded = _MsGraphCalendarCursor.model_validate(data)
            canonical = _encode_canonical_payload(decoded.model_dump(mode="json"))
            if canonical != cursor.value:
                raise ValueError("cursor value is not canonical")
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Microsoft Graph Calendar knowledge cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if (
            decoded.mailbox_user_id != mailbox_user_id
            or decoded.calendar_remote_id != calendar_remote_id
            or decoded.sync_strategy != sync_strategy
            or decoded.window_start_at != window_start_at
            or decoded.window_end_at != window_end_at
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Microsoft Graph Calendar knowledge cursor scope does not match source"
                ),
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded

    def _to_provider_continuation(
        self,
        decoded: _MsGraphCalendarCursor | None,
    ) -> MsGraphKnowledgeContinuation | None:
        if decoded is None:
            return None
        if decoded.phase == "next_page":
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
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except MsGraphCalendarEventContentTooLarge:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Calendar event exceeds the configured content limit",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except IntegrationConfigurationError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Microsoft Graph Calendar knowledge adapter configuration is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except MsGraphCalendarEventChanged:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Calendar knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Calendar knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Microsoft Graph Calendar knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Microsoft Graph Calendar knowledge dependency is unavailable",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None


def _validate_exact_durable_id(
    value: object,
    *,
    validator: Callable[[object], str],
) -> str:
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


def _parse_scope_window_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("window datetime must be timezone-aware")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        raise ValueError("window datetime must be a datetime")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("window datetime must not be empty")
    normalized = cleaned.replace("Z", "+00:00") if cleaned.endswith("Z") else cleaned
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError("window datetime must be valid ISO-8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("window datetime must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _encode_canonical_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_canonical_payload(value: str) -> dict[str, object]:
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(value + padding)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    return data


def _participant_to_record(
    participant: MsGraphCalendarParticipant | None,
) -> dict[str, object] | None:
    if participant is None:
        return None
    return {
        "display_name": participant.display_name,
        "address": participant.address,
    }


def _response_status_to_record(status: MsGraphCalendarResponseStatus) -> dict[str, object]:
    return {
        "response": status.response.value,
        "responded_at": (
            status.responded_at.isoformat() if status.responded_at is not None else None
        ),
    }


def _attendee_to_record(attendee: MsGraphCalendarAttendee) -> dict[str, object]:
    return {
        "participant": _participant_to_record(attendee.participant),
        "attendee_type": attendee.attendee_type.value,
        "status": _response_status_to_record(attendee.status),
    }


def _location_to_record(location: MsGraphCalendarLocation | None) -> dict[str, object] | None:
    if location is None:
        return None
    return {
        "display_name": location.display_name,
        "location_type": location.location_type.value,
        "street": location.street,
        "city": location.city,
        "state": location.state,
        "country_or_region": location.country_or_region,
        "postal_code": location.postal_code,
    }


def _recurrence_pattern_to_record(
    pattern: MsGraphCalendarRecurrencePattern,
) -> dict[str, object]:
    return {
        "pattern_type": pattern.pattern_type.value,
        "interval": pattern.interval,
        "month": pattern.month,
        "day_of_month": pattern.day_of_month,
        "days_of_week": [item.value for item in pattern.days_of_week],
        "first_day_of_week": (
            pattern.first_day_of_week.value if pattern.first_day_of_week is not None else None
        ),
        "index": pattern.index.value if pattern.index is not None else None,
    }


def _recurrence_range_to_record(range_model: MsGraphCalendarRecurrenceRange) -> dict[str, object]:
    return {
        "range_type": range_model.range_type.value,
        "start_date": range_model.start_date.isoformat(),
        "end_date": range_model.end_date.isoformat() if range_model.end_date is not None else None,
        "number_of_occurrences": range_model.number_of_occurrences,
        "recurrence_time_zone": range_model.recurrence_time_zone,
    }


def _recurrence_to_record(recurrence: MsGraphCalendarRecurrence | None) -> dict[str, object] | None:
    if recurrence is None:
        return None
    return {
        "pattern": _recurrence_pattern_to_record(recurrence.pattern),
        "range": _recurrence_range_to_record(recurrence.range),
    }


def _attachment_inventory_record(attachment: MsGraphCalendarAttachment) -> dict[str, object]:
    return {
        "attachment_remote_id": attachment.remote_id,
        "kind": attachment.kind.value,
        "name": attachment.name,
        "content_type": attachment.content_type,
        "size_bytes": attachment.size_bytes,
        "is_inline": attachment.is_inline,
        "last_modified_at": attachment.last_modified_at.isoformat(),
    }


def register_msgraph_calendar_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> MsGraphCalendarKnowledgeAdapter:
    adapter = MsGraphCalendarKnowledgeAdapter()
    registry.register(adapter)
    return adapter
