# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace Calendar Vendor Knowledge adapter."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
    GoogleCalendarAttendee,
    GoogleCalendarConferenceData,
    GoogleCalendarConferenceEntryPoint,
    GoogleCalendarConferenceSolution,
    GoogleCalendarEvent,
    GoogleCalendarEventDateTime,
    GoogleCalendarEventPage,
    GoogleCalendarEventStatus,
    GoogleCalendarEventType,
    GoogleCalendarPerson,
    GoogleCalendarReminder,
    GoogleCalendarReminders,
    GoogleCalendarSyncToken,
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

GOOGLE_CALENDAR_SCOPE_TYPE = "google_workspace_calendar"
GOOGLE_CALENDAR_CURSOR_VERSION = "google_workspace.calendar.cursor.v1"
GOOGLE_CALENDAR_ITEM_METADATA_VERSION = "google_workspace.calendar.item.v1"
GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA = (
    "google_workspace.calendar.event.knowledge.v1"
)
GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE = (
    "application/vnd.intergrax.google-workspace-calendar-event+json"
)

_GOOGLE_CALENDAR_EVENT_ITEM_TYPE = "google_workspace_calendar_event"
_INVALID_SCOPE_MESSAGE = "Google Workspace Calendar knowledge source scope is invalid"
_INVALID_CURSOR_MESSAGE = "Google Workspace Calendar knowledge cursor is invalid"
_INVALID_PROVIDER_RESPONSE_MESSAGE = (
    "Google Workspace Calendar knowledge provider response is invalid"
)
_PROVIDER_INVALID_REQUEST_MESSAGE = (
    "Google Workspace Calendar knowledge provider rejected the request"
)
_INVALID_DESCRIPTOR_MESSAGE = "Google Workspace Calendar event descriptor is invalid"
_CONFIGURATION_ERROR_MESSAGE = "Google Workspace Calendar knowledge page limit is invalid"
_RECONCILIATION_REQUIRED_MESSAGE = (
    "Google Workspace Calendar synchronization requires full reconciliation"
)
_DEPENDENCY_UNAVAILABLE_MESSAGE = (
    "Google Workspace Calendar knowledge dependency is unavailable"
)
_CONTENT_CHANGED_MESSAGE = (
    "Google Workspace Calendar event content changed since descriptor creation"
)
_UNSUPPORTED_PERMISSIONS_MESSAGE = (
    "Authoritative Google Calendar permissions projection is not implemented"
)
_INTEGRATION_REQUIRED_MESSAGE = (
    "Google Workspace Calendar knowledge adapter requires "
    "Google Workspace collaboration-suite integration"
)

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")
_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)$")
_CURSOR_ALPHABET = re.compile(r"^[A-Za-z0-9_-]+$")
_MAX_CALENDAR_ID_LENGTH = 1024
_MAX_EVENT_ID_LENGTH = 1024
_MAX_TOKEN_LENGTH = 4096
_MAX_ENCODED_CURSOR_LENGTH = 24_576
_MAX_TEXT_LENGTH = 16_384
_PROVIDER_PAGE_LIMIT = 2500

_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "structured_record_schema",
        "calendar_id_hash",
        "status",
        "event_type",
        "recurring_event",
        "all_day",
    }
)

_T = TypeVar("_T")


def _validate_cursor_token(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_TOKEN_LENGTH
        or _ASCII_CONTROL.search(value)
    ):
        raise ValueError("invalid cursor token")
    return value


def _validate_event_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_EVENT_ID_LENGTH
        or _ASCII_CONTROL.search(value)
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("invalid event id")
    return value


class _GoogleCalendarCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["google_workspace.calendar.cursor.v1"]
    scope_fingerprint: str = Field(repr=False)
    phase: Literal["inventory", "changes"]
    page_token: str | None = Field(default=None, repr=False)
    sync_token: str | None = Field(default=None, repr=False)

    @field_validator("scope_fingerprint", mode="before")
    @classmethod
    def _validate_fingerprint(cls, value: object) -> str:
        if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
            raise ValueError("invalid scope fingerprint")
        return value

    @field_validator("page_token", "sync_token", mode="before")
    @classmethod
    def _validate_token(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_cursor_token(value)

    @model_validator(mode="after")
    def _validate_invariants(self) -> _GoogleCalendarCursor:
        if self.phase == "inventory":
            if self.page_token is None or self.sync_token is not None:
                raise ValueError("invalid inventory cursor")
        elif self.sync_token is None:
            raise ValueError("invalid changes cursor")
        return self


class GoogleWorkspaceCalendarKnowledgeAdapter:
    """Maps one Google Calendar event to one structured knowledge item."""

    @property
    def provider_id(self) -> str:
        return GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.COLLABORATION_SUITE

    @property
    def source_kind(self) -> str:
        return GOOGLE_CALENDAR_SOURCE_KIND

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
        self._require_google_integration(integration)
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
        google_integration = self._require_google_integration(integration)
        validated_source = self._validate_source(source)
        calendar_id = validated_source.scope.remote_scope_id
        provider_limit = self._validate_limit(limit)
        decoded = self._decode_cursor(cursor, calendar_id=calendar_id)

        if decoded is None:
            page = await self._read_provider_page(
                google_integration,
                calendar_id=calendar_id,
                page_token=None,
                sync_token=None,
                limit=provider_limit,
            )
            phase = "inventory"
        else:
            page = await self._read_provider_page(
                google_integration,
                calendar_id=calendar_id,
                page_token=(
                    GoogleWorkspacePageToken(value=decoded.page_token)
                    if decoded.page_token is not None
                    else None
                ),
                sync_token=(
                    GoogleCalendarSyncToken(value=decoded.sync_token)
                    if decoded.sync_token is not None
                    else None
                ),
                limit=provider_limit,
                reconciliation_required_on_expiry=(
                    decoded is not None
                    and decoded.phase == "changes"
                    and decoded.sync_token is not None
                ),
            )
            phase = decoded.phase

        validated_page = self._reconstruct_page(page, calendar_id=calendar_id)
        changes = tuple(self._event_to_change(event, calendar_id=calendar_id) for event in validated_page.events)

        if phase == "inventory":
            if validated_page.next_page_token is not None:
                continuation_cursor = self._encode_cursor(
                    self._cursor(
                        calendar_id=calendar_id,
                        phase="inventory",
                        page_token=validated_page.next_page_token.value,
                        sync_token=None,
                    )
                )
                return KnowledgePage(
                    changes=changes,
                    next_cursor=continuation_cursor,
                    proposed_checkpoint=continuation_cursor,
                    has_more=True,
                )
            if validated_page.next_sync_token is None:
                raise self._invalid_provider_response_error()
            return KnowledgePage(
                changes=changes,
                next_cursor=None,
                proposed_checkpoint=self._encode_cursor(
                    self._cursor(
                        calendar_id=calendar_id,
                        phase="changes",
                        page_token=None,
                        sync_token=validated_page.next_sync_token.value,
                    )
                ),
                has_more=False,
            )

        if validated_page.next_page_token is not None:
            if decoded is None or decoded.sync_token is None:
                raise self._invalid_provider_response_error()
            continuation_cursor = self._encode_cursor(
                self._cursor(
                    calendar_id=calendar_id,
                    phase="changes",
                    page_token=validated_page.next_page_token.value,
                    sync_token=decoded.sync_token,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=continuation_cursor,
                proposed_checkpoint=continuation_cursor,
                has_more=True,
            )
        if validated_page.next_sync_token is None:
            raise self._invalid_provider_response_error()
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=self._encode_cursor(
                self._cursor(
                    calendar_id=calendar_id,
                    phase="changes",
                    page_token=None,
                    sync_token=validated_page.next_sync_token.value,
                )
            ),
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        google_integration = self._require_google_integration(integration)
        validated_source = self._validate_source(source)
        calendar_id = validated_source.scope.remote_scope_id
        descriptor = self._reconstruct_descriptor(item)
        self._validate_descriptor(
            descriptor,
            source=validated_source,
            calendar_id=calendar_id,
        )

        event = await self._invoke_integration(
            lambda: google_integration.read_calendar_event(
                calendar_id=calendar_id,
                event_id=descriptor.identity.remote_id,
            )
        )
        validated_event = self._reconstruct_event(event)
        if validated_event.id != descriptor.identity.remote_id:
            raise self._invalid_provider_response_error()
        if validated_event.status is GoogleCalendarEventStatus.CANCELLED:
            raise self._content_changed_error()
        record = _build_structured_record(calendar_id, validated_event)
        content_hash = _compute_content_hash(record)
        if content_hash != descriptor.revision.content_hash:
            raise self._content_changed_error()
        return KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record=record,
            mime_type=GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE,
            content_hash=content_hash,
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self._require_google_integration(integration)
        validated_source = self._validate_source(source)
        descriptor = self._reconstruct_descriptor(item)
        self._validate_descriptor(
            descriptor,
            source=validated_source,
            calendar_id=validated_source.scope.remote_scope_id,
        )
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message=_UNSUPPORTED_PERMISSIONS_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    async def _read_provider_page(
        self,
        integration: GoogleWorkspaceCollaborationSuiteIntegration,
        *,
        calendar_id: str,
        page_token: GoogleWorkspacePageToken | None,
        sync_token: GoogleCalendarSyncToken | None,
        limit: int,
        reconciliation_required_on_expiry: bool = False,
    ) -> GoogleCalendarEventPage:
        return await self._invoke_integration(
            lambda: integration.list_calendar_events_page(
                calendar_id=calendar_id,
                page_token=page_token,
                sync_token=sync_token,
                max_results=limit,
            ),
            reconciliation_required_on_expiry=reconciliation_required_on_expiry,
        )

    def _event_to_change(
        self,
        event: GoogleCalendarEvent,
        *,
        calendar_id: str,
    ) -> KnowledgeChange:
        if event.status is GoogleCalendarEventStatus.CANCELLED:
            return KnowledgeChange(
                kind=KnowledgeChangeKind.DELETED,
                remote_id=event.id,
                descriptor=None,
            )
        record = _build_structured_record(calendar_id, event)
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=event.id,
            descriptor=self._event_to_descriptor(
                event,
                calendar_id=calendar_id,
                content_hash=_compute_content_hash(record),
            ),
        )

    def _event_to_descriptor(
        self,
        event: GoogleCalendarEvent,
        *,
        calendar_id: str,
        content_hash: str,
    ) -> KnowledgeItemDescriptor:
        updated_at = _parse_updated_at(event.updated)
        return KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=event.id,
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision(
                version=str(event.sequence) if event.sequence is not None else None,
                etag=event.etag,
                content_hash=content_hash,
                acl_hash=None,
                updated_at=updated_at,
            ),
            title=event.summary or "Calendar event",
            item_type=_GOOGLE_CALENDAR_EVENT_ITEM_TYPE,
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=event.id,
                web_url=None,
                safe_locator=None,
            ),
            metadata=_event_metadata(calendar_id, event),
        )

    def _invalid_scope_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message=_INVALID_SCOPE_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_cursor_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_CURSOR,
            safe_message=_INVALID_CURSOR_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_provider_response_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message=_INVALID_PROVIDER_RESPONSE_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _invalid_descriptor_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_SCOPE,
            safe_message=_INVALID_DESCRIPTOR_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _content_changed_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
            safe_message=_CONTENT_CHANGED_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=True,
        )

    def _integration_required_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message=_INTEGRATION_REQUIRED_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
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
            if type(source.scope) is not KnowledgeSourceScope:
                raise ValueError("invalid scope type")
            if (
                type(source.tenant_id) is not str
                or type(source.provider_id) is not str
                or type(source.integration_kind) is not IntegrationCategory
                or type(source.source_kind) is not str
                or (
                    source.connection_ref is not None
                    and type(source.connection_ref) is not str
                )
                or type(source.scope.remote_scope_id) is not str
                or type(source.scope.remote_scope_type) is not str
                or type(source.scope.safe_display_name) is not str
                or type(source.scope.parameters) is not dict
            ):
                raise ValueError("invalid source fields")
            snapshot = source.model_dump(mode="python")
            scope_data = snapshot.get("scope")
            if not isinstance(scope_data, dict):
                raise ValueError("invalid scope data")
            snapshot["scope"] = KnowledgeSourceScope(**scope_data)
            return KnowledgeSourceRef(**snapshot)
        except Exception:
            raise self._invalid_scope_error() from None

    def _validate_calendar_id(self, value: object) -> str:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or len(value) > _MAX_CALENDAR_ID_LENGTH
            or _ASCII_CONTROL.search(value)
            or "/" in value
            or "\\" in value
        ):
            raise ValueError("invalid calendar id")
        return value

    def _validate_source(self, source: KnowledgeSourceRef) -> KnowledgeSourceRef:
        reconstructed = self._reconstruct_source(source)
        if (
            reconstructed.provider_id != self.provider_id
            or reconstructed.integration_kind != self.integration_kind
            or reconstructed.source_kind != self.source_kind
        ):
            raise self._invalid_scope_error()
        scope = reconstructed.scope
        if scope.remote_scope_type != GOOGLE_CALENDAR_SCOPE_TYPE or scope.parameters:
            raise self._invalid_scope_error()
        try:
            calendar_id = self._validate_calendar_id(scope.remote_scope_id)
        except ValueError:
            raise self._invalid_scope_error() from None
        if calendar_id != scope.remote_scope_id:
            raise self._invalid_scope_error()
        return KnowledgeSourceRef(
            tenant_id=reconstructed.tenant_id,
            provider_id=reconstructed.provider_id,
            integration_kind=reconstructed.integration_kind,
            source_kind=reconstructed.source_kind,
            connection_ref=reconstructed.connection_ref,
            scope=KnowledgeSourceScope(
                remote_scope_id=calendar_id,
                remote_scope_type=scope.remote_scope_type,
                safe_display_name=scope.safe_display_name,
                parameters={},
            ),
        )

    def _validate_limit(self, limit: object) -> int:
        if type(limit) is not int or not 1 <= limit <= _PROVIDER_PAGE_LIMIT:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CONFIGURATION_ERROR_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return limit

    def _scope_fingerprint(self, calendar_id: str) -> str:
        return hashlib.sha256(
            f"google_workspace\x00calendar\x00{calendar_id}".encode("utf-8")
        ).hexdigest()

    def _cursor(
        self,
        *,
        calendar_id: str,
        phase: Literal["inventory", "changes"],
        page_token: str | None,
        sync_token: str | None,
    ) -> _GoogleCalendarCursor:
        return _GoogleCalendarCursor(
            schema_version=GOOGLE_CALENDAR_CURSOR_VERSION,
            scope_fingerprint=self._scope_fingerprint(calendar_id),
            phase=phase,
            page_token=page_token,
            sync_token=sync_token,
        )

    def _encode_cursor(self, cursor: _GoogleCalendarCursor) -> KnowledgeCursor:
        raw = json.dumps(
            cursor.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return KnowledgeCursor(value=encoded, version=GOOGLE_CALENDAR_CURSOR_VERSION)

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        calendar_id: str,
    ) -> _GoogleCalendarCursor | None:
        if cursor is None:
            return None
        try:
            if type(cursor) is not KnowledgeCursor:
                raise ValueError("invalid cursor type")
            outer = KnowledgeCursor(**cursor.model_dump(mode="python"))
            value = outer.value
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > _MAX_ENCODED_CURSOR_LENGTH
                or "=" in value
                or _CURSOR_ALPHABET.fullmatch(value) is None
                or outer.version != GOOGLE_CALENDAR_CURSOR_VERSION
            ):
                raise ValueError("invalid cursor envelope")
            padding = "=" * (-len(value) % 4)
            raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            )
            if type(payload) is not dict:
                raise ValueError("invalid cursor payload")
            decoded = _GoogleCalendarCursor.model_validate(payload)
            canonical = self._encode_cursor(decoded)
            if canonical.value != outer.value or canonical.version != outer.version:
                raise ValueError("noncanonical cursor")
        except Exception:
            raise self._invalid_cursor_error() from None
        if decoded.scope_fingerprint != self._scope_fingerprint(calendar_id):
            raise self._invalid_cursor_error()
        return decoded

    def _reconstruct_page(
        self,
        page: object,
        *,
        calendar_id: str,
    ) -> GoogleCalendarEventPage:
        try:
            if type(page) is not GoogleCalendarEventPage:
                raise ValueError("invalid page type")
            events = tuple(self._reconstruct_event(event) for event in page.events)
            next_page_token = (
                GoogleWorkspacePageToken(value=page.next_page_token.value)
                if page.next_page_token is not None
                else None
            )
            next_sync_token = (
                GoogleCalendarSyncToken(value=page.next_sync_token.value)
                if page.next_sync_token is not None
                else None
            )
            rebuilt = GoogleCalendarEventPage(
                calendar_id=page.calendar_id,
                summary=page.summary,
                description=page.description,
                updated=page.updated,
                time_zone=page.time_zone,
                access_role=page.access_role,
                events=events,
                next_page_token=next_page_token,
                next_sync_token=next_sync_token,
            )
        except Exception:
            raise self._invalid_provider_response_error() from None
        if rebuilt.calendar_id != calendar_id:
            raise self._invalid_provider_response_error()
        return rebuilt

    def _reconstruct_event(self, event: object) -> GoogleCalendarEvent:
        try:
            if type(event) is not GoogleCalendarEvent:
                raise ValueError("invalid event type")
            data = event.model_dump(mode="python")
            for field_name in ("start", "end", "original_start_time"):
                value = getattr(event, field_name)
                data[field_name] = (
                    None
                    if value is None
                    else GoogleCalendarEventDateTime(
                        **value.model_dump(mode="python")
                    )
                )
            for field_name in ("creator", "organizer"):
                value = getattr(event, field_name)
                data[field_name] = (
                    None
                    if value is None
                    else GoogleCalendarPerson(**value.model_dump(mode="python"))
                )
            data["attendees"] = tuple(
                GoogleCalendarAttendee(**attendee.model_dump(mode="python"))
                for attendee in event.attendees
                if type(attendee) is GoogleCalendarAttendee
            )
            if len(data["attendees"]) != len(event.attendees):
                raise ValueError("invalid attendee type")
            if event.conference_data is None:
                data["conference_data"] = None
            else:
                conference = event.conference_data
                if type(conference) is not GoogleCalendarConferenceData:
                    raise ValueError("invalid conference type")
                conference_data = conference.model_dump(mode="python")
                conference_data["entry_points"] = tuple(
                    GoogleCalendarConferenceEntryPoint(
                        **entry_point.model_dump(mode="python")
                    )
                    for entry_point in conference.entry_points
                    if type(entry_point) is GoogleCalendarConferenceEntryPoint
                )
                if len(conference_data["entry_points"]) != len(conference.entry_points):
                    raise ValueError("invalid entry point type")
                if conference.conference_solution is None:
                    conference_data["conference_solution"] = None
                else:
                    solution = conference.conference_solution
                    if type(solution) is not GoogleCalendarConferenceSolution:
                        raise ValueError("invalid solution type")
                    conference_data["conference_solution"] = (
                        GoogleCalendarConferenceSolution(
                            **solution.model_dump(mode="python")
                        )
                    )
                data["conference_data"] = GoogleCalendarConferenceData(**conference_data)
            if event.reminders is None:
                data["reminders"] = None
            else:
                reminders = event.reminders
                if type(reminders) is not GoogleCalendarReminders:
                    raise ValueError("invalid reminders type")
                reminder_data = reminders.model_dump(mode="python")
                reminder_data["overrides"] = tuple(
                    GoogleCalendarReminder(**reminder.model_dump(mode="python"))
                    for reminder in reminders.overrides
                    if type(reminder) is GoogleCalendarReminder
                )
                if len(reminder_data["overrides"]) != len(reminders.overrides):
                    raise ValueError("invalid reminder type")
                data["reminders"] = GoogleCalendarReminders(**reminder_data)
            return GoogleCalendarEvent(**data)
        except Exception:
            raise self._invalid_provider_response_error() from None

    def _reconstruct_descriptor(self, item: object) -> KnowledgeItemDescriptor:
        try:
            if type(item) is not KnowledgeItemDescriptor:
                raise ValueError("invalid descriptor type")
            if (
                type(item.identity) is not KnowledgeItemIdentity
                or type(item.revision) is not KnowledgeItemRevision
                or type(item.provenance) is not KnowledgeItemProvenance
            ):
                raise ValueError("invalid descriptor parts")
            if (
                type(item.title) is not str
                or type(item.item_type) is not str
                or type(item.content_mode) is not KnowledgeContentMode
                or type(item.content_available) is not bool
                or type(item.identity.remote_id) is not str
                or (
                    item.identity.parent_remote_id is not None
                    and type(item.identity.parent_remote_id) is not str
                )
                or (
                    item.identity.logical_key is not None
                    and type(item.identity.logical_key) is not str
                )
                or (
                    item.revision.version is not None
                    and type(item.revision.version) is not str
                )
                or (
                    item.revision.etag is not None
                    and type(item.revision.etag) is not str
                )
                or (
                    item.revision.content_hash is not None
                    and type(item.revision.content_hash) is not str
                )
                or (
                    item.revision.acl_hash is not None
                    and type(item.revision.acl_hash) is not str
                )
                or (
                    item.revision.updated_at is not None
                    and type(item.revision.updated_at) is not datetime
                )
                or type(item.provenance.provider_id) is not str
                or type(item.provenance.source_kind) is not str
                or type(item.provenance.remote_id) is not str
                or (
                    item.provenance.web_url is not None
                    and type(item.provenance.web_url) is not str
                )
                or (
                    item.provenance.safe_locator is not None
                    and type(item.provenance.safe_locator) is not str
                )
                or type(item.metadata) is not dict
            ):
                raise ValueError("invalid descriptor field types")
            return item.model_copy(deep=True)
        except Exception:
            raise self._invalid_descriptor_error() from None

    def _validate_descriptor(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
        calendar_id: str,
    ) -> None:
        try:
            identity_remote_id = _validate_event_id(item.identity.remote_id)
            provenance_remote_id = _validate_event_id(item.provenance.remote_id)
            if (
                identity_remote_id != provenance_remote_id
                or item.identity.parent_remote_id is not None
                or item.identity.logical_key is not None
                or item.provenance.provider_id != self.provider_id
                or item.provenance.source_kind != self.source_kind
                or item.item_type != _GOOGLE_CALENDAR_EVENT_ITEM_TYPE
                or item.content_mode is not KnowledgeContentMode.STRUCTURED_RECORD
                or item.content_available is not True
                or item.provenance.web_url is not None
                or item.provenance.safe_locator is not None
            ):
                raise ValueError("descriptor identity mismatch")
            if (
                item.provenance.provider_id != source.provider_id
                or item.provenance.source_kind != source.source_kind
            ):
                raise ValueError("descriptor source mismatch")
            if (
                type(item.title) is not str
                or not item.title
                or len(item.title) > _MAX_TEXT_LENGTH
                or _ASCII_CONTROL.search(item.title)
            ):
                raise ValueError("invalid title")
            revision = item.revision
            if revision.version is not None and (
                _VERSION.fullmatch(revision.version) is None
                or len(revision.version) > _MAX_TOKEN_LENGTH
            ):
                raise ValueError("invalid version")
            if revision.etag is not None and (
                type(revision.etag) is not str
                or not revision.etag
                or revision.etag != revision.etag.strip()
                or len(revision.etag) > _MAX_TEXT_LENGTH
                or _ASCII_CONTROL.search(revision.etag)
            ):
                raise ValueError("invalid etag")
            if (
                type(revision.content_hash) is not str
                or _SHA256_HEX.fullmatch(revision.content_hash) is None
                or revision.acl_hash is not None
                or (
                    revision.updated_at is not None
                    and (
                        type(revision.updated_at) is not datetime
                        or revision.updated_at.tzinfo is None
                        or revision.updated_at.utcoffset() is None
                    )
                )
            ):
                raise ValueError("invalid revision")
            metadata = item.metadata
            if type(metadata) is not dict or set(metadata) != _METADATA_KEYS:
                raise ValueError("invalid metadata keys")
            if (
                metadata["schema_version"] != GOOGLE_CALENDAR_ITEM_METADATA_VERSION
                or metadata["structured_record_schema"]
                != GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA
                or metadata["calendar_id_hash"]
                != hashlib.sha256(calendar_id.encode("utf-8")).hexdigest()
                or metadata["status"] not in {
                    GoogleCalendarEventStatus.CONFIRMED.value,
                    GoogleCalendarEventStatus.TENTATIVE.value,
                }
                or metadata["event_type"] is not None
                and metadata["event_type"] not in {event_type.value for event_type in GoogleCalendarEventType}
                or type(metadata["recurring_event"]) is not bool
                or type(metadata["all_day"]) is not bool
            ):
                raise ValueError("invalid metadata values")
        except Exception:
            raise self._invalid_descriptor_error() from None

    async def _invoke_integration(
        self,
        operation: Callable[[], _T],
        *,
        reconciliation_required_on_expiry: bool = False,
    ) -> _T:
        try:
            return await asyncio.to_thread(operation)
        except VendorKnowledgeError:
            raise
        except GoogleWorkspaceApiError as exc:
            if reconciliation_required_on_expiry and exc.status_code == 410:
                raise self._reconciliation_required_error() from None
            raise self._map_google_api_error(exc) from None
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
            raise self._invalid_provider_response_error() from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message=_DEPENDENCY_UNAVAILABLE_MESSAGE,
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=True,
            ) from None

    def _reconciliation_required_error(self) -> VendorKnowledgeError:
        return VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.RECONCILIATION_REQUIRED,
            safe_message=_RECONCILIATION_REQUIRED_MESSAGE,
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _map_google_api_error(
        self,
        exc: GoogleWorkspaceApiError,
    ) -> VendorKnowledgeError:
        if exc.kind is GoogleWorkspaceErrorKind.AUTHENTICATION:
            code, retryable = VendorKnowledgeErrorCode.AUTHENTICATION_FAILED, False
        elif exc.kind is GoogleWorkspaceErrorKind.AUTHORIZATION:
            code, retryable = VendorKnowledgeErrorCode.AUTHORIZATION_DENIED, False
        elif exc.kind is GoogleWorkspaceErrorKind.NOT_FOUND:
            code, retryable = VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND, False
        elif exc.kind is GoogleWorkspaceErrorKind.RATE_LIMITED:
            code, retryable = VendorKnowledgeErrorCode.RATE_LIMITED, True
        elif exc.kind is GoogleWorkspaceErrorKind.TEMPORARY:
            code, retryable = VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE, True
        elif exc.kind in (
            GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT,
        ):
            code, retryable = VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE, False
        elif exc.kind in (
            GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
            GoogleWorkspaceErrorKind.INVALID_REQUEST,
        ):
            code, retryable = VendorKnowledgeErrorCode.CONFIGURATION_ERROR, False
        else:
            code, retryable = VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE, True
        return VendorKnowledgeError(
            code=code,
            safe_message=(
                _PROVIDER_INVALID_REQUEST_MESSAGE
                if exc.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST
                else _CONFIGURATION_ERROR_MESSAGE
                if code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
                else _INVALID_PROVIDER_RESPONSE_MESSAGE
                if code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
                else _DEPENDENCY_UNAVAILABLE_MESSAGE
            ),
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=retryable,
        )


def _parse_updated_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _event_metadata(
    calendar_id: str,
    event: GoogleCalendarEvent,
) -> dict[str, Any]:
    return {
        "schema_version": GOOGLE_CALENDAR_ITEM_METADATA_VERSION,
        "structured_record_schema": GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
        "calendar_id_hash": hashlib.sha256(calendar_id.encode("utf-8")).hexdigest(),
        "status": event.status.value,
        "event_type": event.event_type.value if event.event_type is not None else None,
        "recurring_event": event.recurring_event_id is not None,
        "all_day": event.start is not None and event.start.date is not None,
    }


def _build_structured_record(
    calendar_id: str,
    event: GoogleCalendarEvent,
) -> dict[str, Any]:
    return {
        "schema_version": GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
        "calendar_id": calendar_id,
        "event": event.model_dump(mode="json"),
    }


def _compute_content_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def register_google_workspace_calendar_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> GoogleWorkspaceCalendarKnowledgeAdapter:
    adapter = GoogleWorkspaceCalendarKnowledgeAdapter()
    registry.register(adapter)
    return adapter
