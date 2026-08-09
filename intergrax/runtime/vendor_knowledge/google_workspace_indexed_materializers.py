# © Artur Czarnecki. All rights reserved.

"""Google Workspace provider-owned Indexed materialization strategies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
    GoogleCalendarConferenceData,
    GoogleCalendarConferenceEntryPoint,
    GoogleCalendarConferenceSolution,
    GoogleCalendarAttendee,
    GoogleCalendarEvent,
    GoogleCalendarEventDateTime,
    GoogleCalendarPerson,
    GoogleCalendarReminder,
    GoogleCalendarReminders,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    GOOGLE_CALENDAR_SCOPE_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
)
from intergrax.runtime.vendor_knowledge.indexed_materialization import (
    MaterializedConnectedSourceDocument,
    VendorKnowledgeMaterializationError,
    build_materialized_connected_source_document,
    validate_materializer_source,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemRevision,
    KnowledgePermissions,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

GOOGLE_CALENDAR_INDEXED_RECORD_SCHEMA = GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA

_GOOGLE_CALENDAR_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
)


class _GoogleCalendarStructuredRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        populate_by_name=True,
    )

    schema_: Literal["google_workspace.calendar.event.knowledge.v1"] = Field(alias="schema")
    calendar_id: str
    event: dict[str, object]


@dataclass(frozen=True)
class _ValidatedGoogleCalendarRecord:
    calendar_id: str
    event: GoogleCalendarEvent


class GoogleCalendarStructuredRecordMaterializer:
    """Materialize the bounded structured Google Calendar event projection."""

    identity = _GOOGLE_CALENDAR_IDENTITY
    runtime_ref = "indexed-source:google_workspace:calendar"
    schema_name = GOOGLE_CALENDAR_INDEXED_RECORD_SCHEMA

    def materialize(
        self,
        *,
        source: KnowledgeSourceRef,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
        source_id: str,
        remote_id: str,
        content: KnowledgeContent,
        revision: KnowledgeItemRevision | None,
        permissions: KnowledgePermissions | None,
    ) -> MaterializedConnectedSourceDocument:
        validate_materializer_source(self.identity, source)
        record = self._validate_record(content)
        event = record.event
        if (
            source.scope.remote_scope_type != GOOGLE_CALENDAR_SCOPE_TYPE
            or source.scope.parameters
            or record.calendar_id != source.scope.remote_scope_id
            or event.id != remote_id
            or event.status.value == "cancelled"
        ):
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (tenant_id, workspace_id, binding_id, source_id, remote_id)
        ):
            raise VendorKnowledgeMaterializationError("connected_source_identity_invalid")

        markdown = _render_calendar_event(
            calendar_id=record.calendar_id,
            event=event,
        )
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:16]
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"google-calendar-event-{remote_hash_prefix}.md",
            revision=revision,
            permissions=permissions,
        )

    def _validate_record(self, content: KnowledgeContent) -> _ValidatedGoogleCalendarRecord:
        if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
            raise VendorKnowledgeMaterializationError("connected_source_content_mode_invalid")
        record = content.structured_record
        if not isinstance(record, dict):
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            )
        try:
            parsed = _GoogleCalendarStructuredRecord.model_validate_json(
                json.dumps(record, ensure_ascii=False)
            )
            event_data = dict(parsed.event)
            for field_name in ("start", "end", "original_start_time"):
                event_data[field_name] = _rebuild_optional_model(
                    event_data.get(field_name),
                    GoogleCalendarEventDateTime,
                )
            for field_name in ("creator", "organizer"):
                event_data[field_name] = _rebuild_optional_model(
                    event_data.get(field_name),
                    GoogleCalendarPerson,
                )
            event_data["attendees"] = tuple(
                GoogleCalendarAttendee(**attendee)
                for attendee in _require_list(event_data.get("attendees", []))
            )
            event_data["recurrence"] = tuple(
                _require_string(item)
                for item in _require_list(event_data.get("recurrence", []))
            )
            event_data["conference_data"] = _rebuild_conference_data(
                event_data.get("conference_data")
            )
            event_data["reminders"] = _rebuild_reminders(event_data.get("reminders"))
            return _ValidatedGoogleCalendarRecord(
                calendar_id=parsed.calendar_id,
                event=GoogleCalendarEvent(**event_data),
            )
        except (KeyError, TypeError, ValueError):
            raise VendorKnowledgeMaterializationError(
                "connected_source_structured_record_invalid"
            ) from None


def _render_calendar_event(*, calendar_id: str, event: GoogleCalendarEvent) -> str:
    title = (event.summary or "").strip() or "Calendar event"
    lines = [
        f"# {title}",
        "",
        f"Calendar: {calendar_id}",
        f"Event ID: {event.id}",
        f"Status: {event.status.value}",
    ]
    if event.description:
        lines.extend(["", event.description.strip()])
    if event.start is not None:
        lines.append(f"Starts at: {_datetime_label(event.start)}")
    if event.end is not None:
        lines.append(f"Ends at: {_datetime_label(event.end)}")
    if event.location:
        lines.append(f"Location: {event.location}")
    if event.organizer:
        organizer = _person_label(event.organizer)
        if organizer:
            lines.append(f"Organizer: {organizer}")
    if event.attendees:
        lines.append("")
        lines.append("Attendees:")
        lines.extend(f"- {_attendee_label(attendee)}" for attendee in event.attendees)
    if event.event_type is not None:
        lines.append(f"Event type: {event.event_type.value}")
    if event.visibility is not None:
        lines.append(f"Visibility: {event.visibility.value}")
    if event.transparency is not None:
        lines.append(f"Transparency: {event.transparency.value}")
    if event.created:
        lines.append(f"Created at: {event.created}")
    if event.updated:
        lines.append(f"Updated at: {event.updated}")
    if event.sequence is not None:
        lines.append(f"Sequence: {event.sequence}")
    if event.etag:
        lines.append(f"ETag: {event.etag}")
    if event.recurrence:
        lines.append(f"Recurrence rules: {', '.join(event.recurrence)}")
        lines.append("Complete recurrence expansion is not included.")
    if event.conference_data is not None:
        lines.append("Conference metadata is present; conference content is not included.")
    lines.extend(
        [
            "",
            "Attachment bytes, external document bodies, conference transcripts, "
            "historical versions, and organization-wide attendee ACLs are not included.",
            "Removal is source-owned: cancellation tombstones remove events; absence "
            "from an ordinary snapshot is not authoritative deletion.",
            "",
            f"Provider: {GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID}",
            f"Source kind: {GOOGLE_CALENDAR_SOURCE_KIND}",
            "",
        ]
    )
    return "\n".join(lines)


def _datetime_label(value: GoogleCalendarEventDateTime) -> str:
    if value.date is not None:
        return value.date
    if value.date_time is not None:
        if value.time_zone:
            return f"{value.date_time} ({value.time_zone})"
        return value.date_time
    raise VendorKnowledgeMaterializationError(
        "connected_source_structured_record_invalid"
    )


def _person_label(value: GoogleCalendarPerson) -> str:
    display_name = (value.display_name or "").strip()
    email = (value.email or "").strip()
    if display_name and email:
        return f"{display_name} <{email}>"
    return display_name or email or (value.id or "").strip()


def _attendee_label(value: GoogleCalendarAttendee) -> str:
    label = _person_label(
        GoogleCalendarPerson(
            id=value.id,
            email=value.email,
            display_name=value.display_name,
            self=value.self,
        )
    )
    response = value.response_status.value if value.response_status is not None else None
    if label and response:
        return f"{label} ({response})"
    return label or response or "Attendee metadata"


def _rebuild_optional_model(value: object, model_type: type[BaseModel]) -> BaseModel | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("nested Calendar model is invalid")
    return model_type(**value)


def _rebuild_conference_data(value: object) -> GoogleCalendarConferenceData | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("conference data is invalid")
    data = dict(value)
    data["entry_points"] = tuple(
        GoogleCalendarConferenceEntryPoint(**entry_point)
        for entry_point in _require_list(data.get("entry_points", []))
    )
    solution = data.get("conference_solution")
    data["conference_solution"] = (
        None
        if solution is None
        else GoogleCalendarConferenceSolution(**_require_dict(solution))
    )
    return GoogleCalendarConferenceData(**data)


def _rebuild_reminders(value: object) -> GoogleCalendarReminders | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("reminders are invalid")
    data = dict(value)
    data["overrides"] = tuple(
        GoogleCalendarReminder(**reminder)
        for reminder in _require_list(data.get("overrides", []))
    )
    return GoogleCalendarReminders(**data)


def _require_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("nested Calendar object is invalid")
    return value


def _require_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("nested Calendar list is invalid")
    return value


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Calendar recurrence value is invalid")
    return value


__all__ = [
    "GOOGLE_CALENDAR_INDEXED_RECORD_SCHEMA",
    "GoogleCalendarStructuredRecordMaterializer",
]
