# © Artur Czarnecki. All rights reserved.

"""Google Workspace provider-owned Indexed materialization strategies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

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
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
    GoogleDocsBlock,
    GoogleDocsBlockKind,
    GoogleDocsDocument,
    GoogleDocsInlineKind,
    GoogleDocsInlineElement,
    GoogleDocsNamedStyleType,
    GoogleDocsSegmentKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
    GoogleSheetsCell,
    GoogleSheetsCellValueKind,
    GoogleSheetsNumberFormatType,
    GoogleSheetsRecalculationInterval,
    GoogleSheetsSheetType,
    GoogleSheetsSpreadsheet,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    GOOGLE_CALENDAR_SCOPE_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_docs import (
    GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
    GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_sheets import (
    GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
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
GOOGLE_DOCS_INDEXED_RECORD_SCHEMA = GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA
GOOGLE_SHEETS_INDEXED_RECORD_SCHEMA = GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA

_MAX_GOOGLE_MATERIALIZED_MARKDOWN_CHARS = 8_000_000
_REMOTE_HASH_PREFIX_LEN = 16

_GOOGLE_CALENDAR_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
)

_GOOGLE_DOCS_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=GOOGLE_DOCS_SOURCE_KIND,
)

_GOOGLE_SHEETS_IDENTITY = VendorKnowledgeSourceIdentity(
    provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_category=IntegrationCategory.COLLABORATION_SUITE,
    source_kind=GOOGLE_SHEETS_SOURCE_KIND,
)


class _GoogleCalendarStructuredRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        populate_by_name=True,
    )

    schema_: Literal["google_workspace.calendar.event.knowledge.v1"] = Field(
        validation_alias=AliasChoices("schema", "schema_version")
    )
    calendar_id: str
    event: dict[str, object]


@dataclass(frozen=True)
class _ValidatedGoogleCalendarRecord:
    calendar_id: str
    event: GoogleCalendarEvent


class _GoogleDocsStructuredRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["google_workspace.docs.document.knowledge.v1"]
    document_id: str
    title: str
    suggestions_view_mode: Literal["PREVIEW_WITHOUT_SUGGESTIONS"]
    tabs: tuple[dict[str, object], ...]


class _GoogleSheetsStructuredRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["google_workspace.sheets.spreadsheet.knowledge.v1"]
    spreadsheet_id: str
    title: str
    locale: str
    time_zone: str
    recalculation_interval: str | None
    sheets: tuple[dict[str, object], ...]
    named_ranges: tuple[dict[str, object], ...]


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


class GoogleDocsStructuredRecordMaterializer:
    """Materialize the bounded structured Google Docs document projection."""

    identity = _GOOGLE_DOCS_IDENTITY
    runtime_ref = "indexed-source:google_workspace:docs"
    schema_name = GOOGLE_DOCS_INDEXED_RECORD_SCHEMA

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
        _validate_materializer_identity_values(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
        )
        if (
            source.scope.remote_scope_type != GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE
            or source.scope.parameters
            or source.scope.remote_scope_id != remote_id
        ):
            raise VendorKnowledgeMaterializationError("connected_source_scope_invalid")
        document = _parse_google_docs_record(content)
        if document.document_id != remote_id:
            raise VendorKnowledgeMaterializationError(
                "connected_source_remote_id_mismatch"
            )
        markdown = _render_google_docs_document(document)
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[
            :_REMOTE_HASH_PREFIX_LEN
        ]
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"google-docs-{remote_hash_prefix}.md",
            revision=revision,
            permissions=permissions,
        )


class GoogleSheetsStructuredRecordMaterializer:
    """Materialize the bounded structured Google Sheets tabular projection."""

    identity = _GOOGLE_SHEETS_IDENTITY
    runtime_ref = "indexed-source:google_workspace:sheets"
    schema_name = GOOGLE_SHEETS_INDEXED_RECORD_SCHEMA

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
        _validate_materializer_identity_values(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
        )
        if (
            source.scope.remote_scope_type != GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE
            or source.scope.parameters
            or source.scope.remote_scope_id != remote_id
        ):
            raise VendorKnowledgeMaterializationError("connected_source_scope_invalid")
        spreadsheet = _parse_google_sheets_record(content)
        if spreadsheet.spreadsheet_id != remote_id:
            raise VendorKnowledgeMaterializationError(
                "connected_source_remote_id_mismatch"
            )
        markdown = _render_google_sheets_spreadsheet(spreadsheet)
        remote_hash_prefix = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[
            :_REMOTE_HASH_PREFIX_LEN
        ]
        return build_materialized_connected_source_document(
            identity=self.identity,
            source=source,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
            source_id=source_id,
            remote_id=remote_id,
            markdown=markdown,
            safe_file_name=f"google-sheets-{remote_hash_prefix}.md",
            revision=revision,
            permissions=permissions,
        )


def _parse_google_docs_record(content: KnowledgeContent) -> GoogleDocsDocument:
    if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
        raise VendorKnowledgeMaterializationError("connected_source_content_mode_invalid")
    record = content.structured_record
    if not isinstance(record, dict):
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        )
    expected_keys = {
        "schema_version",
        "document_id",
        "title",
        "suggestions_view_mode",
        "tabs",
    }
    if set(record) != expected_keys:
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        )
    try:
        validated_record = _GoogleDocsStructuredRecord(
            **_normalize_google_record(record, enum_types=_GOOGLE_DOCS_ENUM_TYPES)
        )
        document_data = validated_record.model_dump(exclude={"schema_version"})
        return GoogleDocsDocument(
            **_normalize_google_record(document_data, enum_types=_GOOGLE_DOCS_ENUM_TYPES)
        )
    except (TypeError, ValueError):
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        ) from None


def _parse_google_sheets_record(content: KnowledgeContent) -> GoogleSheetsSpreadsheet:
    if content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
        raise VendorKnowledgeMaterializationError("connected_source_content_mode_invalid")
    record = content.structured_record
    if not isinstance(record, dict):
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        )
    expected_keys = {
        "schema_version",
        "spreadsheet_id",
        "title",
        "locale",
        "time_zone",
        "recalculation_interval",
        "sheets",
        "named_ranges",
    }
    if set(record) != expected_keys:
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        )
    try:
        validated_record = _GoogleSheetsStructuredRecord(
            **_normalize_google_record(record, enum_types=_GOOGLE_SHEETS_ENUM_TYPES)
        )
        spreadsheet_data = validated_record.model_dump(exclude={"schema_version"})
        return GoogleSheetsSpreadsheet(
            **_normalize_google_record(
                spreadsheet_data,
                enum_types=_GOOGLE_SHEETS_ENUM_TYPES,
            )
        )
    except (TypeError, ValueError):
        raise VendorKnowledgeMaterializationError(
            "connected_source_structured_record_invalid"
        ) from None


_GOOGLE_DOCS_ENUM_TYPES = (
    GoogleDocsInlineKind,
    GoogleDocsBlockKind,
    GoogleDocsSegmentKind,
    GoogleDocsNamedStyleType,
)
_GOOGLE_SHEETS_ENUM_TYPES = (
    GoogleSheetsCellValueKind,
    GoogleSheetsNumberFormatType,
    GoogleSheetsRecalculationInterval,
    GoogleSheetsSheetType,
)


def _normalize_google_record(
    value: object,
    *,
    enum_types: tuple[type, ...],
) -> object:
    if isinstance(value, list):
        return tuple(
            _normalize_google_record(item, enum_types=enum_types) for item in value
        )
    if isinstance(value, dict):
        normalized = {
            key: _normalize_google_record(item, enum_types=enum_types)
            for key, item in value.items()
        }
        if "kind" in normalized:
            normalized["kind"] = _coerce_google_enum(normalized["kind"], enum_types)
        for key in (
            "named_style_type",
            "recalculation_interval",
            "format_type",
            "sheet_type",
        ):
            if key in normalized and normalized[key] is not None:
                normalized[key] = _coerce_google_enum(
                    normalized[key],
                    enum_types,
                )
        return normalized
    return value


def _coerce_google_enum(value: object, enum_types: tuple[type, ...]) -> object:
    if not isinstance(value, str):
        return value
    for enum_type in enum_types:
        try:
            return enum_type(value)
        except ValueError:
            continue
    return value


def _validate_materializer_identity_values(
    *,
    tenant_id: str,
    workspace_id: str,
    binding_id: str,
    source_id: str,
    remote_id: str,
) -> None:
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (tenant_id, workspace_id, binding_id, source_id, remote_id)
    ):
        raise VendorKnowledgeMaterializationError("connected_source_identity_invalid")


def _safe_markdown_text(value: str) -> str:
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace(
        "|", "\\|"
    ).replace("`", "'")


def _bounded_google_markdown(lines: list[str]) -> str:
    markdown = "\n".join(lines)
    if len(markdown) > _MAX_GOOGLE_MATERIALIZED_MARKDOWN_CHARS:
        raise VendorKnowledgeMaterializationError("connected_source_content_too_large")
    return markdown


def _render_google_docs_document(document: GoogleDocsDocument) -> str:
    lines = [
        f"# {_safe_markdown_text(document.title)}",
        "",
        f"Document ID: {_safe_markdown_text(document.document_id)}",
    ]
    meaningful_content = False
    for tab in document.tabs:
        lines.extend(["", f"## Tab: {_safe_markdown_text(tab.title)}"])
        for segment in tab.segments:
            lines.extend(["", f"### {segment.kind.value.title()}"])
            for block in segment.blocks:
                block_text = _render_google_docs_block(block, lines)
                meaningful_content = meaningful_content or bool(block_text.strip())
    if not meaningful_content:
        raise VendorKnowledgeMaterializationError(
            "connected_source_meaningful_content_missing"
        )
    lines.extend(
        [
            "",
            "Known-document scope only; broad discovery, organization-wide traversal, "
            "authoritative deletion, item ACLs and external object bodies are not included.",
            "Absence from an ordinary snapshot is not authoritative deletion.",
            "",
            f"Provider: {GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID}",
            f"Source kind: {GOOGLE_DOCS_SOURCE_KIND}",
            "",
        ]
    )
    return _bounded_google_markdown(lines)


def _render_google_docs_block(block: GoogleDocsBlock, lines: list[str]) -> str:
    if block.kind is GoogleDocsBlockKind.PARAGRAPH and block.paragraph is not None:
        text = "".join(
            _render_google_docs_inline(element) for element in block.paragraph.elements
        )
        text = _safe_markdown_text(text.strip())
        if text:
            if block.paragraph.bullet is not None:
                lines.append(f"- {text}")
            elif block.paragraph.named_style_type not in {
                None,
                GoogleDocsNamedStyleType.NORMAL_TEXT,
            }:
                lines.append(f"#### {text}")
            else:
                lines.append(text)
        return text
    if block.kind is GoogleDocsBlockKind.TABLE and block.table is not None:
        table_text: list[str] = []
        for row_number, row in enumerate(block.table.table_rows, start=1):
            cells = [
                _safe_markdown_text(_render_google_docs_blocks_to_text(cell.blocks))
                for cell in row.cells
            ]
            row_text = " | ".join(cells)
            if row_text.strip():
                lines.append(f"Table row {row_number}: {row_text}")
                table_text.append(row_text)
        return " ".join(table_text)
    if block.kind is GoogleDocsBlockKind.TABLE_OF_CONTENTS:
        nested_text = [
            _render_google_docs_block(child, lines) for child in block.children
        ]
        return " ".join(nested_text)
    if block.kind is GoogleDocsBlockKind.SECTION_BREAK:
        lines.append("[Section break]")
    return ""


def _render_google_docs_blocks_to_text(blocks: tuple[GoogleDocsBlock, ...]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.kind is GoogleDocsBlockKind.PARAGRAPH and block.paragraph is not None:
            parts.append(
                "".join(
                    _render_google_docs_inline(element)
                    for element in block.paragraph.elements
                )
            )
        elif block.kind is GoogleDocsBlockKind.TABLE and block.table is not None:
            for row in block.table.table_rows:
                parts.extend(
                    _render_google_docs_blocks_to_text(cell.blocks) for cell in row.cells
                )
        elif block.children:
            parts.append(_render_google_docs_blocks_to_text(block.children))
    return " ".join(parts)


def _render_google_docs_inline(element: object) -> str:
    if not isinstance(element, GoogleDocsInlineElement):
        return ""
    kind = element.kind
    text = element.text
    if isinstance(text, str) and text:
        return text
    if kind is GoogleDocsInlineKind.AUTO_TEXT:
        return f"[{element.reference_id}]"
    if kind is GoogleDocsInlineKind.INLINE_OBJECT:
        return "[Inline object]"
    if kind is GoogleDocsInlineKind.PAGE_BREAK:
        return "[Page break]"
    if kind is GoogleDocsInlineKind.COLUMN_BREAK:
        return "[Column break]"
    if kind is GoogleDocsInlineKind.HORIZONTAL_RULE:
        return "[Horizontal rule]"
    if kind is GoogleDocsInlineKind.EQUATION:
        return "[Equation]"
    return ""


def _render_google_sheets_spreadsheet(spreadsheet: GoogleSheetsSpreadsheet) -> str:
    lines = [
        f"# {_safe_markdown_text(spreadsheet.title)}",
        "",
        f"Spreadsheet ID: {_safe_markdown_text(spreadsheet.spreadsheet_id)}",
        f"Locale: {_safe_markdown_text(spreadsheet.locale)}",
        f"Time zone: {_safe_markdown_text(spreadsheet.time_zone)}",
    ]
    meaningful_cells = 0
    for sheet in sorted(spreadsheet.sheets, key=lambda item: (item.index, item.sheet_id)):
        lines.extend(
            [
                "",
                f"## Sheet: {_safe_markdown_text(sheet.title)}",
                f"Type: {sheet.sheet_type.value}",
                f"Hidden: {str(sheet.hidden).lower()}",
            ]
        )
        for grid_data in sorted(
            sheet.grid_data,
            key=lambda item: (item.start_row_index, item.start_column_index),
        ):
            for row in grid_data.rows:
                rendered_cells: list[str] = []
                for cell in row.cells:
                    value = _render_google_sheets_cell(cell)
                    if value:
                        meaningful_cells += 1
                        rendered_cells.append(f"C{cell.column_index + 1}={value}")
                if rendered_cells:
                    lines.append(f"Row {row.row_index + 1}: " + "; ".join(rendered_cells))
        if sheet.merged_ranges:
            lines.append(f"Merged ranges: {len(sheet.merged_ranges)}")
    for named_range in sorted(
        spreadsheet.named_ranges,
        key=lambda item: (item.name, item.named_range_id),
    ):
        lines.append(
            f"Named range: {_safe_markdown_text(named_range.name)} "
            f"({_safe_markdown_text(named_range.named_range_id)})"
        )
    if not meaningful_cells:
        raise VendorKnowledgeMaterializationError(
            "connected_source_meaningful_content_missing"
        )
    lines.extend(
        [
            "",
            "Known-spreadsheet scope only; formulas are not evaluated, arbitrary workbook "
            "ranges are not expanded, binary XLSX extraction is not included, and item ACLs "
            "are unproven.",
            "Absence from an ordinary snapshot is not authoritative deletion.",
            "",
            f"Provider: {GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID}",
            f"Source kind: {GOOGLE_SHEETS_SOURCE_KIND}",
            "",
        ]
    )
    return _bounded_google_markdown(lines)


def _render_google_sheets_cell(cell: GoogleSheetsCell) -> str:
    entered = _render_google_sheets_value(cell.user_entered_value)
    effective = _render_google_sheets_value(cell.effective_value)
    if entered and effective and entered != effective:
        value = f"entered {entered}; effective {effective}"
    else:
        value = effective or entered
    if not value and cell.formatted_value:
        value = cell.formatted_value
    if cell.note:
        note = _safe_markdown_text(cell.note)
        value = f"{value}; note {note}" if value else f"note {note}"
    return _safe_markdown_text(value)


def _render_google_sheets_value(value: object) -> str:
    if value is None:
        return ""
    if value.kind in {GoogleSheetsCellValueKind.STRING, GoogleSheetsCellValueKind.FORMULA}:
        return value.text or ""
    if value.kind is GoogleSheetsCellValueKind.NUMBER:
        return str(value.number)
    if value.kind is GoogleSheetsCellValueKind.BOOLEAN:
        return str(value.boolean).lower()
    if value.kind is GoogleSheetsCellValueKind.ERROR and value.error is not None:
        suffix = f": {value.error.message}" if value.error.message else ""
        return f"error {value.error.error_type}{suffix}"
    return ""


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
