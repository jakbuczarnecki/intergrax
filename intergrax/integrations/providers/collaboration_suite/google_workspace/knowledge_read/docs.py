# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Docs knowledge-read: structured document content via shared transport."""

from __future__ import annotations

import re
from contextlib import contextmanager
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
    normalize_google_workspace_media_type,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
)

GOOGLE_DOCS_SOURCE_KIND = "docs"
GOOGLE_DOCS_NATIVE_MIME_TYPE = "application/vnd.google-apps.document"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)

_INVALID_IDENTIFIER_MESSAGE = "invalid Google Docs document identifier"
_UNEXPECTED_RESPONSE_MESSAGE = "unexpected Google Docs provider response"
_REQUEST_FAILED_MESSAGE = "Google Docs provider request failed"

_MAX_DOCUMENT_ID_LENGTH = 1024
_MAX_TITLE_LENGTH = 4096
_MAX_REVISION_ID_LENGTH = 1024
_MAX_RESOURCE_ID_LENGTH = 1024
_MAX_TEXT_FIELD_LENGTH = 4096
_MAX_TIMESTAMP_LENGTH = 256

_MAX_TABS = 256
_MAX_TAB_DEPTH = 32
_MAX_SEGMENTS = 4096
_MAX_BLOCKS = 100000
_MAX_INLINE_ELEMENTS = 200000
_MAX_STRUCTURAL_DEPTH = 32
_MAX_TEXT_CHARS = 4000000

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_TEXT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_STRUCTURAL_UNION_KEYS = frozenset(
    {"paragraph", "table", "sectionBreak", "tableOfContents"}
)
_INLINE_UNION_KEYS = frozenset(
    {
        "textRun",
        "autoText",
        "pageBreak",
        "columnBreak",
        "footnoteReference",
        "horizontalRule",
        "equation",
        "inlineObjectElement",
        "person",
        "richLink",
        "dateElement",
    }
)

_STRUCTURAL_ALLOWED_KEYS = frozenset(
    {
        "startIndex",
        "endIndex",
        "paragraph",
        "sectionBreak",
        "table",
        "tableOfContents",
    }
)
_INLINE_ALLOWED_KEYS = frozenset(
    {
        "startIndex",
        "endIndex",
        "textRun",
        "autoText",
        "pageBreak",
        "columnBreak",
        "footnoteReference",
        "horizontalRule",
        "equation",
        "inlineObjectElement",
        "person",
        "richLink",
        "dateElement",
    }
)
_TABLE_CELL_ALLOWED_KEYS = frozenset(
    {
        "startIndex",
        "endIndex",
        "content",
        "tableCellStyle",
        "suggestedInsertionIds",
        "suggestedDeletionIds",
        "suggestedTableCellStyleChanges",
    }
)
_PERSON_FORBIDDEN_TOP_LEVEL_KEYS = frozenset({"email", "name"})

class GoogleDocsNamedStyleType(StrEnum):
    NORMAL_TEXT = "NORMAL_TEXT"
    TITLE = "TITLE"
    SUBTITLE = "SUBTITLE"
    HEADING_1 = "HEADING_1"
    HEADING_2 = "HEADING_2"
    HEADING_3 = "HEADING_3"
    HEADING_4 = "HEADING_4"
    HEADING_5 = "HEADING_5"
    HEADING_6 = "HEADING_6"


class GoogleDocsSegmentKind(StrEnum):
    BODY = "BODY"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    FOOTNOTE = "FOOTNOTE"


class GoogleDocsBlockKind(StrEnum):
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    SECTION_BREAK = "SECTION_BREAK"
    TABLE_OF_CONTENTS = "TABLE_OF_CONTENTS"


class GoogleDocsInlineKind(StrEnum):
    TEXT_RUN = "TEXT_RUN"
    AUTO_TEXT = "AUTO_TEXT"
    PAGE_BREAK = "PAGE_BREAK"
    COLUMN_BREAK = "COLUMN_BREAK"
    FOOTNOTE_REFERENCE = "FOOTNOTE_REFERENCE"
    HORIZONTAL_RULE = "HORIZONTAL_RULE"
    EQUATION = "EQUATION"
    INLINE_OBJECT = "INLINE_OBJECT"
    PERSON = "PERSON"
    RICH_LINK = "RICH_LINK"
    DATE = "DATE"


_NAMED_STYLE_MAP: dict[str, GoogleDocsNamedStyleType | None] = {
    member.value: member for member in GoogleDocsNamedStyleType
}
_NAMED_STYLE_MAP["NAMED_STYLE_TYPE_UNSPECIFIED"] = None


def _validate_resource_identifier(value: object) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(trimmed) > _MAX_RESOURCE_ID_LENGTH:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return trimmed


def _validate_document_identifier(value: object) -> str:
    if type(value) is not str:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    if len(trimmed) > _MAX_DOCUMENT_ID_LENGTH:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    return trimmed


def _validate_document_id_for_request(document_id: object) -> str:
    try:
        validated = _validate_document_identifier(document_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_IDENTIFIER_MESSAGE) from None
    if "/" in validated or "\\" in validated:
        raise IntegrationConfigurationError(_INVALID_IDENTIFIER_MESSAGE)
    return validated


def _validate_exact_title(value: object) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > _MAX_TITLE_LENGTH:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _validate_bounded_text_field(value: object, *, max_length: int) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > max_length:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _validate_optional_revision_id(value: object) -> str | None:
    if value is None:
        return None
    return _validate_bounded_text_field(value, max_length=_MAX_REVISION_ID_LENGTH)


def _require_exact_dict(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _require_exact_list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _require_present_exact_dict(
    mapping: dict[str, object],
    key: str,
) -> dict[str, object]:
    if key not in mapping:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return _require_exact_dict(mapping[key])


def _require_present_exact_list(
    mapping: dict[str, object],
    key: str,
) -> list[object]:
    if key not in mapping:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return _require_exact_list(mapping[key])


def _require_exact_int(value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value < 0:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _require_positive_int(value: object) -> int:
    validated = _require_exact_int(value)
    if validated <= 0:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return validated


def _parse_optional_positive_int_in_style(
    style: dict[str, object],
    key: str,
    *,
    default: int = 1,
) -> int:
    if key not in style:
        return default
    return _require_positive_int(style[key])


def _validate_table_cell_ignored_fields(cell_mapping: dict[str, object]) -> None:
    if "suggestedInsertionIds" in cell_mapping:
        _require_exact_list(cell_mapping["suggestedInsertionIds"])
    if "suggestedDeletionIds" in cell_mapping:
        _require_exact_list(cell_mapping["suggestedDeletionIds"])
    if "suggestedTableCellStyleChanges" in cell_mapping:
        _require_exact_dict(cell_mapping["suggestedTableCellStyleChanges"])


def _validate_exact_unique_resource_id_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    validated: list[str] = []
    seen: set[str] = set()
    for item in value:
        identifier = _validate_resource_identifier(item)
        if identifier in seen:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen.add(identifier)
        validated.append(identifier)
    return tuple(validated)


def _validate_index_range(
    start: int,
    end: int,
    *,
    parent_start: int | None = None,
    parent_end: int | None = None,
) -> None:
    if start >= end:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if parent_start is not None and parent_end is not None:
        if start < parent_start or end > parent_end:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _validate_monotonic_ranges(ranges: list[tuple[int, int]]) -> None:
    previous_end = 0
    for start, end in ranges:
        if start < previous_end:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        previous_end = end


def _count_union_fields(mapping: dict[str, object], union_keys: frozenset[str]) -> tuple[str, ...]:
    present = tuple(key for key in union_keys if key in mapping)
    return present


def _reject_unknown_top_level_fields(
    mapping: dict[str, object],
    allowed_keys: frozenset[str],
) -> None:
    for key in mapping:
        if key not in allowed_keys:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _validate_exact_nonblank_bounded_field(value: object, *, max_length: int) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > max_length:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _validate_preserved_bounded_text(
    value: object,
    budget: _ParserBudget,
    *,
    max_length: int,
    require_nonblank: bool = False,
) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if require_nonblank and not value.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > max_length:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    budget.add_text(len(value))
    return value


def _parse_table_cell_spans(cell_mapping: dict[str, object]) -> tuple[int, int]:
    if "tableCellStyle" not in cell_mapping:
        return 1, 1
    style = _require_exact_dict(cell_mapping["tableCellStyle"])
    row_span = _parse_optional_positive_int_in_style(style, "rowSpan")
    column_span = _parse_optional_positive_int_in_style(style, "columnSpan")
    return row_span, column_span


class _ParserBudget:
    def __init__(self) -> None:
        self.tab_count = 0
        self.segment_count = 0
        self.block_count = 0
        self.inline_count = 0
        self.text_chars = 0
        self.depth = 0

    def add_tab(self) -> None:
        self.tab_count += 1
        if self.tab_count > _MAX_TABS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_segment(self) -> None:
        self.segment_count += 1
        if self.segment_count > _MAX_SEGMENTS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_block(self) -> None:
        self.block_count += 1
        if self.block_count > _MAX_BLOCKS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_inline(self) -> None:
        self.inline_count += 1
        if self.inline_count > _MAX_INLINE_ELEMENTS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_text(self, length: int) -> None:
        self.text_chars += length
        if self.text_chars > _MAX_TEXT_CHARS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    @contextmanager
    def depth_guard(self):
        self.depth += 1
        if self.depth > _MAX_STRUCTURAL_DEPTH:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            yield
        finally:
            self.depth -= 1


def _validate_text_content(value: object, budget: _ParserBudget) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _UNSAFE_TEXT_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    budget.add_text(len(value))
    return value


def _parse_string_map_keys(
    value: object,
    *,
    budget: _ParserBudget,
) -> tuple[str, ...]:
    if value is None:
        return ()
    mapping = _require_exact_dict(value)
    keys: list[str] = []
    seen: set[str] = set()
    for raw_key in sorted(mapping.keys()):
        if type(raw_key) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        validated = _validate_resource_identifier(raw_key)
        if validated in seen:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen.add(validated)
        _require_exact_dict(mapping[raw_key])
        keys.append(validated)
    return tuple(keys)


def _safe_construct(model_cls: type[BaseModel], **kwargs: object) -> BaseModel:
    try:
        return model_cls(**kwargs)
    except Exception:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


class GoogleDocsBullet(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    list_id: str = Field(repr=False)
    nesting_level: int

    @field_validator("list_id", mode="before")
    @classmethod
    def _validate_list_id(cls, value: object) -> str:
        return _validate_resource_identifier(value)

    @field_validator("nesting_level", mode="before")
    @classmethod
    def _validate_nesting_level(cls, value: object) -> int:
        return _require_exact_int(value)


class GoogleDocsInlineElement(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: GoogleDocsInlineKind
    start_index: int
    end_index: int

    text: str | None = Field(default=None, repr=False)
    reference_id: str | None = Field(default=None, repr=False)
    auxiliary_text: str | None = Field(default=None, repr=False)
    mime_type: str | None = None

    @field_validator("start_index", "end_index", mode="before")
    @classmethod
    def _validate_indexes(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("reference_id", mode="before")
    @classmethod
    def _validate_reference_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_resource_identifier(value)

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text_field(cls, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value

    @field_validator("auxiliary_text", mode="before")
    @classmethod
    def _validate_auxiliary_text_field(cls, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value

    @field_validator("mime_type", mode="before")
    @classmethod
    def _validate_mime_type_field(cls, value: object) -> str | None:
        if value is None:
            return None
        return normalize_google_workspace_media_type(value)

    @model_validator(mode="after")
    def _validate_invariants(self) -> GoogleDocsInlineElement:
        _validate_index_range(self.start_index, self.end_index)
        if self.kind is GoogleDocsInlineKind.TEXT_RUN:
            if self.text is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if _UNSAFE_TEXT_CONTROL.search(self.text):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if len(self.text) > _MAX_TEXT_CHARS:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.text is not None:
            if not self.text.strip() or _ASCII_CONTROL.search(self.text):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if len(self.text) > _MAX_TEXT_FIELD_LENGTH:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if self.kind is GoogleDocsInlineKind.TEXT_RUN:
            if (
                self.reference_id is not None
                or self.auxiliary_text is not None
                or self.mime_type is not None
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsInlineKind.AUTO_TEXT:
            if self.reference_id not in {"PAGE_NUMBER", "PAGE_COUNT"}:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if self.text is not None or self.auxiliary_text is not None or self.mime_type is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind in {
            GoogleDocsInlineKind.PAGE_BREAK,
            GoogleDocsInlineKind.COLUMN_BREAK,
            GoogleDocsInlineKind.HORIZONTAL_RULE,
            GoogleDocsInlineKind.EQUATION,
        }:
            if (
                self.text is not None
                or self.reference_id is not None
                or self.auxiliary_text is not None
                or self.mime_type is not None
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsInlineKind.FOOTNOTE_REFERENCE:
            if (
                self.reference_id is None
                or self.text is None
                or self.auxiliary_text is not None
                or self.mime_type is not None
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsInlineKind.INLINE_OBJECT:
            if (
                self.reference_id is None
                or self.text is not None
                or self.auxiliary_text is not None
                or self.mime_type is not None
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsInlineKind.PERSON:
            if (
                self.reference_id is None
                or self.text is None
                or self.auxiliary_text is None
                or self.mime_type is not None
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if (
                not self.auxiliary_text.strip()
                or _ASCII_CONTROL.search(self.auxiliary_text)
                or len(self.auxiliary_text) > _MAX_TEXT_FIELD_LENGTH
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsInlineKind.RICH_LINK:
            if self.reference_id is None or self.text is None or self.auxiliary_text is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsInlineKind.DATE:
            if (
                self.reference_id is None
                or self.text is None
                or self.auxiliary_text is None
                or self.mime_type is not None
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if (
                not self.auxiliary_text.strip()
                or _ASCII_CONTROL.search(self.auxiliary_text)
                or len(self.auxiliary_text) > _MAX_TIMESTAMP_LENGTH
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleDocsParagraph(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    elements: tuple[GoogleDocsInlineElement, ...]
    named_style_type: GoogleDocsNamedStyleType | None = None
    heading_id: str | None = Field(default=None, repr=False)
    bullet: GoogleDocsBullet | None = None
    positioned_object_ids: tuple[str, ...] = Field(default=(), repr=False)

    @field_validator("positioned_object_ids", mode="before")
    @classmethod
    def _validate_positioned_object_ids(cls, value: object) -> tuple[str, ...]:
        return _validate_exact_unique_resource_id_tuple(value)

    @model_validator(mode="after")
    def _validate_paragraph_invariants(self) -> GoogleDocsParagraph:
        ranges = [(element.start_index, element.end_index) for element in self.elements]
        _validate_monotonic_ranges(ranges)
        if len(self.positioned_object_ids) != len(set(self.positioned_object_ids)):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleDocsTableCell(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    start_index: int
    end_index: int
    row_span: int
    column_span: int
    blocks: tuple[GoogleDocsBlock, ...]

    @field_validator("start_index", "end_index", mode="before")
    @classmethod
    def _validate_indexes(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("row_span", "column_span", mode="before")
    @classmethod
    def _validate_spans(cls, value: object) -> int:
        return _require_positive_int(value)

    @model_validator(mode="after")
    def _validate_cell_invariants(self) -> GoogleDocsTableCell:
        _validate_index_range(self.start_index, self.end_index)
        block_ranges = [(block.start_index, block.end_index) for block in self.blocks]
        for block_start, block_end in block_ranges:
            _validate_index_range(
                block_start,
                block_end,
                parent_start=self.start_index,
                parent_end=self.end_index,
            )
        _validate_monotonic_ranges(block_ranges)
        return self


class GoogleDocsTableRow(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    start_index: int
    end_index: int
    cells: tuple[GoogleDocsTableCell, ...]

    @field_validator("start_index", "end_index", mode="before")
    @classmethod
    def _validate_indexes(cls, value: object) -> int:
        return _require_exact_int(value)

    @model_validator(mode="after")
    def _validate_row_invariants(self) -> GoogleDocsTableRow:
        _validate_index_range(self.start_index, self.end_index)
        if not self.cells:
            return self
        cell_ranges = [(cell.start_index, cell.end_index) for cell in self.cells]
        for cell_start, cell_end in cell_ranges:
            _validate_index_range(
                cell_start,
                cell_end,
                parent_start=self.start_index,
                parent_end=self.end_index,
            )
        _validate_monotonic_ranges(cell_ranges)
        return self


def _validate_table_grid(
    *,
    rows: int,
    columns: int,
    table_rows: tuple[GoogleDocsTableRow, ...],
) -> None:
    active_spans = [0] * columns

    for row_index, row in enumerate(table_rows):
        occupied = [active_spans[column] > 0 for column in range(columns)]
        column = 0

        for cell in row.cells:
            while column < columns and occupied[column]:
                column += 1

            if column + cell.column_span > columns:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

            for span_column in range(column, column + cell.column_span):
                if occupied[span_column]:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

            if row_index + cell.row_span > rows:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

            for span_column in range(column, column + cell.column_span):
                occupied[span_column] = True
                active_spans[span_column] = max(active_spans[span_column], cell.row_span)

            column += cell.column_span

        if not all(occupied):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

        for span_column in range(columns):
            if active_spans[span_column] > 0:
                active_spans[span_column] -= 1

    if any(remaining > 0 for remaining in active_spans):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


class GoogleDocsTable(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    rows: int
    columns: int
    table_rows: tuple[GoogleDocsTableRow, ...]

    @field_validator("rows", "columns", mode="before")
    @classmethod
    def _validate_dimensions(cls, value: object) -> int:
        return _require_positive_int(value)

    @model_validator(mode="after")
    def _validate_table_invariants(self) -> GoogleDocsTable:
        if self.rows != len(self.table_rows):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        row_ranges = [(row.start_index, row.end_index) for row in self.table_rows]
        _validate_monotonic_ranges(row_ranges)
        _validate_table_grid(rows=self.rows, columns=self.columns, table_rows=self.table_rows)
        return self


class GoogleDocsBlock(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: GoogleDocsBlockKind
    start_index: int
    end_index: int

    paragraph: GoogleDocsParagraph | None = None
    table: GoogleDocsTable | None = None
    children: tuple[GoogleDocsBlock, ...] = ()

    @field_validator("start_index", "end_index", mode="before")
    @classmethod
    def _validate_indexes(cls, value: object) -> int:
        return _require_exact_int(value)

    @model_validator(mode="after")
    def _validate_block_invariants(self) -> GoogleDocsBlock:
        _validate_index_range(self.start_index, self.end_index)
        if self.kind is GoogleDocsBlockKind.PARAGRAPH:
            if self.paragraph is None or self.table is not None or self.children:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsBlockKind.TABLE:
            if self.table is None or self.paragraph is not None or self.children:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsBlockKind.SECTION_BREAK:
            if self.paragraph is not None or self.table is not None or self.children:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleDocsBlockKind.TABLE_OF_CONTENTS:
            if self.paragraph is not None or self.table is not None or not self.children:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleDocsSegment(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: GoogleDocsSegmentKind
    segment_id: str | None = Field(default=None, repr=False)
    blocks: tuple[GoogleDocsBlock, ...]

    @field_validator("segment_id", mode="before")
    @classmethod
    def _validate_segment_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_resource_identifier(value)

    @model_validator(mode="after")
    def _validate_segment_invariants(self) -> GoogleDocsSegment:
        if self.kind is GoogleDocsSegmentKind.BODY:
            if self.segment_id is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        else:
            if self.segment_id is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        block_ranges = [(block.start_index, block.end_index) for block in self.blocks]
        _validate_monotonic_ranges(block_ranges)
        return self


class GoogleDocsTab(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    tab_id: str = Field(repr=False)
    title: str = Field(repr=False)
    parent_tab_id: str | None = Field(default=None, repr=False)
    index: int
    nesting_level: int

    list_ids: tuple[str, ...] = Field(default=(), repr=False)
    inline_object_ids: tuple[str, ...] = Field(default=(), repr=False)
    positioned_object_ids: tuple[str, ...] = Field(default=(), repr=False)

    segments: tuple[GoogleDocsSegment, ...]

    @field_validator("tab_id", "parent_tab_id", mode="before")
    @classmethod
    def _validate_tab_ids(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_resource_identifier(value)

    @field_validator("title", mode="before")
    @classmethod
    def _validate_title(cls, value: object) -> str:
        return _validate_exact_title(value)

    @field_validator("index", "nesting_level", mode="before")
    @classmethod
    def _validate_int_fields(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("list_ids", "inline_object_ids", "positioned_object_ids", mode="before")
    @classmethod
    def _validate_tab_resource_id_tuples(cls, value: object) -> tuple[str, ...]:
        return _validate_exact_unique_resource_id_tuple(value)

    @model_validator(mode="after")
    def _validate_tab_invariants(self) -> GoogleDocsTab:
        if not self.segments:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if self.segments[0].kind is not GoogleDocsSegmentKind.BODY:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        segment_kinds = [segment.kind for segment in self.segments[1:]]
        expected_order = [
            GoogleDocsSegmentKind.HEADER,
            GoogleDocsSegmentKind.FOOTER,
            GoogleDocsSegmentKind.FOOTNOTE,
        ]
        current_stage = 0
        for kind in segment_kinds:
            while current_stage < len(expected_order) and expected_order[current_stage] != kind:
                current_stage += 1
            if current_stage >= len(expected_order) or expected_order[current_stage] != kind:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen_by_kind: dict[GoogleDocsSegmentKind, set[str]] = {
            GoogleDocsSegmentKind.HEADER: set(),
            GoogleDocsSegmentKind.FOOTER: set(),
            GoogleDocsSegmentKind.FOOTNOTE: set(),
        }
        for segment in self.segments:
            if segment.kind in seen_by_kind:
                if segment.segment_id is None or segment.segment_id in seen_by_kind[segment.kind]:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                seen_by_kind[segment.kind].add(segment.segment_id)
        return self


class GoogleDocsDocument(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    document_id: str = Field(repr=False)
    title: str = Field(repr=False)
    revision_id: str | None = Field(default=None, repr=False)
    suggestions_view_mode: Literal["PREVIEW_WITHOUT_SUGGESTIONS"]
    tabs: tuple[GoogleDocsTab, ...]

    @field_validator("document_id", mode="before")
    @classmethod
    def _validate_document_id(cls, value: object) -> str:
        return _validate_resource_identifier(value)

    @field_validator("title", mode="before")
    @classmethod
    def _validate_title(cls, value: object) -> str:
        return _validate_exact_title(value)

    @field_validator("revision_id", mode="before")
    @classmethod
    def _validate_revision_id(cls, value: object) -> str | None:
        return _validate_optional_revision_id(value)

    @model_validator(mode="after")
    def _validate_document_invariants(self) -> GoogleDocsDocument:
        if not self.tabs:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen_tab_ids: set[str] = set()
        seen_parents: set[str] = set()
        siblings_by_parent: dict[str | None, list[int]] = {}
        for tab in self.tabs:
            if tab.tab_id in seen_tab_ids:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_tab_ids.add(tab.tab_id)
            if tab.nesting_level == 0:
                if tab.parent_tab_id is not None:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            else:
                if tab.parent_tab_id is None or tab.parent_tab_id not in seen_parents:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                parent_tab = next(
                    candidate
                    for candidate in self.tabs
                    if candidate.tab_id == tab.parent_tab_id
                )
                if tab.nesting_level != parent_tab.nesting_level + 1:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_parents.add(tab.tab_id)
            siblings_by_parent.setdefault(tab.parent_tab_id, []).append(tab.index)
        for indexes in siblings_by_parent.values():
            if indexes != list(range(len(indexes))):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

        ancestry_stack: list[str] = []
        for tab in self.tabs:
            target_depth = tab.nesting_level
            while len(ancestry_stack) > target_depth:
                ancestry_stack.pop()
            if tab.parent_tab_id is not None:
                if not ancestry_stack or ancestry_stack[-1] != tab.parent_tab_id:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            ancestry_stack.append(tab.tab_id)

        return self


def _parse_named_style_type(value: object) -> GoogleDocsNamedStyleType | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value not in _NAMED_STYLE_MAP:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return _NAMED_STYLE_MAP[value]


def _parse_inline_element(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    footnote_ids: frozenset[str],
    inline_object_ids: frozenset[str],
) -> GoogleDocsInlineElement:
    budget.add_inline()
    _reject_unknown_top_level_fields(mapping, _INLINE_ALLOWED_KEYS)
    start_index = _require_exact_int(mapping.get("startIndex"))
    end_index = _require_exact_int(mapping.get("endIndex"))
    _validate_index_range(start_index, end_index)

    union_fields = _count_union_fields(mapping, _INLINE_UNION_KEYS)
    if len(union_fields) != 1:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    union_key = union_fields[0]

    if union_key == "textRun":
        text_run = _require_exact_dict(mapping["textRun"])
        if "content" not in text_run:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        text = _validate_text_content(text_run["content"], budget)
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.TEXT_RUN,
            start_index=start_index,
            end_index=end_index,
            text=text,
        )

    if union_key == "autoText":
        auto_text = _require_exact_dict(mapping["autoText"])
        auto_type = auto_text.get("type")
        if type(auto_type) is not str or auto_type not in {"PAGE_NUMBER", "PAGE_COUNT"}:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.AUTO_TEXT,
            start_index=start_index,
            end_index=end_index,
            reference_id=auto_type,
        )

    if union_key == "pageBreak":
        _require_exact_dict(mapping["pageBreak"])
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.PAGE_BREAK,
            start_index=start_index,
            end_index=end_index,
        )

    if union_key == "columnBreak":
        _require_exact_dict(mapping["columnBreak"])
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.COLUMN_BREAK,
            start_index=start_index,
            end_index=end_index,
        )

    if union_key == "horizontalRule":
        _require_exact_dict(mapping["horizontalRule"])
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.HORIZONTAL_RULE,
            start_index=start_index,
            end_index=end_index,
        )

    if union_key == "equation":
        _require_exact_dict(mapping["equation"])
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.EQUATION,
            start_index=start_index,
            end_index=end_index,
        )

    if union_key == "footnoteReference":
        footnote_ref = _require_exact_dict(mapping["footnoteReference"])
        footnote_id = _validate_resource_identifier(footnote_ref.get("footnoteId"))
        if footnote_id not in footnote_ids:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        footnote_number = _validate_preserved_bounded_text(
            footnote_ref.get("footnoteNumber"),
            budget,
            max_length=_MAX_TEXT_FIELD_LENGTH,
            require_nonblank=True,
        )
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.FOOTNOTE_REFERENCE,
            start_index=start_index,
            end_index=end_index,
            reference_id=footnote_id,
            text=footnote_number,
        )

    if union_key == "inlineObjectElement":
        inline_obj = _require_exact_dict(mapping["inlineObjectElement"])
        inline_object_id = _validate_resource_identifier(inline_obj.get("inlineObjectId"))
        if inline_object_id not in inline_object_ids:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.INLINE_OBJECT,
            start_index=start_index,
            end_index=end_index,
            reference_id=inline_object_id,
        )

    if union_key == "person":
        person = _require_exact_dict(mapping["person"])
        for forbidden_key in _PERSON_FORBIDDEN_TOP_LEVEL_KEYS:
            if forbidden_key in person:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        person_id = _validate_resource_identifier(person.get("personId"))
        properties = _require_exact_dict(person.get("personProperties"))
        email = _validate_preserved_bounded_text(
            properties.get("email"),
            budget,
            max_length=_MAX_TEXT_FIELD_LENGTH,
            require_nonblank=True,
        )
        name_value = properties.get("name")
        if name_value is not None:
            text = _validate_preserved_bounded_text(
                name_value,
                budget,
                max_length=_MAX_TEXT_FIELD_LENGTH,
                require_nonblank=True,
            )
        else:
            text = email
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.PERSON,
            start_index=start_index,
            end_index=end_index,
            reference_id=person_id,
            text=text,
            auxiliary_text=email,
        )

    if union_key == "richLink":
        rich_link = _require_exact_dict(mapping["richLink"])
        rich_link_id = _validate_resource_identifier(rich_link.get("richLinkId"))
        properties = _require_exact_dict(rich_link.get("richLinkProperties"))
        _validate_exact_nonblank_bounded_field(
            properties.get("uri"),
            max_length=_MAX_TEXT_FIELD_LENGTH,
        )
        title = _validate_preserved_bounded_text(
            properties.get("title"),
            budget,
            max_length=_MAX_TEXT_FIELD_LENGTH,
            require_nonblank=True,
        )
        mime_value = properties.get("mimeType")
        mime_type: str | None = None
        if mime_value is not None:
            mime_type = normalize_google_workspace_media_type(mime_value)
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.RICH_LINK,
            start_index=start_index,
            end_index=end_index,
            reference_id=rich_link_id,
            text=title,
            mime_type=mime_type,
        )

    if union_key == "dateElement":
        date_elem = _require_exact_dict(mapping["dateElement"])
        date_id = _validate_resource_identifier(date_elem.get("dateId"))
        properties = _require_exact_dict(date_elem.get("dateElementProperties"))
        display_text = _validate_preserved_bounded_text(
            properties.get("displayText"),
            budget,
            max_length=_MAX_TEXT_FIELD_LENGTH,
            require_nonblank=True,
        )
        timestamp = _validate_preserved_bounded_text(
            properties.get("timestamp"),
            budget,
            max_length=_MAX_TIMESTAMP_LENGTH,
            require_nonblank=True,
        )
        return _safe_construct(
            GoogleDocsInlineElement,
            kind=GoogleDocsInlineKind.DATE,
            start_index=start_index,
            end_index=end_index,
            reference_id=date_id,
            text=display_text,
            auxiliary_text=timestamp,
        )

    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _parse_paragraph_elements(
    value: object,
    budget: _ParserBudget,
    *,
    parent_start: int,
    parent_end: int,
    footnote_ids: frozenset[str],
    inline_object_ids: frozenset[str],
) -> tuple[GoogleDocsInlineElement, ...]:
    elements_list = _require_exact_list(value)
    ranges: list[tuple[int, int]] = []
    parsed: list[GoogleDocsInlineElement] = []
    for raw_element in elements_list:
        element_mapping = _require_exact_dict(raw_element)
        element = _parse_inline_element(
            element_mapping,
            budget,
            footnote_ids=footnote_ids,
            inline_object_ids=inline_object_ids,
        )
        _validate_index_range(
            element.start_index,
            element.end_index,
            parent_start=parent_start,
            parent_end=parent_end,
        )
        ranges.append((element.start_index, element.end_index))
        parsed.append(element)
    _validate_monotonic_ranges(ranges)
    return tuple(parsed)


def _parse_paragraph(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    parent_start: int,
    parent_end: int,
    list_ids: frozenset[str],
    positioned_object_ids: frozenset[str],
    footnote_ids: frozenset[str],
    inline_object_ids: frozenset[str],
) -> GoogleDocsParagraph:
    elements = _parse_paragraph_elements(
        _require_present_exact_list(mapping, "elements"),
        budget,
        parent_start=parent_start,
        parent_end=parent_end,
        footnote_ids=footnote_ids,
        inline_object_ids=inline_object_ids,
    )

    named_style_type: GoogleDocsNamedStyleType | None = None
    heading_id: str | None = None
    paragraph_style = mapping.get("paragraphStyle")
    if paragraph_style is not None:
        style_mapping = _require_exact_dict(paragraph_style)
        if "namedStyleType" in style_mapping:
            named_style_type = _parse_named_style_type(style_mapping["namedStyleType"])
        heading_value = style_mapping.get("headingId")
        if heading_value is not None:
            heading_id = _validate_resource_identifier(heading_value)

    bullet: GoogleDocsBullet | None = None
    bullet_value = mapping.get("bullet")
    if bullet_value is not None:
        bullet_mapping = _require_exact_dict(bullet_value)
        list_id = _validate_resource_identifier(bullet_mapping.get("listId"))
        if list_id not in list_ids:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        nesting_level = _require_exact_int(bullet_mapping.get("nestingLevel"))
        bullet = _safe_construct(
            GoogleDocsBullet,
            list_id=list_id,
            nesting_level=nesting_level,
        )

    positioned_ids_value = mapping.get("positionedObjectIds")
    positioned_ids: tuple[str, ...] = ()
    if positioned_ids_value is not None:
        positioned_ids = tuple(
            _validate_resource_identifier(item) for item in _require_exact_list(positioned_ids_value)
        )
        for positioned_id in positioned_ids:
            if positioned_id not in positioned_object_ids:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    return _safe_construct(
        GoogleDocsParagraph,
        elements=elements,
        named_style_type=named_style_type,
        heading_id=heading_id,
        bullet=bullet,
        positioned_object_ids=positioned_ids,
    )


def _parse_blocks(
    content: object,
    budget: _ParserBudget,
    *,
    parent_start: int | None = None,
    parent_end: int | None = None,
    list_ids: frozenset[str],
    inline_object_ids: frozenset[str],
    positioned_object_ids: frozenset[str],
    footnote_ids: frozenset[str],
) -> tuple[GoogleDocsBlock, ...]:
    content_list = _require_exact_list(content)
    ranges: list[tuple[int, int]] = []
    blocks: list[GoogleDocsBlock] = []

    with budget.depth_guard():
        for raw_item in content_list:
            budget.add_block()
            item = _require_exact_dict(raw_item)
            _reject_unknown_top_level_fields(item, _STRUCTURAL_ALLOWED_KEYS)
            start_index = _require_exact_int(item.get("startIndex"))
            end_index = _require_exact_int(item.get("endIndex"))
            _validate_index_range(start_index, end_index, parent_start=parent_start, parent_end=parent_end)
            ranges.append((start_index, end_index))

            union_fields = _count_union_fields(item, _STRUCTURAL_UNION_KEYS)
            if len(union_fields) != 1:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            union_key = union_fields[0]

            if union_key == "paragraph":
                paragraph = _parse_paragraph(
                    _require_exact_dict(item["paragraph"]),
                    budget,
                    parent_start=start_index,
                    parent_end=end_index,
                    list_ids=list_ids,
                    positioned_object_ids=positioned_object_ids,
                    footnote_ids=footnote_ids,
                    inline_object_ids=inline_object_ids,
                )
                blocks.append(
                    _safe_construct(
                        GoogleDocsBlock,
                        kind=GoogleDocsBlockKind.PARAGRAPH,
                        start_index=start_index,
                        end_index=end_index,
                        paragraph=paragraph,
                    )
                )
            elif union_key == "table":
                table_mapping = _require_exact_dict(item["table"])
                rows = _require_positive_int(table_mapping.get("rows"))
                columns = _require_positive_int(table_mapping.get("columns"))
                raw_rows = _require_exact_list(table_mapping.get("tableRows"))
                if len(raw_rows) != rows:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

                row_ranges: list[tuple[int, int]] = []
                parsed_rows: list[GoogleDocsTableRow] = []
                for raw_row in raw_rows:
                    row_mapping = _require_exact_dict(raw_row)
                    row_start = _require_exact_int(row_mapping.get("startIndex"))
                    row_end = _require_exact_int(row_mapping.get("endIndex"))
                    _validate_index_range(
                        row_start,
                        row_end,
                        parent_start=start_index,
                        parent_end=end_index,
                    )
                    row_ranges.append((row_start, row_end))

                    raw_cells = _require_exact_list(row_mapping.get("tableCells"))
                    if len(raw_cells) > columns:
                        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

                    cell_ranges: list[tuple[int, int]] = []
                    parsed_cells: list[GoogleDocsTableCell] = []
                    for raw_cell in raw_cells:
                        cell_mapping = _require_exact_dict(raw_cell)
                        _reject_unknown_top_level_fields(cell_mapping, _TABLE_CELL_ALLOWED_KEYS)
                        _validate_table_cell_ignored_fields(cell_mapping)
                        cell_start = _require_exact_int(cell_mapping.get("startIndex"))
                        cell_end = _require_exact_int(cell_mapping.get("endIndex"))
                        _validate_index_range(
                            cell_start,
                            cell_end,
                            parent_start=row_start,
                            parent_end=row_end,
                        )
                        cell_ranges.append((cell_start, cell_end))
                        row_span, column_span = _parse_table_cell_spans(cell_mapping)
                        cell_blocks = _parse_blocks(
                            _require_present_exact_list(cell_mapping, "content"),
                            budget,
                            parent_start=cell_start,
                            parent_end=cell_end,
                            list_ids=list_ids,
                            inline_object_ids=inline_object_ids,
                            positioned_object_ids=positioned_object_ids,
                            footnote_ids=footnote_ids,
                        )
                        parsed_cells.append(
                            _safe_construct(
                                GoogleDocsTableCell,
                                start_index=cell_start,
                                end_index=cell_end,
                                row_span=row_span,
                                column_span=column_span,
                                blocks=cell_blocks,
                            )
                        )
                    _validate_monotonic_ranges(cell_ranges)
                    parsed_rows.append(
                        _safe_construct(
                            GoogleDocsTableRow,
                            start_index=row_start,
                            end_index=row_end,
                            cells=tuple(parsed_cells),
                        )
                    )
                _validate_monotonic_ranges(row_ranges)

                table_rows_tuple = tuple(parsed_rows)
                _validate_table_grid(rows=rows, columns=columns, table_rows=table_rows_tuple)
                table = _safe_construct(
                    GoogleDocsTable,
                    rows=rows,
                    columns=columns,
                    table_rows=table_rows_tuple,
                )
                blocks.append(
                    _safe_construct(
                        GoogleDocsBlock,
                        kind=GoogleDocsBlockKind.TABLE,
                        start_index=start_index,
                        end_index=end_index,
                        table=table,
                    )
                )
            elif union_key == "sectionBreak":
                _require_exact_dict(item["sectionBreak"])
                blocks.append(
                    _safe_construct(
                        GoogleDocsBlock,
                        kind=GoogleDocsBlockKind.SECTION_BREAK,
                        start_index=start_index,
                        end_index=end_index,
                    )
                )
            elif union_key == "tableOfContents":
                toc_mapping = _require_exact_dict(item["tableOfContents"])
                children = _parse_blocks(
                    _require_present_exact_list(toc_mapping, "content"),
                    budget,
                    parent_start=start_index,
                    parent_end=end_index,
                    list_ids=list_ids,
                    inline_object_ids=inline_object_ids,
                    positioned_object_ids=positioned_object_ids,
                    footnote_ids=footnote_ids,
                )
                blocks.append(
                    _safe_construct(
                        GoogleDocsBlock,
                        kind=GoogleDocsBlockKind.TABLE_OF_CONTENTS,
                        start_index=start_index,
                        end_index=end_index,
                        children=children,
                    )
                )

    _validate_monotonic_ranges(ranges)
    return tuple(blocks)


def _parse_mapped_segment(
    mapping: dict[str, object],
    *,
    kind: GoogleDocsSegmentKind,
    id_field: str,
    map_key: str,
    budget: _ParserBudget,
    list_ids: frozenset[str],
    inline_object_ids: frozenset[str],
    positioned_object_ids: frozenset[str],
    footnote_ids: frozenset[str],
) -> GoogleDocsSegment:
    budget.add_segment()
    resource_id = _validate_resource_identifier(mapping.get(id_field))
    if resource_id != map_key:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    blocks = _parse_blocks(
        _require_present_exact_list(mapping, "content"),
        budget,
        list_ids=list_ids,
        inline_object_ids=inline_object_ids,
        positioned_object_ids=positioned_object_ids,
        footnote_ids=footnote_ids,
    )
    return _safe_construct(
        GoogleDocsSegment,
        kind=kind,
        segment_id=resource_id,
        blocks=blocks,
    )


def _parse_body_segment(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    list_ids: frozenset[str],
    inline_object_ids: frozenset[str],
    positioned_object_ids: frozenset[str],
    footnote_ids: frozenset[str],
) -> GoogleDocsSegment:
    budget.add_segment()
    blocks = _parse_blocks(
        _require_present_exact_list(mapping, "content"),
        budget,
        list_ids=list_ids,
        inline_object_ids=inline_object_ids,
        positioned_object_ids=positioned_object_ids,
        footnote_ids=footnote_ids,
    )
    return _safe_construct(
        GoogleDocsSegment,
        kind=GoogleDocsSegmentKind.BODY,
        blocks=blocks,
    )


def _parse_resource_map_segments(
    value: object,
    *,
    kind: GoogleDocsSegmentKind,
    id_field: str,
    budget: _ParserBudget,
    list_ids: frozenset[str],
    inline_object_ids: frozenset[str],
    positioned_object_ids: frozenset[str],
    footnote_ids: frozenset[str],
) -> tuple[GoogleDocsSegment, ...]:
    if value is None:
        return ()
    mapping = _require_exact_dict(value)
    segments: list[GoogleDocsSegment] = []
    seen_ids: set[str] = set()
    for map_key in sorted(mapping.keys()):
        if type(map_key) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        validated_key = _validate_resource_identifier(map_key)
        if validated_key in seen_ids:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen_ids.add(validated_key)
        resource = _require_exact_dict(mapping[map_key])
        segments.append(
            _parse_mapped_segment(
                resource,
                kind=kind,
                id_field=id_field,
                map_key=validated_key,
                budget=budget,
                list_ids=list_ids,
                inline_object_ids=inline_object_ids,
                positioned_object_ids=positioned_object_ids,
                footnote_ids=footnote_ids,
            )
        )
    return tuple(segments)


def _collect_footnote_ids(footnotes_value: object) -> frozenset[str]:
    if footnotes_value is None:
        return frozenset()
    mapping = _require_exact_dict(footnotes_value)
    ids: set[str] = set()
    for map_key in mapping:
        if type(map_key) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        validated_key = _validate_resource_identifier(map_key)
        if validated_key in ids:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        resource = _require_exact_dict(mapping[map_key])
        footnote_id = _validate_resource_identifier(resource.get("footnoteId"))
        if footnote_id != validated_key:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        ids.add(footnote_id)
    return frozenset(ids)


def _parse_document_tab(
    raw_tab: dict[str, object],
    budget: _ParserBudget,
    *,
    parent_tab_id: str | None,
    nesting_level: int,
    expected_index: int,
) -> GoogleDocsTab:
    tab_properties = raw_tab.get("tabProperties")
    if tab_properties is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    props = _require_exact_dict(tab_properties)

    tab_id = _validate_resource_identifier(props.get("tabId"))
    title = _validate_exact_title(props.get("title"))
    index = _require_exact_int(props.get("index"))
    if index != expected_index:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    tab_nesting_level = _require_exact_int(props.get("nestingLevel"))
    if tab_nesting_level != nesting_level:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    raw_parent = props.get("parentTabId")
    if nesting_level == 0:
        if raw_parent is None:
            parsed_parent = None
        elif type(raw_parent) is str:
            if raw_parent == "":
                parsed_parent = None
            elif not raw_parent.strip():
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            else:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        else:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        if parent_tab_id is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        parsed_parent = _validate_resource_identifier(raw_parent)
        if parsed_parent != parent_tab_id:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    document_tab = raw_tab.get("documentTab")
    if document_tab is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    doc_tab = _require_exact_dict(document_tab)

    list_ids_tuple = _parse_string_map_keys(doc_tab.get("lists"), budget=budget)
    inline_object_ids_tuple = _parse_string_map_keys(doc_tab.get("inlineObjects"), budget=budget)
    positioned_object_ids_tuple = _parse_string_map_keys(
        doc_tab.get("positionedObjects"),
        budget=budget,
    )
    list_ids = frozenset(list_ids_tuple)
    inline_object_ids = frozenset(inline_object_ids_tuple)
    positioned_object_ids = frozenset(positioned_object_ids_tuple)
    footnote_ids = _collect_footnote_ids(doc_tab.get("footnotes"))

    body = doc_tab.get("body")
    if body is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    body_mapping = _require_exact_dict(body)

    segments: list[GoogleDocsSegment] = [
        _parse_body_segment(
            body_mapping,
            budget,
            list_ids=list_ids,
            inline_object_ids=inline_object_ids,
            positioned_object_ids=positioned_object_ids,
            footnote_ids=footnote_ids,
        ),
    ]
    segments.extend(
        _parse_resource_map_segments(
            doc_tab.get("headers"),
            kind=GoogleDocsSegmentKind.HEADER,
            id_field="headerId",
            budget=budget,
            list_ids=list_ids,
            inline_object_ids=inline_object_ids,
            positioned_object_ids=positioned_object_ids,
            footnote_ids=footnote_ids,
        )
    )
    segments.extend(
        _parse_resource_map_segments(
            doc_tab.get("footers"),
            kind=GoogleDocsSegmentKind.FOOTER,
            id_field="footerId",
            budget=budget,
            list_ids=list_ids,
            inline_object_ids=inline_object_ids,
            positioned_object_ids=positioned_object_ids,
            footnote_ids=footnote_ids,
        )
    )
    segments.extend(
        _parse_resource_map_segments(
            doc_tab.get("footnotes"),
            kind=GoogleDocsSegmentKind.FOOTNOTE,
            id_field="footnoteId",
            budget=budget,
            list_ids=list_ids,
            inline_object_ids=inline_object_ids,
            positioned_object_ids=positioned_object_ids,
            footnote_ids=footnote_ids,
        )
    )

    return _safe_construct(
        GoogleDocsTab,
        tab_id=tab_id,
        title=title,
        parent_tab_id=parsed_parent,
        index=index,
        nesting_level=nesting_level,
        list_ids=list_ids_tuple,
        inline_object_ids=inline_object_ids_tuple,
        positioned_object_ids=positioned_object_ids_tuple,
        segments=tuple(segments),
    )


def _flatten_tabs(
    raw_tabs: object,
    budget: _ParserBudget,
    *,
    parent_tab_id: str | None = None,
    nesting_level: int = 0,
    seen_tab_ids: set[str],
) -> tuple[GoogleDocsTab, ...]:
    if nesting_level > _MAX_TAB_DEPTH:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    tabs_list = _require_exact_list(raw_tabs)
    parsed: list[GoogleDocsTab] = []
    for index, raw_tab in enumerate(tabs_list):
        budget.add_tab()
        tab_mapping = _require_exact_dict(raw_tab)
        tab = _parse_document_tab(
            tab_mapping,
            budget,
            parent_tab_id=parent_tab_id,
            nesting_level=nesting_level,
            expected_index=index,
        )
        if tab.tab_id in seen_tab_ids:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen_tab_ids.add(tab.tab_id)
        parsed.append(tab)

        child_tabs = tab_mapping.get("childTabs")
        if child_tabs is None:
            child_list: list[object] = []
        else:
            child_list = _require_exact_list(child_tabs)
        if child_list:
            children = _flatten_tabs(
                child_list,
                budget,
                parent_tab_id=tab.tab_id,
                nesting_level=nesting_level + 1,
                seen_tab_ids=seen_tab_ids,
            )
            parsed.extend(children)
    return tuple(parsed)


def _parse_document(payload: dict[str, object], *, requested_id: str) -> GoogleDocsDocument:
    if "body" in payload:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    comments = payload.get("comments")
    if comments is not None:
        comments_list = _require_exact_list(comments)
        if comments_list:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    suggestions = payload.get("suggestions")
    if suggestions is not None:
        suggestions_list = _require_exact_list(suggestions)
        if suggestions_list:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    document_id = _validate_resource_identifier(payload.get("documentId"))
    if document_id != requested_id:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    title = _validate_exact_title(payload.get("title"))
    revision_id = _validate_optional_revision_id(payload.get("revisionId"))

    suggestions_view_mode = payload.get("suggestionsViewMode")
    if suggestions_view_mode != "PREVIEW_WITHOUT_SUGGESTIONS":
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    raw_tabs = payload.get("tabs")
    if raw_tabs is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    tabs_list = _require_exact_list(raw_tabs)
    if not tabs_list:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    budget = _ParserBudget()
    tabs = _flatten_tabs(tabs_list, budget, seen_tab_ids=set())

    return _safe_construct(
        GoogleDocsDocument,
        document_id=document_id,
        title=title,
        revision_id=revision_id,
        suggestions_view_mode="PREVIEW_WITHOUT_SUGGESTIONS",
        tabs=tabs,
    )


@runtime_checkable
class GoogleDocsKnowledgeReadClient(Protocol):
    def read_document(
        self,
        *,
        document_id: str,
    ) -> GoogleDocsDocument:
        ...


class GoogleDocsKnowledgeReader:
    """Stateless Google Docs knowledge reader using one shared transport."""

    def __init__(
        self,
        *,
        transport: GoogleWorkspaceTransport,
    ) -> None:
        if not isinstance(transport, GoogleWorkspaceTransport):
            raise IntegrationConfigurationError(_UNEXPECTED_RESPONSE_MESSAGE)
        self._transport = transport

    def read_document(
        self,
        *,
        document_id: str,
    ) -> GoogleDocsDocument:
        validated_id = _validate_document_id_for_request(document_id)
        encoded_id = quote(validated_id, safe="")
        try:
            payload = self._transport.get_json(
                source_kind=GoogleWorkspaceSourceKind.DOCS,
                relative_path=f"/documents/{encoded_id}",
                params={
                    "includeTabsContent": True,
                    "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
                },
            )
        except GoogleWorkspaceApiError:
            raise
        except Exception:
            raise IntegrationDependencyError(_REQUEST_FAILED_MESSAGE) from None

        if not isinstance(payload, dict):
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            return _parse_document(payload, requested_id=validated_id)
        except Exception:
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None
