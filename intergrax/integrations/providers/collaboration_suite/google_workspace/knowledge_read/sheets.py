# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Sheets knowledge-read: structured spreadsheet content via shared transport."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
)

GOOGLE_SHEETS_SOURCE_KIND = "sheets"
GOOGLE_SHEETS_NATIVE_MIME_TYPE = "application/vnd.google-apps.spreadsheet"

_GOOGLE_SHEETS_SPREADSHEET_FIELDS = (
    "spreadsheetId,"
    "properties(title,locale,timeZone,autoRecalc),"
    "sheets("
    "properties("
    "sheetId,title,index,sheetType,"
    "gridProperties("
    "rowCount,columnCount,frozenRowCount,frozenColumnCount"
    "),"
    "hidden,rightToLeft"
    "),"
    "data("
    "startRow,startColumn,"
    "rowData(values("
    "userEnteredValue,"
    "effectiveValue,"
    "formattedValue,"
    "note,"
    "effectiveFormat(numberFormat(type,pattern))"
    "))"
    "),"
    "merges"
    "),"
    "namedRanges(namedRangeId,name,range)"
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)

_INVALID_IDENTIFIER_MESSAGE = "invalid Google Sheets spreadsheet identifier"
_UNEXPECTED_RESPONSE_MESSAGE = "unexpected Google Sheets provider response"
_REQUEST_FAILED_MESSAGE = "Google Sheets provider request failed"

_MAX_SPREADSHEET_ID_LENGTH = 1024
_MAX_RESOURCE_ID_LENGTH = 1024
_MAX_TITLE_LENGTH = 4096
_MAX_LOCALE_LENGTH = 128
_MAX_TIME_ZONE_LENGTH = 256

_MAX_CELL_TEXT_LENGTH = 100_000
_MAX_NUMBER_FORMAT_PATTERN_LENGTH = 4096
_MAX_ERROR_MESSAGE_LENGTH = 4096
_MAX_TOTAL_TEXT_CHARS = 4_000_000

_MAX_SHEETS = 256
_MAX_GRID_DATA_BLOCKS = 4096
_MAX_ROWS = 200_000
_MAX_CELLS = 500_000
_MAX_MERGED_RANGES = 100_000
_MAX_NAMED_RANGES = 10_000

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_TEXT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_EXTENDED_VALUE_KEYS = frozenset(
    {"stringValue", "numberValue", "boolValue", "formulaValue", "errorValue"}
)
_ACCEPTED_ERROR_TYPES = frozenset(
    {
        "ERROR",
        "NULL_VALUE",
        "DIVIDE_BY_ZERO",
        "VALUE",
        "REF",
        "NAME",
        "NUM",
        "N_A",
        "LOADING",
    }
)

_SPREADSHEET_ALLOWED_KEYS = frozenset(
    {"spreadsheetId", "properties", "sheets", "namedRanges"}
)
_SPREADSHEET_PROPERTIES_ALLOWED_KEYS = frozenset(
    {"title", "locale", "timeZone", "autoRecalc"}
)
_SHEET_ALLOWED_KEYS = frozenset({"properties", "data", "merges"})
_SHEET_PROPERTIES_ALLOWED_KEYS = frozenset(
    {
        "sheetId",
        "title",
        "index",
        "sheetType",
        "gridProperties",
        "hidden",
        "rightToLeft",
    }
)
_GRID_PROPERTIES_ALLOWED_KEYS = frozenset(
    {"rowCount", "columnCount", "frozenRowCount", "frozenColumnCount"}
)
_GRID_DATA_ALLOWED_KEYS = frozenset({"startRow", "startColumn", "rowData"})
_ROW_DATA_ALLOWED_KEYS = frozenset({"values"})
_CELL_DATA_ALLOWED_KEYS = frozenset(
    {
        "userEnteredValue",
        "effectiveValue",
        "formattedValue",
        "note",
        "effectiveFormat",
    }
)
_EFFECTIVE_FORMAT_ALLOWED_KEYS = frozenset({"numberFormat"})
_NUMBER_FORMAT_ALLOWED_KEYS = frozenset({"type", "pattern"})
_ERROR_VALUE_ALLOWED_KEYS = frozenset({"type", "message"})
_GRID_RANGE_ALLOWED_KEYS = frozenset(
    {
        "sheetId",
        "startRowIndex",
        "endRowIndex",
        "startColumnIndex",
        "endColumnIndex",
    }
)
_NAMED_RANGE_ALLOWED_KEYS = frozenset({"namedRangeId", "name", "range"})


class GoogleSheetsSheetType(StrEnum):
    GRID = "GRID"
    OBJECT = "OBJECT"
    DATA_SOURCE = "DATA_SOURCE"


class GoogleSheetsCellValueKind(StrEnum):
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    FORMULA = "FORMULA"
    ERROR = "ERROR"


class GoogleSheetsNumberFormatType(StrEnum):
    TEXT = "TEXT"
    NUMBER = "NUMBER"
    PERCENT = "PERCENT"
    CURRENCY = "CURRENCY"
    DATE = "DATE"
    TIME = "TIME"
    DATE_TIME = "DATE_TIME"
    SCIENTIFIC = "SCIENTIFIC"


class GoogleSheetsRecalculationInterval(StrEnum):
    ON_CHANGE = "ON_CHANGE"
    MINUTE = "MINUTE"
    HOUR = "HOUR"


_SHEET_TYPE_MAP: dict[str, GoogleSheetsSheetType] = {
    member.value: member for member in GoogleSheetsSheetType
}
_NUMBER_FORMAT_TYPE_MAP: dict[str, GoogleSheetsNumberFormatType] = {
    member.value: member for member in GoogleSheetsNumberFormatType
}
_RECALCULATION_INTERVAL_MAP: dict[str, GoogleSheetsRecalculationInterval | None] = {
    member.value: member for member in GoogleSheetsRecalculationInterval
}
_RECALCULATION_INTERVAL_MAP["RECALCULATION_INTERVAL_UNSPECIFIED"] = None


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


def _validate_spreadsheet_identifier(value: object) -> str:
    if type(value) is not str:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    if len(trimmed) > _MAX_SPREADSHEET_ID_LENGTH:
        raise ValueError(_INVALID_IDENTIFIER_MESSAGE)
    return trimmed


def _validate_spreadsheet_id_for_request(spreadsheet_id: object) -> str:
    try:
        validated = _validate_spreadsheet_identifier(spreadsheet_id)
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


def _validate_bounded_field(value: object, *, max_length: int) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > max_length:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _require_exact_dict(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _require_exact_list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


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


def _require_exact_tuple(value: object) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


def _reject_unknown_fields(mapping: dict[str, object], allowed_keys: frozenset[str]) -> None:
    for key in mapping:
        if key not in allowed_keys:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _count_union_fields(
    mapping: dict[str, object],
    union_keys: frozenset[str],
) -> tuple[str, ...]:
    return tuple(key for key in union_keys if key in mapping)


def _safe_construct(model_cls: type[BaseModel], **kwargs: object) -> BaseModel:
    try:
        return model_cls(**kwargs)
    except Exception:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _reconstruct_exact_model(
    value: object,
    expected_type: type[_ModelT],
) -> _ModelT:
    if type(value) is expected_type:
        return expected_type(**value.model_dump(mode="python"))
    if type(value) is dict:
        try:
            return expected_type(**value)
        except Exception:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE) from None
    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _reconstruct_exact_model_tuple(
    value: object,
    expected_type: type[_ModelT],
) -> tuple[_ModelT, ...]:
    items = _require_exact_tuple(value)
    return tuple(_reconstruct_exact_model(item, expected_type) for item in items)


def _mapping_value_or_default(
    mapping: dict[str, object],
    key: str,
    *,
    default: object,
) -> object:
    if key not in mapping:
        return default
    field_value = mapping[key]
    if field_value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return field_value


def _normalize_direct_number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if type(value) is not float:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not math.isfinite(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return 0.0
    return value


def _validate_direct_cell_text(value: object) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _UNSAFE_TEXT_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > _MAX_CELL_TEXT_LENGTH:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return value


class _ParserBudget:
    def __init__(self) -> None:
        self.sheet_count = 0
        self.grid_data_block_count = 0
        self.row_count = 0
        self.cell_count = 0
        self.merged_range_count = 0
        self.named_range_count = 0
        self.text_chars = 0

    def add_sheet(self) -> None:
        self.sheet_count += 1
        if self.sheet_count > _MAX_SHEETS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_grid_data_block(self) -> None:
        self.grid_data_block_count += 1
        if self.grid_data_block_count > _MAX_GRID_DATA_BLOCKS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_row(self) -> None:
        self.row_count += 1
        if self.row_count > _MAX_ROWS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_cell(self) -> None:
        self.cell_count += 1
        if self.cell_count > _MAX_CELLS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_merged_range(self) -> None:
        self.merged_range_count += 1
        if self.merged_range_count > _MAX_MERGED_RANGES:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_named_range(self) -> None:
        self.named_range_count += 1
        if self.named_range_count > _MAX_NAMED_RANGES:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def add_text(self, length: int) -> None:
        self.text_chars += length
        if self.text_chars > _MAX_TOTAL_TEXT_CHARS:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _validate_preserved_text(
    value: object,
    budget: _ParserBudget,
    *,
    max_length: int,
    allow_unsafe_controls: bool = False,
) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    pattern = _ASCII_CONTROL if not allow_unsafe_controls else _UNSAFE_TEXT_CONTROL
    if pattern.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > max_length:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    budget.add_text(len(value))
    return value


def _normalize_provider_number(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if type(value) is int:
        return float(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return 0.0
        return value
    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _validate_formula_text(value: object, budget: _ParserBudget) -> str:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if not value.startswith("="):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if _UNSAFE_TEXT_CONTROL.search(value):
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if len(value) > _MAX_CELL_TEXT_LENGTH:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    budget.add_text(len(value))
    return value


def _parse_sheet_type_value(value: object, *, key_present: bool) -> GoogleSheetsSheetType:
    if not key_present:
        return GoogleSheetsSheetType.GRID
    if value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value not in _SHEET_TYPE_MAP:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return _SHEET_TYPE_MAP[value]


def _parse_recalculation_interval_value(value: object, *, key_present: bool) -> GoogleSheetsRecalculationInterval | None:
    if not key_present:
        return None
    if value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value not in _RECALCULATION_INTERVAL_MAP:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return _RECALCULATION_INTERVAL_MAP[value]


def _parse_number_format_type(value: object) -> GoogleSheetsNumberFormatType:
    if type(value) is not str:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if value not in _NUMBER_FORMAT_TYPE_MAP:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    return _NUMBER_FORMAT_TYPE_MAP[value]


class GoogleSheetsCellError(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    error_type: str
    message: str | None = Field(default=None, repr=False)

    @field_validator("error_type", mode="before")
    @classmethod
    def _validate_error_type(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if not value.strip():
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if _ASCII_CONTROL.search(value):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if len(value) > _MAX_RESOURCE_ID_LENGTH:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if value not in _ACCEPTED_ERROR_TYPES:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if _UNSAFE_TEXT_CONTROL.search(value):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if len(value) > _MAX_ERROR_MESSAGE_LENGTH:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value


class GoogleSheetsCellValue(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: GoogleSheetsCellValueKind
    text: str | None = Field(default=None, repr=False)
    number: float | None = None
    boolean: bool | None = None
    error: GoogleSheetsCellError | None = Field(default=None, repr=False)

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_direct_cell_text(value)

    @field_validator("number", mode="before")
    @classmethod
    def _validate_number(cls, value: object) -> float | None:
        if value is None:
            return None
        return _normalize_direct_number(value)

    @field_validator("boolean", mode="before")
    @classmethod
    def _validate_boolean(cls, value: object) -> bool | None:
        if value is None:
            return None
        if type(value) is not bool:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value

    @field_validator("error", mode="before")
    @classmethod
    def _validate_error(cls, value: object) -> GoogleSheetsCellError | None:
        if value is None:
            return None
        return _reconstruct_exact_model(value, GoogleSheetsCellError)

    @model_validator(mode="after")
    def _validate_invariants(self) -> GoogleSheetsCellValue:
        if self.kind is GoogleSheetsCellValueKind.STRING:
            if self.text is None or self.number is not None or self.boolean is not None or self.error is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleSheetsCellValueKind.FORMULA:
            if self.text is None or self.number is not None or self.boolean is not None or self.error is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if not self.text.strip():
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if not self.text.startswith("="):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleSheetsCellValueKind.NUMBER:
            if self.number is None or self.text is not None or self.boolean is not None or self.error is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleSheetsCellValueKind.BOOLEAN:
            if self.boolean is None or self.text is not None or self.number is not None or self.error is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        elif self.kind is GoogleSheetsCellValueKind.ERROR:
            if self.error is None or self.text is not None or self.number is not None or self.boolean is not None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleSheetsNumberFormat(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    format_type: GoogleSheetsNumberFormatType
    pattern: str | None = Field(default=None, repr=False)

    @field_validator("pattern", mode="before")
    @classmethod
    def _validate_pattern(cls, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if _UNSAFE_TEXT_CONTROL.search(value):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if len(value) > _MAX_NUMBER_FORMAT_PATTERN_LENGTH:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return value


class GoogleSheetsCell(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    row_index: int
    column_index: int
    user_entered_value: GoogleSheetsCellValue | None = Field(default=None, repr=False)
    effective_value: GoogleSheetsCellValue | None = Field(default=None, repr=False)
    formatted_value: str | None = Field(default=None, repr=False)
    note: str | None = Field(default=None, repr=False)
    effective_number_format: GoogleSheetsNumberFormat | None = None

    @field_validator("row_index", "column_index", mode="before")
    @classmethod
    def _validate_indexes(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("user_entered_value", "effective_value", mode="before")
    @classmethod
    def _validate_nested_values(cls, value: object) -> GoogleSheetsCellValue | None:
        if value is None:
            return None
        return _reconstruct_exact_model(value, GoogleSheetsCellValue)

    @field_validator("formatted_value", "note", mode="before")
    @classmethod
    def _validate_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        return _validate_direct_cell_text(value)

    @field_validator("effective_number_format", mode="before")
    @classmethod
    def _validate_number_format(cls, value: object) -> GoogleSheetsNumberFormat | None:
        if value is None:
            return None
        return _reconstruct_exact_model(value, GoogleSheetsNumberFormat)


class GoogleSheetsRow(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    row_index: int
    cells: tuple[GoogleSheetsCell, ...]

    @field_validator("row_index", mode="before")
    @classmethod
    def _validate_row_index(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("cells", mode="before")
    @classmethod
    def _validate_cells(cls, value: object) -> tuple[GoogleSheetsCell, ...]:
        return _reconstruct_exact_model_tuple(value, GoogleSheetsCell)

    @model_validator(mode="after")
    def _validate_row_invariants(self) -> GoogleSheetsRow:
        previous_column: int | None = None
        for cell in self.cells:
            if cell.row_index != self.row_index:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if previous_column is not None and cell.column_index <= previous_column:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            previous_column = cell.column_index
        return self


class GoogleSheetsGridData(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    start_row_index: int = 0
    start_column_index: int = 0
    rows: tuple[GoogleSheetsRow, ...] = ()

    @field_validator("start_row_index", "start_column_index", mode="before")
    @classmethod
    def _validate_offsets(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("rows", mode="before")
    @classmethod
    def _validate_rows(cls, value: object) -> tuple[GoogleSheetsRow, ...]:
        return _reconstruct_exact_model_tuple(value, GoogleSheetsRow)

    @model_validator(mode="after")
    def _validate_grid_data_invariants(self) -> GoogleSheetsGridData:
        previous_row: int | None = None
        for offset, row in enumerate(self.rows):
            expected_row = self.start_row_index + offset
            if row.row_index != expected_row:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if previous_row is not None and row.row_index <= previous_row:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            previous_row = row.row_index
            previous_column: int | None = None
            for cell_offset, cell in enumerate(row.cells):
                expected_column = self.start_column_index + cell_offset
                if cell.column_index != expected_column:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if previous_column is not None and cell.column_index <= previous_column:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                previous_column = cell.column_index
        return self


class GoogleSheetsGridRange(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    sheet_id: int
    start_row_index: int | None = None
    end_row_index: int | None = None
    start_column_index: int | None = None
    end_column_index: int | None = None

    @field_validator("sheet_id", mode="before")
    @classmethod
    def _validate_sheet_id(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator(
        "start_row_index",
        "end_row_index",
        "start_column_index",
        "end_column_index",
        mode="before",
    )
    @classmethod
    def _validate_optional_indexes(cls, value: object) -> int | None:
        if value is None:
            return None
        return _require_exact_int(value)

    @model_validator(mode="after")
    def _validate_range_order(self) -> GoogleSheetsGridRange:
        if (
            self.start_row_index is not None
            and self.end_row_index is not None
            and self.start_row_index >= self.end_row_index
        ):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if (
            self.start_column_index is not None
            and self.end_column_index is not None
            and self.start_column_index >= self.end_column_index
        ):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return self


class GoogleSheetsNamedRange(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    named_range_id: str
    name: str = Field(repr=False)
    grid_range: GoogleSheetsGridRange

    @field_validator("named_range_id", mode="before")
    @classmethod
    def _validate_named_range_id(cls, value: object) -> str:
        return _validate_resource_identifier(value)

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: object) -> str:
        return _validate_exact_title(value)

    @field_validator("grid_range", mode="before")
    @classmethod
    def _validate_grid_range(cls, value: object) -> GoogleSheetsGridRange:
        return _reconstruct_exact_model(value, GoogleSheetsGridRange)


class GoogleSheetsSheet(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    sheet_id: int
    title: str = Field(repr=False)
    index: int
    sheet_type: GoogleSheetsSheetType
    hidden: bool = False
    right_to_left: bool = False
    row_count: int | None = None
    column_count: int | None = None
    frozen_row_count: int = 0
    frozen_column_count: int = 0
    grid_data: tuple[GoogleSheetsGridData, ...] = Field(default=(), repr=False)
    merged_ranges: tuple[GoogleSheetsGridRange, ...] = ()

    @field_validator("sheet_id", "index", mode="before")
    @classmethod
    def _validate_int_fields(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("frozen_row_count", "frozen_column_count", mode="before")
    @classmethod
    def _validate_frozen_counts(cls, value: object) -> int:
        return _require_exact_int(value)

    @field_validator("title", mode="before")
    @classmethod
    def _validate_title(cls, value: object) -> str:
        return _validate_exact_title(value)

    @field_validator("grid_data", "merged_ranges", mode="before")
    @classmethod
    def _validate_tuple_fields(
        cls,
        value: object,
        info,
    ) -> tuple[object, ...]:
        if info.field_name == "grid_data":
            return _reconstruct_exact_model_tuple(value, GoogleSheetsGridData)
        return _reconstruct_exact_model_tuple(value, GoogleSheetsGridRange)

    @model_validator(mode="after")
    def _validate_sheet_invariants(self) -> GoogleSheetsSheet:
        if self.sheet_type is GoogleSheetsSheetType.OBJECT:
            if (
                self.row_count is not None
                or self.column_count is not None
                or self.frozen_row_count != 0
                or self.frozen_column_count != 0
                or self.grid_data
                or self.merged_ranges
            ):
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        else:
            if self.row_count is None or self.column_count is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if self.row_count <= 0 or self.column_count <= 0:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if self.frozen_row_count > self.row_count or self.frozen_column_count > self.column_count:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            occupied_cells: set[tuple[int, int]] = set()
            for grid_block in self.grid_data:
                if grid_block.start_row_index >= self.row_count:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if grid_block.start_column_index >= self.column_count:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                for row in grid_block.rows:
                    if row.row_index >= self.row_count:
                        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                    for cell in row.cells:
                        if cell.column_index >= self.column_count:
                            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                        coordinate = (cell.row_index, cell.column_index)
                        if coordinate in occupied_cells:
                            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                        occupied_cells.add(coordinate)
            seen_merges: set[tuple[int, int, int, int, int]] = set()
            for merge_range in self.merged_ranges:
                if merge_range.sheet_id != self.sheet_id:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if (
                    merge_range.start_row_index is None
                    or merge_range.end_row_index is None
                    or merge_range.start_column_index is None
                    or merge_range.end_column_index is None
                ):
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                area = (merge_range.end_row_index - merge_range.start_row_index) * (
                    merge_range.end_column_index - merge_range.start_column_index
                )
                if area <= 1:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if merge_range.start_row_index >= self.row_count:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if merge_range.end_row_index > self.row_count:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if merge_range.start_column_index >= self.column_count:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                if merge_range.end_column_index > self.column_count:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                merge_key = (
                    merge_range.sheet_id,
                    merge_range.start_row_index,
                    merge_range.end_row_index,
                    merge_range.start_column_index,
                    merge_range.end_column_index,
                )
                if merge_key in seen_merges:
                    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
                seen_merges.add(merge_key)
        return self


class GoogleSheetsSpreadsheet(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    spreadsheet_id: str = Field(repr=False)
    title: str = Field(repr=False)
    locale: str
    time_zone: str
    recalculation_interval: GoogleSheetsRecalculationInterval | None = None
    sheets: tuple[GoogleSheetsSheet, ...] = Field(repr=False)
    named_ranges: tuple[GoogleSheetsNamedRange, ...] = ()

    @field_validator("spreadsheet_id", mode="before")
    @classmethod
    def _validate_spreadsheet_id(cls, value: object) -> str:
        return _validate_resource_identifier(value)

    @field_validator("title", mode="before")
    @classmethod
    def _validate_title(cls, value: object) -> str:
        return _validate_exact_title(value)

    @field_validator("locale", mode="before")
    @classmethod
    def _validate_locale(cls, value: object) -> str:
        return _validate_bounded_field(value, max_length=_MAX_LOCALE_LENGTH)

    @field_validator("time_zone", mode="before")
    @classmethod
    def _validate_time_zone(cls, value: object) -> str:
        return _validate_bounded_field(value, max_length=_MAX_TIME_ZONE_LENGTH)

    @field_validator("sheets", "named_ranges", mode="before")
    @classmethod
    def _validate_tuple_fields(
        cls,
        value: object,
        info,
    ) -> tuple[object, ...]:
        if info.field_name == "sheets":
            return _reconstruct_exact_model_tuple(value, GoogleSheetsSheet)
        return _reconstruct_exact_model_tuple(value, GoogleSheetsNamedRange)

    @model_validator(mode="after")
    def _validate_spreadsheet_invariants(self) -> GoogleSheetsSpreadsheet:
        if not self.sheets:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen_ids: set[int] = set()
        seen_titles: set[str] = set()
        seen_indexes: set[int] = set()
        sheet_by_id: dict[int, GoogleSheetsSheet] = {}
        for sheet in self.sheets:
            if sheet.sheet_id in seen_ids:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_ids.add(sheet.sheet_id)
            sheet_by_id[sheet.sheet_id] = sheet
            if sheet.title in seen_titles:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_titles.add(sheet.title)
            if sheet.index in seen_indexes:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_indexes.add(sheet.index)
        expected_indexes = list(range(len(self.sheets)))
        actual_indexes = [sheet.index for sheet in self.sheets]
        if actual_indexes != expected_indexes:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if [sheet.index for sheet in self.sheets] != list(range(len(self.sheets))):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        seen_named_ids: set[str] = set()
        seen_named_names: set[str] = set()
        for named_range in self.named_ranges:
            if named_range.named_range_id in seen_named_ids:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_named_ids.add(named_range.named_range_id)
            if named_range.name in seen_named_names:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_named_names.add(named_range.name)
            referenced_sheet = sheet_by_id.get(named_range.grid_range.sheet_id)
            if referenced_sheet is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if referenced_sheet.sheet_type is GoogleSheetsSheetType.OBJECT:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            _validate_range_within_sheet(
                named_range.grid_range,
                row_count=referenced_sheet.row_count,
                column_count=referenced_sheet.column_count,
            )
        return self


def _parse_error_value(
    mapping: dict[str, object],
    budget: _ParserBudget,
) -> GoogleSheetsCellError:
    _reject_unknown_fields(mapping, _ERROR_VALUE_ALLOWED_KEYS)
    error_type = mapping.get("type")
    if error_type is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if "message" not in mapping:
        message: str | None = None
    elif mapping["message"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        message = _validate_preserved_text(
            mapping["message"],
            budget,
            max_length=_MAX_ERROR_MESSAGE_LENGTH,
            allow_unsafe_controls=True,
        )
    return _safe_construct(
        GoogleSheetsCellError,
        error_type=error_type,
        message=message,
    )


def _parse_extended_value(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    allow_formula: bool,
    allow_error: bool,
) -> GoogleSheetsCellValue:
    _reject_unknown_fields(mapping, _EXTENDED_VALUE_KEYS)
    union_fields = _count_union_fields(mapping, _EXTENDED_VALUE_KEYS)
    if len(union_fields) != 1:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    union_key = union_fields[0]
    raw_value = mapping[union_key]
    if raw_value is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    if union_key == "stringValue":
        text = _validate_preserved_text(
            raw_value,
            budget,
            max_length=_MAX_CELL_TEXT_LENGTH,
            allow_unsafe_controls=True,
        )
        return _safe_construct(
            GoogleSheetsCellValue,
            kind=GoogleSheetsCellValueKind.STRING,
            text=text,
        )
    if union_key == "numberValue":
        number = _normalize_provider_number(raw_value)
        return _safe_construct(
            GoogleSheetsCellValue,
            kind=GoogleSheetsCellValueKind.NUMBER,
            number=number,
        )
    if union_key == "boolValue":
        if type(raw_value) is not bool:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return _safe_construct(
            GoogleSheetsCellValue,
            kind=GoogleSheetsCellValueKind.BOOLEAN,
            boolean=raw_value,
        )
    if union_key == "formulaValue":
        if not allow_formula:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        text = _validate_formula_text(raw_value, budget)
        return _safe_construct(
            GoogleSheetsCellValue,
            kind=GoogleSheetsCellValueKind.FORMULA,
            text=text,
        )
    if union_key == "errorValue":
        if not allow_error:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        error_mapping = _require_exact_dict(raw_value)
        error = _parse_error_value(error_mapping, budget)
        return _safe_construct(
            GoogleSheetsCellValue,
            kind=GoogleSheetsCellValueKind.ERROR,
            error=error,
        )
    raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _parse_number_format(
    mapping: dict[str, object],
    budget: _ParserBudget,
) -> GoogleSheetsNumberFormat:
    _reject_unknown_fields(mapping, _NUMBER_FORMAT_ALLOWED_KEYS)
    format_type_raw = mapping.get("type")
    if format_type_raw is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    format_type = _parse_number_format_type(format_type_raw)
    if "pattern" not in mapping:
        pattern: str | None = None
    elif mapping["pattern"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        pattern = _validate_preserved_text(
            mapping["pattern"],
            budget,
            max_length=_MAX_NUMBER_FORMAT_PATTERN_LENGTH,
            allow_unsafe_controls=True,
        )
    return _safe_construct(
        GoogleSheetsNumberFormat,
        format_type=format_type,
        pattern=pattern,
    )


def _validate_range_within_sheet(
    grid_range: GoogleSheetsGridRange,
    *,
    row_count: int | None,
    column_count: int | None,
) -> None:
    if row_count is not None:
        if grid_range.start_row_index is not None and grid_range.start_row_index >= row_count:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if grid_range.end_row_index is not None and grid_range.end_row_index > row_count:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if column_count is not None:
        if grid_range.start_column_index is not None and grid_range.start_column_index >= column_count:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if grid_range.end_column_index is not None and grid_range.end_column_index > column_count:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)


def _parse_grid_range(
    mapping: dict[str, object],
    *,
    expected_sheet_id: int | None = None,
    require_all_boundaries: bool = False,
    row_count: int | None = None,
    column_count: int | None = None,
) -> GoogleSheetsGridRange:
    _reject_unknown_fields(mapping, _GRID_RANGE_ALLOWED_KEYS)
    sheet_id = _require_exact_int(mapping.get("sheetId"))
    if expected_sheet_id is not None and sheet_id != expected_sheet_id:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    def _optional_boundary(key: str) -> int | None:
        if key not in mapping:
            return None
        boundary_value = mapping[key]
        if boundary_value is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        return _require_exact_int(boundary_value)

    start_row_index = _optional_boundary("startRowIndex")
    end_row_index = _optional_boundary("endRowIndex")
    start_column_index = _optional_boundary("startColumnIndex")
    end_column_index = _optional_boundary("endColumnIndex")
    if require_all_boundaries:
        if (
            start_row_index is None
            or end_row_index is None
            or start_column_index is None
            or end_column_index is None
        ):
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        area = (end_row_index - start_row_index) * (end_column_index - start_column_index)
        if area <= 1:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    grid_range = _safe_construct(
        GoogleSheetsGridRange,
        sheet_id=sheet_id,
        start_row_index=start_row_index,
        end_row_index=end_row_index,
        start_column_index=start_column_index,
        end_column_index=end_column_index,
    )
    _validate_range_within_sheet(
        grid_range,
        row_count=row_count,
        column_count=column_count,
    )
    return grid_range


def _parse_cell_data(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    row_index: int,
    column_index: int,
    row_count: int | None,
    column_count: int | None,
) -> GoogleSheetsCell:
    budget.add_cell()
    _reject_unknown_fields(mapping, _CELL_DATA_ALLOWED_KEYS)
    if row_count is not None and row_index >= row_count:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if column_count is not None and column_index >= column_count:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    user_entered: GoogleSheetsCellValue | None = None
    if "userEnteredValue" in mapping:
        user_raw = mapping["userEnteredValue"]
        if user_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        user_mapping = _require_exact_dict(user_raw)
        user_entered = _parse_extended_value(
            user_mapping,
            budget,
            allow_formula=True,
            allow_error=False,
        )

    effective: GoogleSheetsCellValue | None = None
    if "effectiveValue" in mapping:
        effective_raw = mapping["effectiveValue"]
        if effective_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        effective_mapping = _require_exact_dict(effective_raw)
        effective = _parse_extended_value(
            effective_mapping,
            budget,
            allow_formula=False,
            allow_error=True,
        )

    formatted: str | None = None
    if "formattedValue" in mapping:
        formatted_raw = mapping["formattedValue"]
        if formatted_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        formatted = _validate_preserved_text(
            formatted_raw,
            budget,
            max_length=_MAX_CELL_TEXT_LENGTH,
            allow_unsafe_controls=True,
        )

    note: str | None = None
    if "note" in mapping:
        note_raw = mapping["note"]
        if note_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        note = _validate_preserved_text(
            note_raw,
            budget,
            max_length=_MAX_CELL_TEXT_LENGTH,
            allow_unsafe_controls=True,
        )

    number_format: GoogleSheetsNumberFormat | None = None
    if "effectiveFormat" in mapping:
        effective_format_raw = mapping["effectiveFormat"]
        if effective_format_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        effective_format = _require_exact_dict(effective_format_raw)
        _reject_unknown_fields(effective_format, _EFFECTIVE_FORMAT_ALLOWED_KEYS)
        if "numberFormat" in effective_format:
            number_format_raw = effective_format["numberFormat"]
            if number_format_raw is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            number_format_mapping = _require_exact_dict(number_format_raw)
            number_format = _parse_number_format(number_format_mapping, budget)

    return _safe_construct(
        GoogleSheetsCell,
        row_index=row_index,
        column_index=column_index,
        user_entered_value=user_entered,
        effective_value=effective,
        formatted_value=formatted,
        note=note,
        effective_number_format=number_format,
    )


def _parse_grid_data(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    sheet_id: int,
    row_count: int | None,
    column_count: int | None,
    occupied_cells: set[tuple[int, int]],
) -> GoogleSheetsGridData:
    budget.add_grid_data_block()
    _reject_unknown_fields(mapping, _GRID_DATA_ALLOWED_KEYS)
    start_row_raw = _mapping_value_or_default(mapping, "startRow", default=0)
    start_col_raw = _mapping_value_or_default(mapping, "startColumn", default=0)
    start_row = _require_exact_int(start_row_raw)
    start_column = _require_exact_int(start_col_raw)
    if row_count is not None and start_row >= row_count:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    if column_count is not None and start_column >= column_count:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    if "rowData" not in mapping:
        row_data_list: list[object] = []
    elif mapping["rowData"] is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        row_data_list = _require_exact_list(mapping["rowData"])

    parsed_rows: list[GoogleSheetsRow] = []
    previous_row_index: int | None = None
    for row_offset, raw_row in enumerate(row_data_list):
        budget.add_row()
        row_mapping = _require_exact_dict(raw_row)
        _reject_unknown_fields(row_mapping, _ROW_DATA_ALLOWED_KEYS)
        absolute_row = start_row + row_offset
        if row_count is not None and absolute_row >= row_count:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if previous_row_index is not None and absolute_row <= previous_row_index:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        previous_row_index = absolute_row

        if "values" not in row_mapping:
            values_list: list[object] = []
        elif row_mapping["values"] is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        else:
            values_list = _require_exact_list(row_mapping["values"])

        parsed_cells: list[GoogleSheetsCell] = []
        previous_column: int | None = None
        for value_offset, raw_cell in enumerate(values_list):
            absolute_column = start_column + value_offset
            if column_count is not None and absolute_column >= column_count:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            if previous_column is not None and absolute_column <= previous_column:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            previous_column = absolute_column
            coordinate = (absolute_row, absolute_column)
            if coordinate in occupied_cells:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            occupied_cells.add(coordinate)
            cell_mapping = _require_exact_dict(raw_cell)
            cell = _parse_cell_data(
                cell_mapping,
                budget,
                row_index=absolute_row,
                column_index=absolute_column,
                row_count=row_count,
                column_count=column_count,
            )
            if cell.row_index != absolute_row:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            parsed_cells.append(cell)

        parsed_rows.append(
            _safe_construct(
                GoogleSheetsRow,
                row_index=absolute_row,
                cells=tuple(parsed_cells),
            )
        )

    return _safe_construct(
        GoogleSheetsGridData,
        start_row_index=start_row,
        start_column_index=start_column,
        rows=tuple(parsed_rows),
    )


def _parse_sheet_properties(
    mapping: dict[str, object],
    budget: _ParserBudget,
) -> tuple[
    int,
    str,
    int,
    GoogleSheetsSheetType,
    bool,
    bool,
    int | None,
    int | None,
    int,
    int,
]:
    _reject_unknown_fields(mapping, _SHEET_PROPERTIES_ALLOWED_KEYS)
    sheet_id = _require_exact_int(mapping.get("sheetId"))
    title = _validate_preserved_text(
        mapping.get("title"),
        budget,
        max_length=_MAX_TITLE_LENGTH,
    )
    if not title.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    index = _require_exact_int(mapping.get("index"))
    sheet_type = _parse_sheet_type_value(
        mapping["sheetType"] if "sheetType" in mapping else None,
        key_present="sheetType" in mapping,
    )

    hidden_raw = _mapping_value_or_default(mapping, "hidden", default=False)
    if type(hidden_raw) is not bool:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    hidden = hidden_raw

    rtl_raw = _mapping_value_or_default(mapping, "rightToLeft", default=False)
    if type(rtl_raw) is not bool:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    right_to_left = rtl_raw

    row_count: int | None = None
    column_count: int | None = None
    frozen_row_count = 0
    frozen_column_count = 0

    grid_properties_raw = mapping.get("gridProperties")
    if sheet_type is GoogleSheetsSheetType.OBJECT:
        if grid_properties_raw is not None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        if grid_properties_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        grid_properties = _require_exact_dict(grid_properties_raw)
        _reject_unknown_fields(grid_properties, _GRID_PROPERTIES_ALLOWED_KEYS)
        row_count = _require_positive_int(grid_properties.get("rowCount"))
        column_count = _require_positive_int(grid_properties.get("columnCount"))
        frozen_row_raw = _mapping_value_or_default(grid_properties, "frozenRowCount", default=0)
        frozen_col_raw = _mapping_value_or_default(grid_properties, "frozenColumnCount", default=0)
        frozen_row_count = _require_exact_int(frozen_row_raw)
        frozen_column_count = _require_exact_int(frozen_col_raw)

    return (
        sheet_id,
        title,
        index,
        sheet_type,
        hidden,
        right_to_left,
        row_count,
        column_count,
        frozen_row_count,
        frozen_column_count,
    )


def _parse_sheet(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    expected_index: int,
) -> GoogleSheetsSheet:
    budget.add_sheet()
    _reject_unknown_fields(mapping, _SHEET_ALLOWED_KEYS)
    properties_raw = mapping.get("properties")
    if properties_raw is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    properties = _require_exact_dict(properties_raw)
    (
        sheet_id,
        title,
        index,
        sheet_type,
        hidden,
        right_to_left,
        row_count,
        column_count,
        frozen_row_count,
        frozen_column_count,
    ) = _parse_sheet_properties(properties, budget)
    if index != expected_index:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    occupied_cells: set[tuple[int, int]] = set()
    grid_data_blocks: list[GoogleSheetsGridData] = []

    if sheet_type is GoogleSheetsSheetType.OBJECT:
        if "data" in mapping:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        if "merges" in mapping:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    else:
        if "data" in mapping:
            data_raw = mapping["data"]
            if data_raw is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            data_list = _require_exact_list(data_raw)
        else:
            data_list = []
        for raw_grid in data_list:
            grid_mapping = _require_exact_dict(raw_grid)
            grid_data_blocks.append(
                _parse_grid_data(
                    grid_mapping,
                    budget,
                    sheet_id=sheet_id,
                    row_count=row_count,
                    column_count=column_count,
                    occupied_cells=occupied_cells,
                )
            )

        if "merges" in mapping:
            merges_raw = mapping["merges"]
            if merges_raw is None:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            merges_list = _require_exact_list(merges_raw)
        else:
            merges_list = []

    merged_ranges: list[GoogleSheetsGridRange] = []
    seen_merges: set[tuple[int, int, int, int, int]] = set()
    if sheet_type is not GoogleSheetsSheetType.OBJECT:
        for raw_merge in merges_list:
            merge_mapping = _require_exact_dict(raw_merge)
            budget.add_merged_range()
            merge_range = _parse_grid_range(
                merge_mapping,
                expected_sheet_id=sheet_id,
                require_all_boundaries=True,
                row_count=row_count,
                column_count=column_count,
            )
            merge_key = (
                merge_range.sheet_id,
                merge_range.start_row_index or 0,
                merge_range.end_row_index or 0,
                merge_range.start_column_index or 0,
                merge_range.end_column_index or 0,
            )
            if merge_key in seen_merges:
                raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
            seen_merges.add(merge_key)
            merged_ranges.append(merge_range)

    return _safe_construct(
        GoogleSheetsSheet,
        sheet_id=sheet_id,
        title=title,
        index=index,
        sheet_type=sheet_type,
        hidden=hidden,
        right_to_left=right_to_left,
        row_count=row_count,
        column_count=column_count,
        frozen_row_count=frozen_row_count,
        frozen_column_count=frozen_column_count,
        grid_data=tuple(grid_data_blocks),
        merged_ranges=tuple(merged_ranges),
    )


def _parse_named_range(
    mapping: dict[str, object],
    budget: _ParserBudget,
    *,
    sheet_lookup: dict[int, GoogleSheetsSheet],
) -> GoogleSheetsNamedRange:
    budget.add_named_range()
    _reject_unknown_fields(mapping, _NAMED_RANGE_ALLOWED_KEYS)
    named_range_id = _validate_resource_identifier(mapping.get("namedRangeId"))
    budget.add_text(len(named_range_id))
    name = _validate_exact_title(mapping.get("name"))
    budget.add_text(len(name))
    range_raw = mapping.get("range")
    if range_raw is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    range_mapping = _require_exact_dict(range_raw)
    sheet_id = _require_exact_int(range_mapping.get("sheetId"))
    if sheet_id not in sheet_lookup:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    sheet = sheet_lookup[sheet_id]
    grid_range = _parse_grid_range(
        range_mapping,
        row_count=sheet.row_count,
        column_count=sheet.column_count,
    )
    return _safe_construct(
        GoogleSheetsNamedRange,
        named_range_id=named_range_id,
        name=name,
        grid_range=grid_range,
    )


def _parse_spreadsheet(payload: dict[str, object], *, requested_id: str) -> GoogleSheetsSpreadsheet:
    _reject_unknown_fields(payload, _SPREADSHEET_ALLOWED_KEYS)
    budget = _ParserBudget()
    spreadsheet_id = _validate_resource_identifier(payload.get("spreadsheetId"))
    budget.add_text(len(spreadsheet_id))
    if spreadsheet_id != requested_id:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    properties_raw = payload.get("properties")
    if properties_raw is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    properties = _require_exact_dict(properties_raw)
    _reject_unknown_fields(properties, _SPREADSHEET_PROPERTIES_ALLOWED_KEYS)
    title = _validate_preserved_text(
        properties.get("title"),
        budget,
        max_length=_MAX_TITLE_LENGTH,
    )
    if not title.strip():
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    locale = _validate_preserved_text(
        properties.get("locale"),
        budget,
        max_length=_MAX_LOCALE_LENGTH,
    )
    time_zone = _validate_preserved_text(
        properties.get("timeZone"),
        budget,
        max_length=_MAX_TIME_ZONE_LENGTH,
    )
    recalculation_interval = _parse_recalculation_interval_value(
        properties["autoRecalc"] if "autoRecalc" in properties else None,
        key_present="autoRecalc" in properties,
    )

    sheets_raw = payload.get("sheets")
    if sheets_raw is None:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
    sheets_list = _require_exact_list(sheets_raw)
    if not sheets_list:
        raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)

    parsed_sheets: list[GoogleSheetsSheet] = []
    for index, raw_sheet in enumerate(sheets_list):
        sheet_mapping = _require_exact_dict(raw_sheet)
        parsed_sheets.append(_parse_sheet(sheet_mapping, budget, expected_index=index))

    sheet_lookup = {sheet.sheet_id: sheet for sheet in parsed_sheets}

    if "namedRanges" in payload:
        named_ranges_raw = payload["namedRanges"]
        if named_ranges_raw is None:
            raise ValueError(_UNEXPECTED_RESPONSE_MESSAGE)
        named_ranges_list = _require_exact_list(named_ranges_raw)
    else:
        named_ranges_list = []

    parsed_named_ranges: list[GoogleSheetsNamedRange] = []
    for raw_named in named_ranges_list:
        named_mapping = _require_exact_dict(raw_named)
        parsed_named_ranges.append(
            _parse_named_range(named_mapping, budget, sheet_lookup=sheet_lookup)
        )

    return _safe_construct(
        GoogleSheetsSpreadsheet,
        spreadsheet_id=spreadsheet_id,
        title=title,
        locale=locale,
        time_zone=time_zone,
        recalculation_interval=recalculation_interval,
        sheets=tuple(parsed_sheets),
        named_ranges=tuple(parsed_named_ranges),
    )


@runtime_checkable
class GoogleSheetsKnowledgeReadClient(Protocol):
    def read_spreadsheet(
        self,
        *,
        spreadsheet_id: str,
    ) -> GoogleSheetsSpreadsheet:
        ...


class GoogleSheetsKnowledgeReader:
    """Stateless Google Sheets knowledge reader using one shared transport."""

    def __init__(
        self,
        *,
        transport: GoogleWorkspaceTransport,
    ) -> None:
        if not isinstance(transport, GoogleWorkspaceTransport):
            raise IntegrationConfigurationError(_UNEXPECTED_RESPONSE_MESSAGE)
        self._transport = transport

    def read_spreadsheet(
        self,
        *,
        spreadsheet_id: str,
    ) -> GoogleSheetsSpreadsheet:
        validated_id = _validate_spreadsheet_id_for_request(spreadsheet_id)
        encoded_id = quote(validated_id, safe="")
        try:
            payload = self._transport.get_json(
                source_kind=GoogleWorkspaceSourceKind.SHEETS,
                relative_path=f"/spreadsheets/{encoded_id}",
                params={"fields": _GOOGLE_SHEETS_SPREADSHEET_FIELDS},
            )
        except GoogleWorkspaceApiError:
            raise
        except Exception:
            raise IntegrationDependencyError(_REQUEST_FAILED_MESSAGE) from None

        if type(payload) is not dict:
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE)
        try:
            return _parse_spreadsheet(payload, requested_id=validated_id)
        except Exception:
            raise IntegrationDependencyError(_UNEXPECTED_RESPONSE_MESSAGE) from None
