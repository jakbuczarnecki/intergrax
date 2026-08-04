# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite import google_workspace
from intergrax.integrations.providers.collaboration_suite.google_workspace.config import (
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
    GOOGLE_SHEETS_NATIVE_MIME_TYPE,
    GOOGLE_SHEETS_SOURCE_KIND,
    GoogleSheetsCellValueKind,
    GoogleSheetsKnowledgeReader,
    GoogleSheetsNumberFormatType,
    GoogleSheetsRecalculationInterval,
    GoogleSheetsSheetType,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GoogleSheetsCell,
    GoogleSheetsCellError,
    GoogleSheetsCellValue,
    GoogleSheetsGridData,
    GoogleSheetsGridRange,
    GoogleSheetsNamedRange,
    GoogleSheetsRow,
    GoogleSheetsSheet,
    GoogleSheetsSpreadsheet,
    _GOOGLE_SHEETS_SPREADSHEET_FIELDS,
    _MAX_CELL_TEXT_LENGTH,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)

_UNEXPECTED_MESSAGE = "unexpected Google Sheets provider response"
_INVALID_ID_MESSAGE = "invalid Google Sheets spreadsheet identifier"
_REQUEST_FAILED_MESSAGE = "Google Sheets provider request failed"

_SPREADSHEET_ID = "sheet-main-1"
_SPREADSHEET_TITLE = "Structured Spreadsheet"
_LOCALE = "en_US"
_TIME_ZONE = "America/New_York"


@dataclass
class _RecordingTransport:
    responses: list[dict[str, object]] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)
    exception: Exception | None = None

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            }
        )
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            return {}
        return self.responses.pop(0)


def _assert_safe_dependency_error(
    exc_info: pytest.ExceptionInfo[IntegrationDependencyError],
) -> None:
    assert exc_info.value.__cause__ is None
    rendered = str(exc_info.value)
    assert _SPREADSHEET_ID not in rendered
    assert "Bearer" not in rendered
    assert "access_token" not in rendered


def _reader_with_payload(
    payload: dict[str, object],
) -> tuple[GoogleSheetsKnowledgeReader, _RecordingTransport]:
    transport = _RecordingTransport(responses=[payload])
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    return reader, transport


def _grid_sheet_payload(
    *,
    sheet_id: int = 100,
    title: str = "GridSheet",
    index: int = 0,
    row_count: int = 10,
    column_count: int = 10,
    data: list[dict[str, object]] | None = None,
    merges: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sheet: dict[str, object] = {
        "properties": {
            "sheetId": sheet_id,
            "title": title,
            "index": index,
            "sheetType": "GRID",
            "gridProperties": {
                "rowCount": row_count,
                "columnCount": column_count,
                "frozenRowCount": 1,
                "frozenColumnCount": 1,
            },
            "hidden": True,
            "rightToLeft": True,
        },
    }
    if data is not None:
        sheet["data"] = data
    if merges is not None:
        sheet["merges"] = merges
    return sheet


def _grid_sheet_payload_with_options(
    *,
    sheet_id: int = 100,
    title: str = "GridSheet",
    index: int = 0,
    row_count: int = 10,
    column_count: int = 10,
    data: list[dict[str, object]] | None = None,
    merges: list[dict[str, object]] | None = None,
    sheet_type_null: bool = False,
    hidden_null: bool = False,
    rtl_null: bool = False,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "sheetId": sheet_id,
        "title": title,
        "index": index,
        "gridProperties": {
            "rowCount": row_count,
            "columnCount": column_count,
            "frozenRowCount": 1,
            "frozenColumnCount": 1,
        },
    }
    if sheet_type_null:
        properties["sheetType"] = None
    else:
        properties["sheetType"] = "GRID"
    if hidden_null:
        properties["hidden"] = None
    else:
        properties["hidden"] = True
    if rtl_null:
        properties["rightToLeft"] = None
    else:
        properties["rightToLeft"] = True
    sheet: dict[str, object] = {"properties": properties}
    if data is not None:
        sheet["data"] = data
    if merges is not None:
        sheet["merges"] = merges
    return sheet


def _success_row_values() -> list[dict[str, object]]:
    return [
        {"userEnteredValue": {"stringValue": "hello"}, "effectiveValue": {"stringValue": "hello"}},
        {"userEnteredValue": {"numberValue": 42}, "effectiveValue": {"numberValue": 42}},
        {"userEnteredValue": {"numberValue": 3.14}, "effectiveValue": {"numberValue": 3.14}},
        {"userEnteredValue": {"boolValue": True}, "effectiveValue": {"boolValue": True}},
        {
            "userEnteredValue": {"formulaValue": "=SUM(1,2)"},
            "effectiveValue": {"numberValue": 3},
            "formattedValue": "3",
        },
        {
            "userEnteredValue": {"numberValue": 99.5},
            "effectiveValue": {"numberValue": 99.5},
            "formattedValue": "$99.50",
            "effectiveFormat": {
                "numberFormat": {"type": "CURRENCY", "pattern": "$#,##0.00"},
            },
        },
        {
            "userEnteredValue": {"numberValue": 45000},
            "effectiveValue": {"numberValue": 45000},
            "formattedValue": "1/15/2024",
            "effectiveFormat": {
                "numberFormat": {"type": "DATE", "pattern": "M/d/yyyy"},
            },
            "note": "line one\nline two",
        },
        {
            "userEnteredValue": {"formulaValue": "=1/0"},
            "effectiveValue": {
                "errorValue": {"type": "DIVIDE_BY_ZERO", "message": "division by zero"},
            },
        },
        {},
    ]


def _success_spreadsheet_payload() -> dict[str, object]:
    grid_data = [
        {
            "startRow": 0,
            "startColumn": 0,
            "rowData": [{"values": _success_row_values()}],
        },
    ]
    merges = [
        {
            "sheetId": 100,
            "startRowIndex": 0,
            "endRowIndex": 2,
            "startColumnIndex": 0,
            "endColumnIndex": 2,
        },
    ]
    object_sheet = {
        "properties": {
            "sheetId": 101,
            "title": "ObjectSheet",
            "index": 1,
            "sheetType": "OBJECT",
        },
    }
    data_source_sheet = {
        "properties": {
            "sheetId": 102,
            "title": "DataSourceSheet",
            "index": 2,
            "sheetType": "DATA_SOURCE",
            "gridProperties": {"rowCount": 5, "columnCount": 5},
        },
        "data": [
            {
                "rowData": [
                    {
                        "values": [
                            {
                                "effectiveValue": {"stringValue": "displayed"},
                                "formattedValue": "displayed",
                            },
                        ],
                    },
                ],
            },
        ],
    }
    return {
        "spreadsheetId": _SPREADSHEET_ID,
        "properties": {
            "title": _SPREADSHEET_TITLE,
            "locale": _LOCALE,
            "timeZone": _TIME_ZONE,
            "autoRecalc": "ON_CHANGE",
        },
        "sheets": [
            _grid_sheet_payload(data=grid_data, merges=merges),
            object_sheet,
            data_source_sheet,
        ],
        "namedRanges": [
            {
                "namedRangeId": "nr-bounded",
                "name": "BoundedRange",
                "range": {
                    "sheetId": 100,
                    "startRowIndex": 0,
                    "endRowIndex": 3,
                    "startColumnIndex": 0,
                    "endColumnIndex": 2,
                },
            },
            {
                "namedRangeId": "nr-whole",
                "name": "WholeSheet",
                "range": {"sheetId": 100},
            },
        ],
    }


def test_constants() -> None:
    assert GOOGLE_SHEETS_SOURCE_KIND == "sheets"
    assert GOOGLE_SHEETS_NATIVE_MIME_TYPE == "application/vnd.google-apps.spreadsheet"


def test_structured_spreadsheet_success() -> None:
    reader, _ = _reader_with_payload(_success_spreadsheet_payload())
    spreadsheet = reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)

    assert spreadsheet.spreadsheet_id == _SPREADSHEET_ID
    assert spreadsheet.title == _SPREADSHEET_TITLE
    assert spreadsheet.locale == _LOCALE
    assert spreadsheet.time_zone == _TIME_ZONE
    assert spreadsheet.recalculation_interval is GoogleSheetsRecalculationInterval.ON_CHANGE
    assert len(spreadsheet.sheets) == 3

    grid = spreadsheet.sheets[0]
    assert grid.sheet_type is GoogleSheetsSheetType.GRID
    assert grid.sheet_id == 100
    assert grid.index == 0
    assert grid.hidden is True
    assert grid.right_to_left is True
    assert grid.row_count == 10
    assert grid.column_count == 10
    assert grid.frozen_row_count == 1
    assert grid.frozen_column_count == 1
    assert len(grid.merged_ranges) == 1
    merge = grid.merged_ranges[0]
    assert merge.start_row_index == 0
    assert merge.end_row_index == 2

    cells = grid.grid_data[0].rows[0].cells
    assert cells[0].row_index == 0
    assert cells[0].column_index == 0
    assert cells[0].user_entered_value is not None
    assert cells[0].user_entered_value.kind is GoogleSheetsCellValueKind.STRING
    assert cells[0].user_entered_value.text == "hello"

    assert cells[1].user_entered_value is not None
    assert cells[1].user_entered_value.kind is GoogleSheetsCellValueKind.NUMBER
    assert cells[1].user_entered_value.number == 42.0

    assert cells[2].user_entered_value is not None
    assert cells[2].user_entered_value.number == 3.14

    assert cells[3].user_entered_value is not None
    assert cells[3].user_entered_value.boolean is True

    assert cells[4].user_entered_value is not None
    assert cells[4].user_entered_value.kind is GoogleSheetsCellValueKind.FORMULA
    assert cells[4].user_entered_value.text == "=SUM(1,2)"
    assert cells[4].effective_value is not None
    assert cells[4].effective_value.kind is GoogleSheetsCellValueKind.NUMBER
    assert cells[4].effective_value.number == 3.0

    assert cells[5].formatted_value == "$99.50"
    assert cells[5].effective_number_format is not None
    assert cells[5].effective_number_format.format_type is GoogleSheetsNumberFormatType.CURRENCY

    assert cells[6].note == "line one\nline two"
    assert cells[6].effective_number_format is not None
    assert cells[6].effective_number_format.format_type is GoogleSheetsNumberFormatType.DATE

    assert cells[7].effective_value is not None
    assert cells[7].effective_value.kind is GoogleSheetsCellValueKind.ERROR
    assert cells[7].effective_value.error is not None
    assert cells[7].effective_value.error.error_type == "DIVIDE_BY_ZERO"

    assert cells[8].user_entered_value is None
    assert cells[8].effective_value is None

    object_sheet = spreadsheet.sheets[1]
    assert object_sheet.sheet_type is GoogleSheetsSheetType.OBJECT
    assert object_sheet.row_count is None
    assert object_sheet.column_count is None
    assert object_sheet.grid_data == ()
    assert object_sheet.merged_ranges == ()

    data_source = spreadsheet.sheets[2]
    assert data_source.sheet_type is GoogleSheetsSheetType.DATA_SOURCE
    ds_cell = data_source.grid_data[0].rows[0].cells[0]
    assert ds_cell.effective_value is not None
    assert ds_cell.effective_value.text == "displayed"

    assert len(spreadsheet.named_ranges) == 2
    bounded = spreadsheet.named_ranges[0]
    assert bounded.name == "BoundedRange"
    assert bounded.grid_range.end_row_index == 3
    whole = spreadsheet.named_ranges[1]
    assert whole.name == "WholeSheet"
    assert whole.grid_range.start_row_index is None


def test_exact_transport_request() -> None:
    spreadsheet_id = "sheet+special=id"
    payload = _success_spreadsheet_payload()
    payload["spreadsheetId"] = spreadsheet_id
    transport = _RecordingTransport(responses=[payload])
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    reader.read_spreadsheet(spreadsheet_id=spreadsheet_id)
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["source_kind"] is GoogleWorkspaceSourceKind.SHEETS
    assert call["relative_path"] == "/spreadsheets/sheet%2Bspecial%3Did"
    assert call["params"] == {"fields": _GOOGLE_SHEETS_SPREADSHEET_FIELDS}
    assert call["headers"] == {}
    assert "includeGridData" not in call["params"]
    assert "ranges" not in call["params"]


def test_reader_construction_and_invalid_transport() -> None:
    transport = _RecordingTransport()
    GoogleSheetsKnowledgeReader(transport=transport)
    assert transport.calls == []
    with pytest.raises(IntegrationConfigurationError):
        GoogleSheetsKnowledgeReader(transport=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "spreadsheet_id",
    ["", "   ", 123, True, "a\x00b", "a\x7fb", "x" * 1025, "path/segment"],
)
def test_invalid_spreadsheet_id_rejected(spreadsheet_id: object) -> None:
    transport = _RecordingTransport()
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationConfigurationError, match=_INVALID_ID_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=spreadsheet_id)  # type: ignore[arg-type]
    assert transport.calls == []


def test_transport_api_error_propagates() -> None:
    api_error = GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.NOT_FOUND,
        status_code=404,
        retry_after_seconds=None,
        safe_reason="not_found",
        attempts=1,
    )
    transport = _RecordingTransport(exception=api_error)
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    assert exc_info.value is api_error


def test_transport_runtime_error_normalized() -> None:
    transport = _RecordingTransport(exception=RuntimeError("network blew up"))
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match=_REQUEST_FAILED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    assert "network" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_malformed_top_level_response_type() -> None:
    transport = _RecordingTransport()
    transport.get_json = lambda **kwargs: "not-a-dict"  # type: ignore[method-assign, assignment]
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


def test_spreadsheet_id_mismatch_rejected() -> None:
    payload = _success_spreadsheet_payload()
    payload["spreadsheetId"] = "other-id"
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


def test_sensitive_fields_hidden_from_repr_and_frozen() -> None:
    reader, _ = _reader_with_payload(_success_spreadsheet_payload())
    spreadsheet = reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    rendered = repr(spreadsheet)
    assert _SPREADSHEET_ID not in rendered
    assert _SPREADSHEET_TITLE not in rendered
    assert "hello" not in rendered
    assert "=SUM" not in rendered
    cell = spreadsheet.sheets[0].grid_data[0].rows[0].cells[0]
    assert "hello" not in repr(cell)
    error_cell = spreadsheet.sheets[0].grid_data[0].rows[0].cells[7]
    assert "division" not in repr(error_cell)
    with pytest.raises(ValidationError):
        spreadsheet.title = "changed"  # type: ignore[misc]


def test_model_construct_spreadsheet_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=(),
        )


class _TupleSubclass(tuple[GoogleSheetsSheet, ...]):
    pass


def test_spreadsheet_sheets_exact_tuple_accepted() -> None:
    sheet = GoogleSheetsSheet(
        sheet_id=0,
        title="A",
        index=0,
        sheet_type=GoogleSheetsSheetType.GRID,
        row_count=1,
        column_count=1,
    )
    spreadsheet = GoogleSheetsSpreadsheet(
        spreadsheet_id=_SPREADSHEET_ID,
        title=_SPREADSHEET_TITLE,
        locale=_LOCALE,
        time_zone=_TIME_ZONE,
        sheets=(sheet,),
    )
    assert len(spreadsheet.sheets) == 1


def test_spreadsheet_sheets_list_rejected() -> None:
    sheet = GoogleSheetsSheet(
        sheet_id=0,
        title="A",
        index=0,
        sheet_type=GoogleSheetsSheetType.GRID,
        row_count=1,
        column_count=1,
    )
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=[sheet],  # type: ignore[arg-type]
        )


def test_spreadsheet_sheets_none_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=None,  # type: ignore[arg-type]
        )


def test_spreadsheet_sheets_tuple_subclass_rejected() -> None:
    sheet = GoogleSheetsSheet(
        sheet_id=0,
        title="A",
        index=0,
        sheet_type=GoogleSheetsSheetType.GRID,
        row_count=1,
        column_count=1,
    )
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=_TupleSubclass((sheet,)),
        )


def test_cell_value_invariants_direct() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.STRING)
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(
            kind=GoogleSheetsCellValueKind.STRING,
            text="a",
            number=1.0,
        )


def test_cell_value_model_construct_bypass_rejected() -> None:
    malformed = GoogleSheetsCellValue.model_construct(
        kind=GoogleSheetsCellValueKind.STRING,
        number=1.0,
    )
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(**malformed.model_dump())


def test_cell_value_direct_formula_without_equals_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.FORMULA, text="SUM(1)")


def test_cell_value_direct_blank_formula_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.FORMULA, text="   ")


def test_cell_value_direct_oversized_text_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(
            kind=GoogleSheetsCellValueKind.STRING,
            text="x" * (_MAX_CELL_TEXT_LENGTH + 1),
        )


def test_cell_value_direct_unsafe_control_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.STRING, text="a\x00b")


def test_cell_value_direct_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.NUMBER, number=float("nan"))


def test_cell_value_direct_positive_infinity_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.NUMBER, number=float("inf"))


def test_cell_value_direct_negative_infinity_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.NUMBER, number=float("-inf"))


def test_cell_value_direct_negative_zero_normalized() -> None:
    value = GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.NUMBER, number=-0.0)
    assert value.number == 0.0
    assert math.copysign(1.0, value.number) > 0


def test_cell_value_direct_foreign_nested_error_subclass_rejected() -> None:
    class _ForeignCellError(GoogleSheetsCellError):
        pass

    foreign = _ForeignCellError(error_type="ERROR", message="x")
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.ERROR, error=foreign)


class _DictSubclass(dict[str, object]):
    pass


class _ListSubclass(list[object]):
    pass


def test_provider_top_level_dict_subclass_rejected() -> None:
    transport = _RecordingTransport(responses=[_DictSubclass(_minimal_payload())])
    reader = GoogleSheetsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


def test_provider_nested_dict_subclass_rejected() -> None:
    payload = _minimal_payload()
    payload["properties"] = _DictSubclass(dict(payload["properties"]))
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


def test_provider_list_subclass_rejected() -> None:
    payload = _minimal_payload()
    payload["sheets"] = _ListSubclass(list(payload["sheets"]))
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


@pytest.mark.parametrize(
    "mutation",
    [
        {"properties": {"title": _SPREADSHEET_TITLE, "locale": _LOCALE, "timeZone": _TIME_ZONE, "autoRecalc": None}},
        {"sheets": [_grid_sheet_payload_with_options(sheet_type_null=True)]},
        {"sheets": [_grid_sheet_payload_with_options(hidden_null=True)]},
        {"sheets": [_grid_sheet_payload_with_options(rtl_null=True)]},
        {"sheets": [_grid_sheet_payload_with_options(data=[{"startRow": None, "rowData": []}])]},
        {"sheets": [_grid_sheet_payload_with_options(data=[{"startColumn": None, "rowData": []}])]},
        {"sheets": [_grid_sheet_payload_with_options(data=[{"rowData": None}])]},
        {
            "sheets": [
                _grid_sheet_payload_with_options(
                    data=[{"rowData": [{"values": None}]}],
                ),
            ],
        },
        {
            "sheets": [
                _grid_sheet_payload_with_options(
                    data=[{"rowData": [{"values": [{"userEnteredValue": None}]}]}],
                ),
            ],
        },
        {
            "sheets": [
                _grid_sheet_payload_with_options(
                    data=[
                        {
                            "rowData": [
                                {
                                    "values": [
                                        {"effectiveValue": None, "formattedValue": None, "note": None},
                                    ],
                                },
                            ],
                        },
                    ],
                ),
            ],
        },
        {
            "sheets": [
                _grid_sheet_payload_with_options(
                    data=[
                        {
                            "rowData": [
                                {
                                    "values": [
                                        {
                                            "effectiveFormat": {
                                                "numberFormat": {"type": "TEXT", "pattern": None},
                                            },
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                ),
            ],
        },
        {
            "namedRanges": [
                {
                    "namedRangeId": "nr-1",
                    "name": "Range",
                    "range": {
                        "sheetId": 100,
                        "startRowIndex": None,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": 1,
                    },
                },
            ],
        },
    ],
)
def test_explicit_null_optional_provider_fields_rejected(mutation: dict[str, object]) -> None:
    _read_with_mutation(mutation)


@pytest.mark.parametrize(
    "user_entered,effective",
    [
        ({"userEnteredValue": {}}, None),
        ({"userEnteredValue": {"stringValue": "a", "numberValue": 1}}, None),
        ({"userEnteredValue": {"unknown": "x"}}, None),
        ({"userEnteredValue": {"stringValue": None}}, None),
        ({"userEnteredValue": {"numberValue": None}}, None),
        ({"userEnteredValue": {"boolValue": None}}, None),
        ({"userEnteredValue": {"formulaValue": None}}, None),
        ({"effectiveValue": {"errorValue": None}}, None),
        ({"userEnteredValue": {"numberValue": True}}, None),
        ({"userEnteredValue": {"numberValue": "1"}}, None),
        ({"userEnteredValue": {"numberValue": float("nan")}}, None),
        ({"userEnteredValue": {"numberValue": float("inf")}}, None),
        ({"userEnteredValue": {"numberValue": float("-inf")}}, None),
        ({"userEnteredValue": {"formulaValue": "   "}}, None),
        ({"userEnteredValue": {"formulaValue": "SUM(1)"}}, None),
        ({"userEnteredValue": {"formulaValue": "x" * (_MAX_CELL_TEXT_LENGTH + 1)}}, None),
        ({"userEnteredValue": {"errorValue": {"type": "ERROR"}}}, None),
        ({"effectiveValue": {"formulaValue": "=1"}}, None),
        ({"effectiveValue": {"errorValue": {}}}, None),
        ({"effectiveValue": {"errorValue": {"type": "BAD"}}}, None),
        (
            {
                "effectiveValue": {
                    "errorValue": {"type": "ERROR", "message": "x" * 5000},
                },
            },
            None,
        ),
    ],
)
def test_cell_value_provider_failures(
    user_entered: dict[str, object],
    effective: dict[str, object] | None,
) -> None:
    cell_payload: dict[str, object] = dict(user_entered)
    if effective is not None:
        cell_payload.update(effective)
    payload = _success_spreadsheet_payload()
    payload["sheets"][0]["data"] = [
        {"rowData": [{"values": [cell_payload]}]},
    ]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


def _minimal_payload() -> dict[str, object]:
    return {
        "spreadsheetId": _SPREADSHEET_ID,
        "properties": {
            "title": _SPREADSHEET_TITLE,
            "locale": _LOCALE,
            "timeZone": _TIME_ZONE,
        },
        "sheets": [
            _grid_sheet_payload(
                data=[{"rowData": [{"values": [{"userEnteredValue": {"stringValue": "x"}}]}]}],
            ),
        ],
    }


def _read_with_mutation(mutation: dict[str, object]) -> None:
    payload = _minimal_payload()
    payload.update(mutation)
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


@pytest.mark.parametrize(
    "mutation",
    [
        {"spreadsheetId": None},
        {"properties": None},
        {"sheets": None},
        {"sheets": []},
        {"namedRanges": None},
    ],
)
def test_explicit_null_fields_rejected(mutation: dict[str, object]) -> None:
    _read_with_mutation(mutation)


def test_missing_spreadsheet_id_rejected() -> None:
    payload = _minimal_payload()
    del payload["spreadsheetId"]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_blank_title_rejected() -> None:
    payload = _minimal_payload()
    props = dict(payload["properties"])
    props["title"] = "   "
    payload["properties"] = props
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_duplicate_sheet_id_rejected() -> None:
    payload = _minimal_payload()
    sheet2 = _grid_sheet_payload(sheet_id=100, title="Other", index=1)
    payload["sheets"] = [payload["sheets"][0], sheet2]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_object_sheet_with_grid_rejected() -> None:
    payload = _minimal_payload()
    payload["sheets"] = [
        {
            "properties": {
                "sheetId": 1,
                "title": "Obj",
                "index": 0,
                "sheetType": "OBJECT",
                "gridProperties": {"rowCount": 1, "columnCount": 1},
            },
        },
    ]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_object_sheet_without_grid_properties_succeeds() -> None:
    payload = _minimal_payload()
    payload["sheets"] = [
        {
            "properties": {
                "sheetId": 1,
                "title": "ObjectSheet",
                "index": 0,
                "sheetType": "OBJECT",
            },
        },
    ]
    reader, _ = _reader_with_payload(payload)

    spreadsheet = reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)

    object_sheet = spreadsheet.sheets[0]
    assert object_sheet.sheet_type is GoogleSheetsSheetType.OBJECT
    assert object_sheet.row_count is None
    assert object_sheet.column_count is None
    assert object_sheet.frozen_row_count == 0
    assert object_sheet.frozen_column_count == 0
    assert object_sheet.grid_data == ()
    assert object_sheet.merged_ranges == ()


def test_object_sheet_with_null_grid_properties_rejected() -> None:
    payload = _minimal_payload()
    payload["sheets"] = [
        {
            "properties": {
                "sheetId": 1,
                "title": "ObjectSheet",
                "index": 0,
                "sheetType": "OBJECT",
                "gridProperties": None,
            },
        },
    ]
    reader, _ = _reader_with_payload(payload)

    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)

    _assert_safe_dependency_error(exc_info)
    rendered = str(exc_info.value)
    assert "ObjectSheet" not in rendered
    assert "gridProperties" not in rendered
    assert exc_info.value.__cause__ is None


def test_merge_single_cell_rejected() -> None:
    payload = _minimal_payload()
    grid = payload["sheets"][0]
    grid["merges"] = [
        {
            "sheetId": 100,
            "startRowIndex": 0,
            "endRowIndex": 1,
            "startColumnIndex": 0,
            "endColumnIndex": 1,
        },
    ]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_named_range_unknown_sheet_rejected() -> None:
    payload = _minimal_payload()
    payload["namedRanges"] = [
        {
            "namedRangeId": "nr-1",
            "name": "Bad",
            "range": {"sheetId": 999},
        },
    ]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_unknown_top_level_field_rejected() -> None:
    payload = _minimal_payload()
    payload["spreadsheetUrl"] = "https://example.com"
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_sheet_budget_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        sheets as sheets_module,
    )

    monkeypatch.setattr(sheets_module, "_MAX_SHEETS", 0)
    reader, _ = _reader_with_payload(_minimal_payload())
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_text_budget_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        sheets as sheets_module,
    )

    monkeypatch.setattr(sheets_module, "_MAX_TOTAL_TEXT_CHARS", 1)
    reader, _ = _reader_with_payload(_success_spreadsheet_payload())
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_text_budget_spreadsheet_title_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        sheets as sheets_module,
    )

    monkeypatch.setattr(sheets_module, "_MAX_TOTAL_TEXT_CHARS", 5)
    reader, _ = _reader_with_payload(_minimal_payload())
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_text_budget_named_range_names_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        sheets as sheets_module,
    )

    monkeypatch.setattr(sheets_module, "_MAX_TOTAL_TEXT_CHARS", 40)
    payload = _minimal_payload()
    payload["namedRanges"] = [
        {
            "namedRangeId": "nr-1",
            "name": "x" * 30,
            "range": {"sheetId": 100},
        },
    ]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_text_budget_error_type_below_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        sheets as sheets_module,
    )

    payload = _minimal_payload()
    payload["sheets"] = [
        _grid_sheet_payload(
            data=[
                {
                    "rowData": [
                        {
                            "values": [
                                {
                                    "effectiveValue": {
                                        "errorValue": {"type": "DIVIDE_BY_ZERO"},
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        ),
    ]
    base_text_length = sum(
        len(value)
        for value in (
            _SPREADSHEET_ID,
            _SPREADSHEET_TITLE,
            _LOCALE,
            _TIME_ZONE,
            "GridSheet",
        )
    )
    monkeypatch.setattr(
        sheets_module,
        "_MAX_TOTAL_TEXT_CHARS",
        base_text_length + len("DIVIDE_BY_ZERO") - 1,
    )
    reader, _ = _reader_with_payload(payload)

    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)

    _assert_safe_dependency_error(exc_info)
    rendered = str(exc_info.value)
    for private_value in (
        "DIVIDE_BY_ZERO",
        _SPREADSHEET_TITLE,
        "GridSheet",
        "cell payload",
        "Authorization",
        "Bearer",
        "access_token",
        "credential_ref",
    ):
        assert private_value not in rendered
    assert exc_info.value.__cause__ is None


def test_text_budget_error_type_exact_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        sheets as sheets_module,
    )

    payload = _minimal_payload()
    payload["sheets"] = [
        _grid_sheet_payload(
            data=[
                {
                    "rowData": [
                        {
                            "values": [
                                {
                                    "effectiveValue": {
                                        "errorValue": {"type": "DIVIDE_BY_ZERO"},
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        ),
    ]
    base_text_length = sum(
        len(value)
        for value in (
            _SPREADSHEET_ID,
            _SPREADSHEET_TITLE,
            _LOCALE,
            _TIME_ZONE,
            "GridSheet",
        )
    )
    monkeypatch.setattr(
        sheets_module,
        "_MAX_TOTAL_TEXT_CHARS",
        base_text_length + len("DIVIDE_BY_ZERO"),
    )
    reader, _ = _reader_with_payload(payload)

    spreadsheet = reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)

    effective_value = spreadsheet.sheets[0].grid_data[0].rows[0].cells[0].effective_value
    assert effective_value is not None
    assert effective_value.kind is GoogleSheetsCellValueKind.ERROR
    assert effective_value.error is not None
    assert effective_value.error.error_type == "DIVIDE_BY_ZERO"


@pytest.mark.parametrize(
    "data",
    [
        [{"startRow": 10, "rowData": []}],
        [{"startColumn": 10, "rowData": []}],
        [{"startRow": 8, "rowData": [{}, {}, {}]}],
        [{"startRow": 8, "rowData": [{"values": []}, {"values": []}, {"values": []}]}],
        [{"startRow": 0, "startColumn": 10, "rowData": [{"values": []}]}],
    ],
)
def test_empty_grid_coordinate_bounds_rejected(data: list[dict[str, object]]) -> None:
    payload = _minimal_payload()
    payload["sheets"] = [_grid_sheet_payload(row_count=10, column_count=10, data=data)]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    _assert_safe_dependency_error(exc_info)


def _valid_cell_value() -> GoogleSheetsCellValue:
    return GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.STRING, text="a")


def _valid_cell() -> GoogleSheetsCell:
    return GoogleSheetsCell(row_index=0, column_index=0, user_entered_value=_valid_cell_value())


def _valid_row() -> GoogleSheetsRow:
    return GoogleSheetsRow(row_index=0, cells=(_valid_cell(),))


def _valid_grid_data() -> GoogleSheetsGridData:
    return GoogleSheetsGridData(start_row_index=0, start_column_index=0, rows=(_valid_row(),))


def _valid_grid_sheet() -> GoogleSheetsSheet:
    return GoogleSheetsSheet(
        sheet_id=100,
        title="Grid",
        index=0,
        sheet_type=GoogleSheetsSheetType.GRID,
        row_count=10,
        column_count=10,
        grid_data=(_valid_grid_data(),),
    )


def test_nested_model_construct_cell_error_in_cell_value_rejected() -> None:
    malformed = GoogleSheetsCellError.model_construct(error_type="BAD", message="x")
    with pytest.raises(ValidationError):
        GoogleSheetsCellValue(kind=GoogleSheetsCellValueKind.ERROR, error=malformed)


def test_nested_model_construct_cell_value_in_cell_rejected() -> None:
    malformed = GoogleSheetsCellValue.model_construct(
        kind=GoogleSheetsCellValueKind.STRING,
        number=1.0,
    )
    with pytest.raises(ValidationError):
        GoogleSheetsCell(row_index=0, column_index=0, user_entered_value=malformed)


def test_nested_model_construct_cell_in_row_rejected() -> None:
    malformed = GoogleSheetsCell.model_construct(row_index=1, column_index=0)
    with pytest.raises(ValidationError):
        GoogleSheetsRow(row_index=0, cells=(malformed,))


def test_nested_model_construct_row_in_grid_data_rejected() -> None:
    malformed = GoogleSheetsRow.model_construct(row_index=5, cells=())
    with pytest.raises(ValidationError):
        GoogleSheetsGridData(start_row_index=0, start_column_index=0, rows=(malformed,))


def test_nested_model_construct_grid_data_in_sheet_rejected() -> None:
    malformed = GoogleSheetsGridData.model_construct(start_row_index=99, rows=())
    with pytest.raises(ValidationError):
        GoogleSheetsSheet(
            sheet_id=100,
            title="Grid",
            index=0,
            sheet_type=GoogleSheetsSheetType.GRID,
            row_count=10,
            column_count=10,
            grid_data=(malformed,),
        )


def test_nested_model_construct_grid_range_in_named_range_rejected() -> None:
    malformed = GoogleSheetsGridRange.model_construct(sheet_id=100, start_row_index=5, end_row_index=1)
    with pytest.raises(ValidationError):
        GoogleSheetsNamedRange(named_range_id="nr-1", name="Range", grid_range=malformed)


def test_nested_model_construct_sheet_in_spreadsheet_rejected() -> None:
    malformed = GoogleSheetsSheet.model_construct(
        sheet_id=100,
        title="Grid",
        index=5,
        sheet_type=GoogleSheetsSheetType.GRID,
        row_count=10,
        column_count=10,
    )
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=(malformed,),
        )


def test_nested_model_construct_named_range_in_spreadsheet_rejected() -> None:
    malformed = GoogleSheetsNamedRange.model_construct(
        named_range_id="nr-1",
        name="Range",
        grid_range=GoogleSheetsGridRange(sheet_id=999),
    )
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=(_valid_grid_sheet(),),
            named_ranges=(malformed,),
        )


class _ForeignCellValue(GoogleSheetsCellValue):
    pass


class _ForeignCell(GoogleSheetsCell):
    pass


class _ForeignRow(GoogleSheetsRow):
    pass


class _ForeignGridData(GoogleSheetsGridData):
    pass


class _ForeignGridRange(GoogleSheetsGridRange):
    pass


class _ForeignNamedRange(GoogleSheetsNamedRange):
    pass


class _ForeignSheet(GoogleSheetsSheet):
    pass


def test_nested_subclass_cell_value_in_cell_rejected() -> None:
    foreign = _ForeignCellValue(kind=GoogleSheetsCellValueKind.STRING, text="a")
    with pytest.raises(ValidationError):
        GoogleSheetsCell(row_index=0, column_index=0, user_entered_value=foreign)


def test_nested_subclass_cell_in_row_rejected() -> None:
    foreign = _ForeignCell(row_index=0, column_index=0)
    with pytest.raises(ValidationError):
        GoogleSheetsRow(row_index=0, cells=(foreign,))


def test_nested_subclass_row_in_grid_data_rejected() -> None:
    foreign = _ForeignRow(row_index=0, cells=())
    with pytest.raises(ValidationError):
        GoogleSheetsGridData(start_row_index=0, start_column_index=0, rows=(foreign,))


def test_nested_subclass_grid_data_in_sheet_rejected() -> None:
    foreign = _ForeignGridData(start_row_index=0, start_column_index=0, rows=())
    with pytest.raises(ValidationError):
        GoogleSheetsSheet(
            sheet_id=100,
            title="Grid",
            index=0,
            sheet_type=GoogleSheetsSheetType.GRID,
            row_count=10,
            column_count=10,
            grid_data=(foreign,),
        )


def test_nested_subclass_grid_range_in_named_range_rejected() -> None:
    foreign = _ForeignGridRange(sheet_id=100)
    with pytest.raises(ValidationError):
        GoogleSheetsNamedRange(named_range_id="nr-1", name="Range", grid_range=foreign)


def test_nested_subclass_sheet_in_spreadsheet_rejected() -> None:
    foreign = _ForeignSheet(
        sheet_id=100,
        title="Grid",
        index=0,
        sheet_type=GoogleSheetsSheetType.GRID,
        row_count=10,
        column_count=10,
    )
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=(foreign,),
        )


def test_nested_subclass_named_range_in_spreadsheet_rejected() -> None:
    foreign = _ForeignNamedRange(
        named_range_id="nr-1",
        name="Range",
        grid_range=GoogleSheetsGridRange(sheet_id=100),
    )
    with pytest.raises(ValidationError):
        GoogleSheetsSpreadsheet(
            spreadsheet_id=_SPREADSHEET_ID,
            title=_SPREADSHEET_TITLE,
            locale=_LOCALE,
            time_zone=_TIME_ZONE,
            sheets=(_valid_grid_sheet(),),
            named_ranges=(foreign,),
        )


def test_named_range_object_sheet_rejected() -> None:
    payload = _minimal_payload()
    payload["sheets"] = [
        {
            "properties": {
                "sheetId": 101,
                "title": "ObjectSheet",
                "index": 0,
                "sheetType": "OBJECT",
            },
        },
    ]
    payload["namedRanges"] = [
        {
            "namedRangeId": "nr-obj",
            "name": "ObjectRange",
            "range": {"sheetId": 101},
        },
    ]
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_serialized_output_no_secrets() -> None:
    reader, _ = _reader_with_payload(_success_spreadsheet_payload())
    spreadsheet = reader.read_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    dumped = spreadsheet.model_dump()
    rendered = str(dumped)
    assert "spreadsheetUrl" not in rendered
    assert "Authorization" not in rendered
    assert "Bearer" not in rendered
    assert "access_token" not in rendered
    assert "refresh_token" not in rendered
    assert "client_secret" not in rendered
    assert "x-goog-api-key" not in rendered


@dataclass(frozen=True, slots=True)
class _FakeTransport:
    responses: tuple[dict[str, object], ...]
    calls: list[dict[str, object]] = field(default_factory=list)

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
            }
        )
        return self.responses[0]


@dataclass(frozen=True, slots=True)
class _FakeClientFamily:
    _transport: _FakeTransport

    @property
    def transport(self) -> _FakeTransport:
        return self._transport


def test_integration_read_sheets_spreadsheet_delegates_transport() -> None:
    payload = _minimal_payload()
    transport = _FakeTransport(responses=(payload,))
    family = _FakeClientFamily(_transport=transport)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        family,  # type: ignore[arg-type]
        enabled=True,
    )
    spreadsheet = integration.read_sheets_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)
    assert spreadsheet.spreadsheet_id == _SPREADSHEET_ID
    assert len(transport.calls) == 1
    assert transport.calls[0]["source_kind"] is GoogleWorkspaceSourceKind.SHEETS


def test_disabled_integration_read_sheets_spreadsheet_fails() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=False),
    )
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.read_sheets_spreadsheet(spreadsheet_id=_SPREADSHEET_ID)


def test_knowledge_read_package_exports_sheets_symbols() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        GoogleSheetsKnowledgeReader as PackageReader,
        GoogleSheetsSpreadsheet as PackageSpreadsheet,
    )

    assert PackageReader is GoogleSheetsKnowledgeReader
    assert PackageSpreadsheet is GoogleSheetsSpreadsheet
    assert GOOGLE_SHEETS_SOURCE_KIND == "sheets"
    assert GOOGLE_SHEETS_NATIVE_MIME_TYPE == "application/vnd.google-apps.spreadsheet"


def test_lazy_sheets_exports_resolve() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace import (
        GoogleSheetsKnowledgeReader as TopLevelReader,
        GoogleSheetsSpreadsheet as TopLevelSpreadsheet,
    )
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        GoogleSheetsKnowledgeReader as PackageReader,
        GoogleSheetsSpreadsheet as PackageSpreadsheet,
    )

    assert TopLevelReader is PackageReader
    assert TopLevelSpreadsheet is PackageSpreadsheet
    assert google_workspace.GOOGLE_SHEETS_SOURCE_KIND == GOOGLE_SHEETS_SOURCE_KIND


def test_existing_docs_and_drive_exports_remain() -> None:
    public_names = set(google_workspace.__all__)
    assert "GoogleDocsKnowledgeReader" in public_names
    assert "GoogleDriveKnowledgeReader" in public_names
    assert "GoogleSheetsKnowledgeReader" in public_names
