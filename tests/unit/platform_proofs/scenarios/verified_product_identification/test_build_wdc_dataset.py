from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.build_wdc_dataset import (
    build_dataset,
    parse_json_object,
    record_is_selected,
    serialize_record,
)

pytestmark = pytest.mark.unit


def _write_ndjson(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_output_records(parquet_path: Path) -> list[dict[str, object]]:
    table = pq.read_table(parquet_path)
    record_json_values = table.column("record_json").to_pylist()
    return [json.loads(value) for value in record_json_values]


def test_record_with_key_value_pairs_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    record = {
        "id": 1,
        "cluster_id": 10,
        "keyValuePairs": {"Voltage": "12V"},
        "specTableContent": None,
        "extra_field": {"nested": [1, 2]},
    }
    _write_ndjson(source, [record])

    result = build_dataset(input_path=source, output_path=output)

    assert result.stats.selected_record_count == 1
    stored = _read_output_records(output)
    assert stored == [record]


def test_record_with_spec_table_content_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    record = {
        "id": 2,
        "specTableContent": "Voltage 12V",
        "keyValuePairs": None,
    }
    _write_ndjson(source, [record])

    result = build_dataset(input_path=source, output_path=output)

    assert result.stats.selected_record_count == 1
    assert _read_output_records(output) == [record]


def test_record_with_both_fields_is_preserved_once(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    record = {
        "id": 3,
        "keyValuePairs": {"Color": "red"},
        "specTableContent": "Color red",
    }
    _write_ndjson(source, [record])

    result = build_dataset(input_path=source, output_path=output)

    assert result.stats.selected_record_count == 1
    assert result.stats.records_with_both == 1
    assert len(_read_output_records(output)) == 1


def test_record_without_rich_fields_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    _write_ndjson(
        source,
        [
            {
                "id": 4,
                "keyValuePairs": None,
                "specTableContent": None,
            }
        ],
    )

    result = build_dataset(input_path=source, output_path=output)

    assert result.stats.selected_record_count == 0
    assert result.stats.rejected_record_count == 1
    assert _read_output_records(output) == []


def test_all_source_fields_and_nested_values_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    record = {
        "id": 5,
        "cluster_id": 99,
        "identifiers": [{"/gtin13": "[123]"}, {"/productID": "[abc]"}],
        "title": "Example",
        "unknown_future_field": {"level": {"deep": True}},
        "keyValuePairs": {"Wattage": "60W"},
        "specTableContent": "Wattage 60W",
    }
    _write_ndjson(source, [record])

    build_dataset(input_path=source, output_path=output)

    stored = _read_output_records(output)[0]
    assert stored == record
    assert stored["identifiers"] == record["identifiers"]
    assert stored["unknown_future_field"] == record["unknown_future_field"]


def test_malformed_input_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    source.write_text(
        "\n".join(
            [
                '{"id": 6, "keyValuePairs": {"A": "1"}}',
                "not-json",
                "",
                "[1,2,3]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_dataset(input_path=source, output_path=output)

    assert result.stats.source_record_count == 4
    assert result.stats.malformed_record_count == 3
    assert result.stats.selected_record_count == 1


def test_manifest_has_expected_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.ndjson"
    output = tmp_path / "output.parquet"
    manifest = tmp_path / "manifest.json"
    _write_ndjson(
        source,
        [
            {"id": 1, "keyValuePairs": {"A": "1"}},
            {"id": 2, "specTableContent": "B"},
            {"id": 3, "keyValuePairs": None, "specTableContent": None},
            "broken",
        ],
    )

    result = build_dataset(
        input_path=source,
        output_path=output,
        manifest_path=manifest,
    )

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["source_record_count"] == 4
    assert manifest_data["selected_record_count"] == 2
    assert manifest_data["rejected_record_count"] == 1
    assert manifest_data["malformed_record_count"] == 1
    assert manifest_data["records_with_key_value_pairs"] == 1
    assert manifest_data["records_with_spec_table_content"] == 1
    assert manifest_data["records_with_both"] == 0
    assert manifest_data["output_sha256"] == result.output_sha256
    assert manifest_data["compression"] == "zstd"


def test_selection_helpers() -> None:
    record = parse_json_object(
        '{"keyValuePairs": {"x": 1}, "specTableContent": null, "id": 1}'
    )
    assert record_is_selected(record) is True
    assert serialize_record(record).startswith('{"keyValuePairs"')
