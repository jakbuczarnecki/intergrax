from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.profile_selected_dataset import (
    build_profile_document,
    profile_dataset,
    profile_selected_dataset,
    write_profile_json,
)

pytestmark = pytest.mark.unit


def _write_fixture_parquet(path: Path, records: list[dict[str, object]]) -> None:
    serialized = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    table = pa.Table.from_arrays(
        [pa.array(serialized, type=pa.string())],
        schema=pa.schema([("record_json", pa.string())]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


@pytest.fixture
def fixture_records() -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "cluster_id": 100,
            "category": "Electronics",
            "identifiers": [
                {"/gtin13": "[111]"},
                {"/sku": "[sku-1]"},
            ],
            "title": "Phone",
            "description": "A phone",
            "brand": "Acme",
            "price": "10.00",
            "keyValuePairs": {"Brand": "Acme", "Voltage": "5V"},
            "specTableContent": "Voltage 5V",
            "unknown_future_field": True,
        },
        {
            "id": 2,
            "cluster_id": 100,
            "category": "Electronics",
            "identifiers": [{"/mpn": "[mpn-2]"}],
            "title": "   ",
            "description": None,
            "brand": None,
            "price": None,
            "keyValuePairs": {"Brand": "Acme"},
            "specTableContent": "Brand Acme",
        },
        {
            "id": 3,
            "cluster_id": 200,
            "category": None,
            "identifiers": None,
            "title": "Book",
            "description": "Long text",
            "brand": "Publisher",
            "price": "5",
            "keyValuePairs": None,
            "specTableContent": "Pages 100",
        },
        {
            "id": 4,
            "cluster_id": 300,
            "category": "Books",
            "identifiers": [{"/productID": "[book-4]"}],
            "title": "No attrs",
            "description": "",
            "brand": "",
            "price": "",
            "keyValuePairs": {},
            "specTableContent": "Only spec",
        },
        {
            "id": 5,
            "cluster_id": 400,
            "identifiers": [{"/gtin14": "[222]"}, {"/gtin14": "[222-dup]"}],
            "title": "Duplicate gtin entry",
            "keyValuePairs": {"EAN": "222"},
            "specTableContent": "EAN 222",
        },
    ]


def test_profile_counts_categories_clusters_and_kvp(
    tmp_path: Path,
    fixture_records: list[dict[str, object]],
) -> None:
    parquet_path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(parquet_path, fixture_records)

    accumulator, _ = profile_dataset(input_path=parquet_path, batch_size=2)
    profile = build_profile_document(
        input_path=parquet_path,
        accumulator=accumulator,
        profiling_started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        profiling_completed_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        peak_memory_bytes=123,
    )

    assert profile["dataset"]["total_records"] == 5
    assert profile["categories"]["unique_category_count"] == 2
    assert profile["categories"]["null_category_count"] == 1
    assert profile["categories"]["items"][0]["category"] == "Electronics"
    assert profile["categories"]["items"][0]["record_count"] == 2

    clusters = profile["clusters"]
    assert clusters["unique_cluster_count"] == 4
    assert clusters["singleton_cluster_count"] == 3
    assert clusters["multi_offer_cluster_count"] == 1
    assert clusters["records_in_multi_offer_clusters"] == 2
    assert clusters["max_cluster_size"] == 2
    assert clusters["size_distribution"]["2"] == 1

    kvp = profile["key_value_pairs"]
    assert kvp["records_with_key_value_pairs"] == 3
    assert kvp["records_without_key_value_pairs"] == 2
    assert kvp["attribute_count_distribution"]["1"] == 2
    assert kvp["attribute_count_distribution"]["2-5"] == 1
    assert kvp["attribute_count_distribution"]["0"] == 2
    assert kvp["unique_attribute_name_count"] == 3
    top_names = [item["attribute_name"] for item in kvp["attribute_names"][:3]]
    assert top_names == ["Brand", "EAN", "Voltage"]

    identifiers = profile["identifiers"]
    assert identifiers["records_with_identifiers"] == 4
    assert identifiers["records_without_identifiers"] == 1
    assert identifiers["records_with_any_gtin"] == 2
    assert identifiers["records_with_mpn"] == 1
    assert identifiers["records_with_sku"] == 1
    assert identifiers["records_with_product_id"] == 1
    assert identifiers["records_with_multiple_identifier_types"] == 1
    by_key = {item["identifier_key"]: item["record_count"] for item in identifiers["by_key"]}
    assert by_key["/gtin13"] == 1
    assert by_key["/gtin14"] == 1
    assert by_key["/sku"] == 1
    assert by_key["/mpn"] == 1
    assert by_key["/productID"] == 1


def test_profile_top_level_fields_and_string_stats(
    tmp_path: Path,
    fixture_records: list[dict[str, object]],
) -> None:
    parquet_path = tmp_path / "fixture.parquet"
    _write_fixture_parquet(parquet_path, fixture_records)

    accumulator, _ = profile_dataset(input_path=parquet_path)
    profile = build_profile_document(
        input_path=parquet_path,
        accumulator=accumulator,
        profiling_started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        profiling_completed_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        peak_memory_bytes=None,
    )

    fields = profile["top_level_fields"]
    assert fields["unknown_future_field"]["present_count"] == 1
    assert fields["category"]["null_count"] == 1
    assert fields["category"]["missing_count"] == 1

    title = profile["string_fields"]["title"]
    assert title["empty"] == 1
    assert title["non_empty"] == 4
    assert title["min_length"] == 4
    assert title["max_length"] == len("Duplicate gtin entry")

    quality = profile["quality"]
    assert quality["records_with_empty_title"] == 1
    assert quality["records_without_brand"] == 3
    assert quality["records_without_description"] == 3
    assert quality["records_without_price"] == 3
    assert quality["records_without_category"] == 2
    assert quality["records_with_spec_but_no_kvp"] == 2
    assert quality["records_with_kvp_but_no_spec"] == 0
    assert quality["records_with_both_spec_and_kvp"] == 3


def test_profile_contract_violations_and_malformed_records(tmp_path: Path) -> None:
    parquet_path = tmp_path / "fixture.parquet"
    records = [
        {"id": 1, "keyValuePairs": {"A": "1"}, "specTableContent": "A"},
        "not-json",
        [1, 2, 3],
        {
            "id": 2,
            "cluster_id": "bad",
            "identifiers": "bad",
            "keyValuePairs": ["bad"],
            "specTableContent": "ok",
        },
    ]
    serialized = [
        json.dumps(record, ensure_ascii=False)
        if not isinstance(record, str)
        else record
        for record in records
    ]
    table = pa.Table.from_arrays(
        [pa.array(serialized, type=pa.string())],
        schema=pa.schema([("record_json", pa.string())]),
    )
    pq.write_table(table, parquet_path, compression="zstd")

    accumulator, _ = profile_dataset(input_path=parquet_path)
    profile = build_profile_document(
        input_path=parquet_path,
        accumulator=accumulator,
        profiling_started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        profiling_completed_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        peak_memory_bytes=None,
    )

    violations = profile["contract_violations"]
    assert violations["malformed_record_json_count"] == 1
    assert violations["non_object_record_count"] == 1
    assert violations["unexpected_field_type_counts"]["cluster_id:expected_int_or_null_got_str"] == 1
    assert violations["unexpected_field_type_counts"]["identifiers:expected_list_or_null_got_str"] == 1
    assert violations["unexpected_field_type_counts"]["keyValuePairs:expected_dict_or_null_got_list"] == 1
    assert profile["dataset"]["total_records"] == 2


def test_profile_serialization_writes_json(tmp_path: Path, fixture_records: list[dict[str, object]]) -> None:
    parquet_path = tmp_path / "fixture.parquet"
    output_path = tmp_path / "profile.json"
    markdown_path = tmp_path / "profile.md"
    _write_fixture_parquet(parquet_path, fixture_records)

    result = profile_selected_dataset(
        input_path=parquet_path,
        output_path=output_path,
        markdown_path=markdown_path,
        batch_size=2,
    )

    assert output_path.is_file()
    assert markdown_path.is_file()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["dataset"]["total_records"] == result.profile["dataset"]["total_records"]
    assert loaded["profile_version"] == "1.0.0"
