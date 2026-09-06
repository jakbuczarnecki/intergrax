"""Strict selected-dataset manifest decoding tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.dataset_manifest import (
    decode_dataset_manifest_payload,
    load_dataset_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.json_decode import (
    JsonValue,
)

pytestmark = pytest.mark.unit

_CANONICAL_SHA256 = "fc1268a9c4b3e37325919cd127912a67db0a0b6d1943229a2026d9fedff1d998"


def _valid_payload() -> dict[str, JsonValue]:
    return {
        "source_dataset_name": "offers_corpus_all_v2_non_norm",
        "output_path": "/data/selected.parquet",
        "output_sha256": "a" * 64,
        "selected_record_count": 120,
    }


def _realistic_wdc_manifest_payload() -> dict[str, JsonValue]:
    return {
        "builder_version": "verified_product_identification_wdc_builder/1.0.0",
        "source_dataset_name": "offers_corpus_all_v2_non_norm",
        "source_path": "/data/raw/nonnormalized_offersV2",
        "selection_rule": "keyValuePairs != null OR specTableContent != null",
        "source_record_count": 26507210,
        "selected_record_count": 3770377,
        "rejected_record_count": 22736833,
        "malformed_record_count": 0,
        "records_with_key_value_pairs": 2492991,
        "records_with_spec_table_content": 3770377,
        "records_with_both": 2492991,
        "unique_cluster_count": None,
        "unique_cluster_count_skipped_reason": "skipped for this offline builder",
        "output_format": "parquet",
        "compression": "zstd",
        "output_path": "/data/processed/selected_offers.parquet",
        "output_size_bytes": 1838502691,
        "output_sha256": _CANONICAL_SHA256,
        "parquet_representation": {
            "columns": ["record_json"],
            "nested_encoding": "lossless JSON string per record",
        },
        "build_started_at": "2026-09-01T06:16:10.734496+00:00",
        "build_completed_at": "2026-09-01T06:46:45.704149+00:00",
    }


def test_valid_canonical_manifest_passes() -> None:
    identity = decode_dataset_manifest_payload(_valid_payload())
    assert identity.dataset_record_count == 120
    assert identity.dataset_sha256 == "a" * 64


def test_realistic_wdc_manifest_with_provenance_fields_passes() -> None:
    identity = decode_dataset_manifest_payload(_realistic_wdc_manifest_payload())
    assert identity.dataset_name == "offers_corpus_all_v2_non_norm"
    assert identity.dataset_path == "/data/processed/selected_offers.parquet"
    assert identity.dataset_sha256 == _CANONICAL_SHA256
    assert identity.dataset_record_count == 3770377


def test_forward_compatible_unknown_provenance_field_passes() -> None:
    payload = _valid_payload()
    payload["future_provenance_field"] = "some value"
    identity = decode_dataset_manifest_payload(payload)
    assert identity.dataset_record_count == 120


def test_missing_source_dataset_name_fails() -> None:
    payload = _valid_payload()
    del payload["source_dataset_name"]
    with pytest.raises(VpiDataPackBuildError, match="source_dataset_name"):
        decode_dataset_manifest_payload(payload)


def test_missing_output_path_fails() -> None:
    payload = _valid_payload()
    del payload["output_path"]
    with pytest.raises(VpiDataPackBuildError, match="output_path"):
        decode_dataset_manifest_payload(payload)


def test_missing_checksum_fails() -> None:
    payload = _valid_payload()
    del payload["output_sha256"]
    with pytest.raises(VpiDataPackBuildError, match="output_sha256"):
        decode_dataset_manifest_payload(payload)


def test_missing_selected_record_count_fails() -> None:
    payload = _valid_payload()
    del payload["selected_record_count"]
    with pytest.raises(VpiDataPackBuildError, match="selected_record_count"):
        decode_dataset_manifest_payload(payload)


def test_wrong_source_dataset_name_type_fails() -> None:
    payload = _valid_payload()
    payload["source_dataset_name"] = 123
    with pytest.raises(VpiDataPackBuildError, match="source_dataset_name must be a string"):
        decode_dataset_manifest_payload(payload)


def test_wrong_output_path_type_fails() -> None:
    payload = _valid_payload()
    payload["output_path"] = 123
    with pytest.raises(VpiDataPackBuildError, match="output_path must be a string"):
        decode_dataset_manifest_payload(payload)


def test_wrong_output_sha256_type_fails() -> None:
    payload = _valid_payload()
    payload["output_sha256"] = 123
    with pytest.raises(VpiDataPackBuildError, match="output_sha256 must be a string"):
        decode_dataset_manifest_payload(payload)


def test_wrong_selected_count_type_fails() -> None:
    payload = _valid_payload()
    payload["selected_record_count"] = "120"
    with pytest.raises(VpiDataPackBuildError, match="selected_record_count must be an integer"):
        decode_dataset_manifest_payload(payload)


def test_bool_selected_count_fails() -> None:
    payload = _valid_payload()
    payload["selected_record_count"] = True
    with pytest.raises(VpiDataPackBuildError, match="selected_record_count must be an integer"):
        decode_dataset_manifest_payload(payload)


def test_selected_record_count_zero_fails() -> None:
    payload = _valid_payload()
    payload["selected_record_count"] = 0
    with pytest.raises(VpiDataPackBuildError, match="selected_record_count must be >= 1"):
        decode_dataset_manifest_payload(payload)


def test_bad_sha256_fails() -> None:
    payload = _valid_payload()
    payload["output_sha256"] = "not-a-valid-sha256"
    with pytest.raises(VpiDataPackBuildError, match="output_sha256 must be a 64-character"):
        decode_dataset_manifest_payload(payload)


def test_empty_output_path_fails() -> None:
    payload = _valid_payload()
    payload["output_path"] = ""
    with pytest.raises(VpiDataPackBuildError, match="output_path must be a non-empty string"):
        decode_dataset_manifest_payload(payload)


def test_load_dataset_identity_from_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    identity = load_dataset_identity(manifest_path)
    assert identity.dataset_name == "offers_corpus_all_v2_non_norm"


def test_load_dataset_identity_rejects_non_object_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(VpiDataPackBuildError, match="dataset manifest must be a JSON object"):
        load_dataset_identity(manifest_path)


def test_load_dataset_identity_rejects_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(VpiDataPackBuildError, match="dataset manifest is not valid JSON"):
        load_dataset_identity(manifest_path)
