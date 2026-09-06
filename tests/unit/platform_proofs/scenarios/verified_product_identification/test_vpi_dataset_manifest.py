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

pytestmark = pytest.mark.unit


def _valid_payload() -> dict[str, object]:
    return {
        "source_dataset_name": "offers_corpus_all_v2_non_norm",
        "output_path": "/data/selected.parquet",
        "output_sha256": "a" * 64,
        "selected_record_count": 120,
    }


def test_valid_canonical_manifest_passes() -> None:
    identity = decode_dataset_manifest_payload(_valid_payload())
    assert identity.dataset_record_count == 120
    assert identity.dataset_sha256 == "a" * 64


def test_missing_checksum_fails() -> None:
    payload = _valid_payload()
    del payload["output_sha256"]
    with pytest.raises(VpiDataPackBuildError):
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
