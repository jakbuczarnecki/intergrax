"""Strict decoding for selected-dataset manifest JSON."""

from __future__ import annotations

import json
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
    VpiDataPackFormatError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.json_decode import (
    JsonValue,
    require_int,
    require_mapping,
    require_sha256_hex,
    require_str,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    SourceDatasetIdentity,
)

def decode_dataset_manifest_payload(payload: dict[str, JsonValue]) -> SourceDatasetIdentity:
    require_mapping(payload, field_name="dataset manifest")
    try:
        return SourceDatasetIdentity(
            dataset_name=require_str(payload, "source_dataset_name"),
            dataset_path=require_str(payload, "output_path"),
            dataset_sha256=require_sha256_hex(payload, "output_sha256"),
            dataset_record_count=require_int(payload, "selected_record_count", minimum=1),
        )
    except VpiDataPackFormatError as exc:
        raise VpiDataPackBuildError(str(exc)) from exc


def load_dataset_identity(dataset_manifest_path: Path) -> SourceDatasetIdentity:
    try:
        raw_payload = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VpiDataPackBuildError("dataset manifest is not valid JSON") from exc
    if not isinstance(raw_payload, dict):
        raise VpiDataPackBuildError("dataset manifest must be a JSON object")
    return decode_dataset_manifest_payload(raw_payload)
