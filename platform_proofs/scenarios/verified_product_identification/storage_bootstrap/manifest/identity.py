"""Dataset identity resolution for bootstrap manifest compatibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DatasetVerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapConfigurationError,
    VpiBootstrapDataError,
)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_path: str
    dataset_checksum: str
    dataset_record_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VpiBootstrapDataError(f"failed to read dataset manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise VpiBootstrapDataError("dataset manifest must be a JSON object")
    return payload


def resolve_dataset_identity(
    *,
    dataset_path: Path,
    dataset_manifest_path: Path | None,
    verification_mode: DatasetVerificationMode,
) -> DatasetIdentity:
    if not dataset_path.is_file():
        raise VpiBootstrapConfigurationError(f"dataset path does not exist: {dataset_path}")

    manifest_payload: dict[str, object] | None = None
    if dataset_manifest_path is not None and dataset_manifest_path.is_file():
        manifest_payload = _load_manifest_json(dataset_manifest_path)

    if verification_mode is DatasetVerificationMode.FULL:
        checksum = _sha256_file(dataset_path)
        record_count = _record_count_from_manifest(manifest_payload) if manifest_payload else None
        if record_count is None:
            raise VpiBootstrapDataError(
                "FULL dataset verification requires a trusted manifest with selected_record_count"
            )
        return DatasetIdentity(
            dataset_path=str(dataset_path),
            dataset_checksum=checksum,
            dataset_record_count=record_count,
        )

    if manifest_payload is None:
        raise VpiBootstrapConfigurationError(
            "FAST dataset verification requires dataset manifest with output_sha256 "
            "and selected_record_count"
        )

    checksum_raw = manifest_payload.get("output_sha256")
    count_raw = manifest_payload.get("selected_record_count")
    if not isinstance(checksum_raw, str) or not checksum_raw.strip():
        raise VpiBootstrapDataError("dataset manifest missing output_sha256")
    if not isinstance(count_raw, int) or count_raw <= 0:
        raise VpiBootstrapDataError("dataset manifest missing selected_record_count")

    return DatasetIdentity(
        dataset_path=str(dataset_path),
        dataset_checksum=checksum_raw.strip(),
        dataset_record_count=count_raw,
    )


def _record_count_from_manifest(manifest_payload: dict[str, object] | None) -> int | None:
    if manifest_payload is None:
        return None
    count_raw = manifest_payload.get("selected_record_count")
    if isinstance(count_raw, int) and count_raw > 0:
        return count_raw
    return None
