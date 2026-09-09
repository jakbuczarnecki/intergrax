"""Typed handoff from 5C4G validation to 5C4H descriptor generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    DataPackValidationVerdict,
)


@dataclass(frozen=True, slots=True)
class DataPackDescriptorValidationGate:
    verdict: DataPackValidationVerdict
    finalized_artifact_valid: bool
    relational_shard_count: int
    embedding_shard_count: int
    expected_record_count: int
    observed_relational_count: int
    observed_embedding_count: int
    artifact_root: str


def load_descriptor_validation_gate(path: Path) -> DataPackDescriptorValidationGate:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VpiDataPackageDescriptorBuildError(
            f"failed to read validation report: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise VpiDataPackageDescriptorBuildError("validation report must be a JSON object")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise VpiDataPackageDescriptorBuildError("validation report summary is required")

    verdict_raw = payload.get("verdict")
    if not isinstance(verdict_raw, str):
        raise VpiDataPackageDescriptorBuildError("validation report verdict is required")
    try:
        verdict = DataPackValidationVerdict(verdict_raw)
    except ValueError as exc:
        raise VpiDataPackageDescriptorBuildError(
            f"invalid validation verdict: {verdict_raw}"
        ) from exc

    return DataPackDescriptorValidationGate(
        verdict=verdict,
        finalized_artifact_valid=_require_bool(summary, "finalized_artifact_valid"),
        relational_shard_count=_require_int(summary, "relational_shard_count"),
        embedding_shard_count=_require_int(summary, "embedding_shard_count"),
        expected_record_count=_require_int(summary, "expected_record_count"),
        observed_relational_count=_require_int(summary, "observed_relational_count"),
        observed_embedding_count=_require_int(summary, "observed_embedding_count"),
        artifact_root=_require_str(summary, "artifact_root"),
    )


def assert_descriptor_generation_preconditions(
    gate: DataPackDescriptorValidationGate,
    artifact_root: Path,
) -> None:
    if gate.verdict is not DataPackValidationVerdict.PASS:
        raise VpiDataPackageDescriptorBuildError(
            f"validation verdict must be PASS, got {gate.verdict.value}"
        )
    if not gate.finalized_artifact_valid:
        raise VpiDataPackageDescriptorBuildError("finalized_artifact_valid must be true")
    if gate.relational_shard_count != gate.embedding_shard_count:
        raise VpiDataPackageDescriptorBuildError("relational and embedding shard counts must match")
    if gate.observed_relational_count != gate.expected_record_count:
        raise VpiDataPackageDescriptorBuildError("observed relational count mismatch")
    if gate.observed_embedding_count != gate.expected_record_count:
        raise VpiDataPackageDescriptorBuildError("observed embedding count mismatch")
    if gate.artifact_root != str(artifact_root.resolve()):
        raise VpiDataPackageDescriptorBuildError(
            "validation report artifact_root does not match descriptor artifact_root"
        )


def _require_str(payload: dict[str, str | int | bool | list[object] | dict[str, str | int | bool]], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VpiDataPackageDescriptorBuildError(f"validation report {key} is required")
    return value


def _require_int(payload: dict[str, str | int | bool | list[object] | dict[str, str | int | bool]], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise VpiDataPackageDescriptorBuildError(f"validation report {key} must be an integer")
    return value


def _require_bool(payload: dict[str, str | int | bool | list[object] | dict[str, str | int | bool]], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise VpiDataPackageDescriptorBuildError(f"validation report {key} must be a boolean")
    return value
