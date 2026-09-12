"""Logical dataset manifest access for reference provisioning — JSON only, no storage vendor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResolvedScenarioVariant:
    variant_id: str
    relative_path: str
    logical_fingerprint: str


@dataclass(frozen=True, slots=True)
class DatasetManifestResolution:
    qualification_id: str
    scenario_slug: str
    schema_version: str
    variant: ResolvedScenarioVariant


class DatasetManifestError(Exception):
    """Base error while reading logical dataset manifest (reference layer only)."""


class InvalidDatasetError(DatasetManifestError):
    """Manifest or shared entity graph failed validation."""


class MissingScenarioVariantError(DatasetManifestError):
    """Requested variant is absent from manifest or variant slice file."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_variant_from_manifest(
    *,
    dataset_package_root: Path,
    qualification_id: str,
    scenario_slug: str,
    variant_id: str,
) -> DatasetManifestResolution:
    manifest_path = dataset_package_root / "manifest.json"
    if not manifest_path.is_file():
        raise InvalidDatasetError(f"manifest missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidDatasetError(f"manifest is not valid JSON: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise InvalidDatasetError("manifest root must be an object")

    manifest_qual = manifest.get("qualification_id")
    manifest_slug = manifest.get("scenario_slug")
    schema_version = manifest.get("schema_version")
    variants = manifest.get("variants")

    if manifest_qual != qualification_id or manifest_slug != scenario_slug:
        raise InvalidDatasetError(
            "manifest identity does not match provisioning context "
            f"({manifest_qual!r}/{manifest_slug!r})"
        )
    if not isinstance(schema_version, str) or not schema_version:
        raise InvalidDatasetError("manifest schema_version must be a non-empty string")
    if not isinstance(variants, list):
        raise InvalidDatasetError("manifest variants must be a list")

    variant_entry: dict[str, object] | None = None
    for entry in variants:
        if not isinstance(entry, dict):
            continue
        if entry.get("variant_id") == variant_id:
            variant_entry = entry
            break

    if variant_entry is None:
        raise MissingScenarioVariantError(f"variant not listed in manifest: {variant_id}")

    relative_path = variant_entry.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise InvalidDatasetError(f"variant path invalid for {variant_id}")

    variant_path = dataset_package_root / relative_path
    if not variant_path.is_file():
        raise MissingScenarioVariantError(f"variant slice missing: {variant_path}")

    try:
        variant_document = json.loads(variant_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidDatasetError(f"variant slice is not valid JSON: {variant_path}") from exc

    if not isinstance(variant_document, dict):
        raise InvalidDatasetError("variant slice root must be an object")
    if variant_document.get("variant_id") != variant_id:
        raise InvalidDatasetError(
            f"variant slice id mismatch: expected {variant_id!r}, "
            f"found {variant_document.get('variant_id')!r}"
        )

    fingerprint_material = json.dumps(
        {
            "schema_version": schema_version,
            "qualification_id": qualification_id,
            "scenario_slug": scenario_slug,
            "variant_id": variant_id,
            "variant_path": relative_path,
        },
        sort_keys=True,
    )
    logical_fingerprint = _sha256_text(fingerprint_material)

    return DatasetManifestResolution(
        qualification_id=qualification_id,
        scenario_slug=scenario_slug,
        schema_version=schema_version,
        variant=ResolvedScenarioVariant(
            variant_id=variant_id,
            relative_path=relative_path,
            logical_fingerprint=logical_fingerprint,
        ),
    )
