"""Unit tests for proof data package descriptor contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.proof_data import (
    DataPackageDescriptorError,
    DataPackageFileDescriptor,
    ProofDataPackageDescriptor,
    PublicationStatus,
    load_proof_data_package_descriptor,
    normalize_sha256_hex,
)
from intergrax.proof_data.paths import normalize_relative_path


def test_normalize_sha256_hex_accepts_lowercase() -> None:
    value = "a" * 64
    assert normalize_sha256_hex(value) == value


def test_normalize_sha256_hex_rejects_invalid() -> None:
    with pytest.raises(Exception):
        normalize_sha256_hex("not-a-checksum")


def test_descriptor_rejects_path_traversal() -> None:
    with pytest.raises(Exception):
        DataPackageFileDescriptor(
            relative_path="../escape.bin",
            size_bytes=1,
            sha256="a" * 64,
            role="OTHER",
        )


def test_path_security_rejects_windows_absolute() -> None:
    with pytest.raises(Exception):
        normalize_relative_path("C:/secret.bin")


def test_load_fixture_descriptor(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).resolve().parents[4]
        / "platform_proofs"
        / "scenarios"
        / "verified_product_identification"
        / "data_package"
        / "fixtures"
        / "tiny_v1"
        / "package.json"
    )
    descriptor = load_proof_data_package_descriptor(fixture)
    assert descriptor.package_id == "verified-product-identification"
    assert descriptor.redistribution_status is PublicationStatus.INTERNAL_BUILD


def test_descriptor_extra_fields_forbidden(tmp_path: Path) -> None:
    payload = {
        "schema_version": "intergrax.proof_data_package.v1",
        "package_id": "example",
        "package_version": "1.0.0",
        "description": "test",
        "files": [
            {
                "relative_path": "a.bin",
                "size_bytes": 1,
                "sha256": "a" * 64,
                "role": "OTHER",
            }
        ],
        "redistribution_status": "INTERNAL_BUILD",
        "unexpected": True,
    }
    path = tmp_path / "package.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataPackageDescriptorError):
        load_proof_data_package_descriptor(path)
