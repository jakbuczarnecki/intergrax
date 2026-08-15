# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.core.distribution import (
    DistributionPackageIdentity,
    normalize_distribution_package_name,
    normalize_package_version,
    package_identities_conflict,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Acme-Intergrax", "acme-intergrax"),
        ("  acme_intergrax  ", "acme-intergrax"),
    ],
)
def test_normalize_distribution_package_name_canonicalizes(raw: str, expected: str) -> None:
    assert normalize_distribution_package_name(raw) == expected


@pytest.mark.parametrize("name", ["!!!"])
def test_normalize_distribution_package_name_rejects_invalid_name(name: str) -> None:
    with pytest.raises(ValueError, match="invalid package name"):
        normalize_distribution_package_name(name)


@pytest.mark.parametrize("name", ["", "   "])
def test_normalize_distribution_package_name_rejects_empty(name: str) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        normalize_distribution_package_name(name)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.0", "1.0"),
        ("1.0.0", "1.0.0"),
        ("1.0rc1", "1.0rc1"),
        ("2.0+local", "2.0+local"),
    ],
)
def test_normalize_package_version_valid_pep440(raw: str, expected: str) -> None:
    assert normalize_package_version(raw) == expected


@pytest.mark.parametrize("version", ["not-a-version"])
def test_normalize_package_version_rejects_invalid_version(version: str) -> None:
    with pytest.raises(ValueError, match="invalid package version"):
        normalize_package_version(version)


@pytest.mark.parametrize("version", ["", "   "])
def test_normalize_package_version_rejects_empty(version: str) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        normalize_package_version(version)


def test_distribution_package_identity_uses_canonical_normalizers() -> None:
    identity = DistributionPackageIdentity(name="Acme-Intergrax", version="1.0.0")
    assert identity.name == "acme-intergrax"
    assert identity.version == "1.0.0"


def test_distribution_package_identity_rejects_invalid_fields() -> None:
    with pytest.raises(ValidationError):
        DistributionPackageIdentity(name="!!!", version="1.0.0")
    with pytest.raises(ValidationError):
        DistributionPackageIdentity(name="acme-intergrax", version="not-a-version")


def test_package_identities_conflict() -> None:
    left = DistributionPackageIdentity(name="acme-intergrax", version="1.0.0")
    right = DistributionPackageIdentity(name="acme-intergrax", version="2.0.0")
    assert package_identities_conflict(left, right) is True

    other = DistributionPackageIdentity(name="other-plugin", version="1.0.0")
    assert package_identities_conflict(left, other) is False
