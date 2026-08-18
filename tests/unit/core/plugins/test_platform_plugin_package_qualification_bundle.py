# © Artur Czarnecki. All rights reserved.

"""G4B-1B: immutable package qualification bundle contract."""

from __future__ import annotations

import pytest

from intergrax.core.distribution import DistributionPackageIdentity
from intergrax.core.plugins.discovery import EP_POLICY_RULES, EntryPointSpec
from intergrax.core.plugins.platform_qualification import (
    PlatformPluginPackageQualificationBundle,
    PlatformPluginPackageQualificationBundleError,
    PluginDeliverySource,
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
)
from intergrax.core.qualification import QualificationStatus

pytestmark = pytest.mark.unit

_PACKAGE_NAME = "alpha-policy-plugin"
_PACKAGE_VERSION = "1.0.0"
_OTHER_VERSION = "2.0.0"


def _identity(
    *,
    name: str = _PACKAGE_NAME,
    version: str = _PACKAGE_VERSION,
) -> DistributionPackageIdentity:
    return DistributionPackageIdentity(name=name, version=version)


def _package_qualification(
    *,
    name: str = _PACKAGE_NAME,
    version: str = _PACKAGE_VERSION,
    status: QualificationStatus = QualificationStatus.PRODUCTION_QUALIFIED,
) -> tuple[DistributionPackageIdentity, object]:
    identity = _identity(name=name, version=version)
    qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=identity.name,
            package_version=identity.version,
        ),
        status=status,
        evidence=(),
        reason="fixture qualification",
    )
    return identity, qualification


def _entry_point_spec(
    *,
    distribution: str | None = _PACKAGE_NAME,
) -> EntryPointSpec:
    return EntryPointSpec(
        name="alpha",
        value="tests.unit.core.plugins.test_platform_plugin_package_qualification_bundle:_Handler",
        group=EP_POLICY_RULES,
        distribution=distribution,
    )


def test_exact_package_identity_lookup_succeeds() -> None:
    identity, qualification = _package_qualification()
    bundle = PlatformPluginPackageQualificationBundle([(identity, qualification)])

    assert bundle.lookup_for_package(identity) is qualification


def test_lookup_for_entry_point_resolves_exact_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, qualification = _package_qualification()
    bundle = PlatformPluginPackageQualificationBundle([(identity, qualification)])
    spec = _entry_point_spec()
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_entry_point_distribution_identity",
        lambda _spec: identity,
    )
    assert bundle.lookup_for_entry_point(spec) is qualification


def test_unknown_package_returns_none() -> None:
    identity, qualification = _package_qualification()
    bundle = PlatformPluginPackageQualificationBundle([(identity, qualification)])

    unknown = _identity(name="other-package")
    assert bundle.lookup_for_package(unknown) is None


def test_version_mismatch_returns_none() -> None:
    identity, qualification = _package_qualification()
    bundle = PlatformPluginPackageQualificationBundle([(identity, qualification)])

    wrong_version = _identity(version=_OTHER_VERSION)
    assert bundle.lookup_for_package(wrong_version) is None


def test_qualification_subject_key_mismatch_rejects_construction() -> None:
    identity = _identity()
    _, qualification = _package_qualification(version=_OTHER_VERSION)
    with pytest.raises(PlatformPluginPackageQualificationBundleError, match="version"):
        PlatformPluginPackageQualificationBundle([(identity, qualification)])


def test_non_package_subject_rejects_construction() -> None:
    identity = _identity()
    qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.DOMAIN,
            package_name=identity.name,
            package_version=identity.version,
            domain="policy",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="wrong level",
    )
    with pytest.raises(PlatformPluginPackageQualificationBundleError, match="package-level"):
        PlatformPluginPackageQualificationBundle([(identity, qualification)])


def test_non_external_delivery_source_rejects_construction() -> None:
    from dataclasses import replace

    identity = _identity()
    subject = build_external_package_subject(
        level=PluginQualificationLevel.PACKAGE,
        package_name=identity.name,
        package_version=identity.version,
    )
    qualification = build_qualification_result(
        subject=replace(
            subject,
            delivery_source=PluginDeliverySource.HOST_EMBEDDED_EXTENSION,
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="wrong delivery source",
    )
    with pytest.raises(
        PlatformPluginPackageQualificationBundleError,
        match="external-package",
    ):
        PlatformPluginPackageQualificationBundle([(identity, qualification)])


def test_duplicate_exact_identity_rejects_construction() -> None:
    first = _package_qualification()
    second = _package_qualification()
    with pytest.raises(PlatformPluginPackageQualificationBundleError, match="duplicate"):
        PlatformPluginPackageQualificationBundle([first, second])


def test_snapshot_immune_to_caller_input_mutation() -> None:
    entries: list[tuple[DistributionPackageIdentity, object]] = [
        _package_qualification(),
    ]
    bundle = PlatformPluginPackageQualificationBundle(entries)
    original = bundle.lookup_for_package(entries[0][0])
    entries.clear()
    assert bundle.lookup_for_package(_identity()) is original


def test_exposed_mapping_is_not_mutable() -> None:
    identity, qualification = _package_qualification()
    bundle = PlatformPluginPackageQualificationBundle([(identity, qualification)])
    with pytest.raises(TypeError):
        bundle._qualifications[identity.name, identity.version] = qualification  # type: ignore[index]
