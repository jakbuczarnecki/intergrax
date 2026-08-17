# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform Plugin trust, qualification, and production gate contracts (PLATFORM-PLUGIN-7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from intergrax.core.distribution import (
    DistributionPackageIdentity,
    PlatformCompatibilityResult,
    check_platform_compatibility,
)
from intergrax.core.plugins.discovery import EntryPointSpec
from intergrax.core.plugins.errors import ProductionQualificationRequiredError
from intergrax.core.qualification import QualificationEvidence, QualificationStatus

if TYPE_CHECKING:
    from importlib.metadata import Distribution

    from intergrax.core.plugins.package_contract import PlatformPluginManifest


class PlatformPluginTrustModel(StrEnum):
    """Execution trust classification for Platform Plugin code."""

    TRUSTED_IN_PROCESS = "trusted_in_process"


class PluginTrustOrigin(StrEnum):
    """Optional audit distinction for trusted-in-process code origin."""

    HOST_LOCAL_CODE = "host_local_code"
    INSTALLED_THIRD_PARTY_PACKAGE = "installed_third_party_package"


class PluginDeliverySource(StrEnum):
    """How an extension entered the host — does not imply qualification."""

    EXTERNAL_PACKAGE = "external_package"
    HOST_EMBEDDED_EXTENSION = "host_embedded_extension"


class PluginQualificationLevel(StrEnum):
    """Granularity at which qualification evidence applies."""

    PACKAGE = "package"
    CAPABILITY = "capability"
    DOMAIN = "domain"


class PluginQualificationEvidenceKind(StrEnum):
    """Auditable evidence categories; domains own required thresholds."""

    CONTRACT_VALIDATION = "contract_validation"
    PLATFORM_COMPATIBILITY = "platform_compatibility"
    FOCUSED_AUTOMATED_TESTS = "focused_automated_tests"
    DOMAIN_QUALIFICATION = "domain_qualification"
    LIVE_QUALIFICATION = "live_qualification"


@dataclass(frozen=True, slots=True)
class PluginQualificationSubject:
    """Identity of the qualification subject at package, capability, or domain level."""

    level: PluginQualificationLevel
    delivery_source: PluginDeliverySource
    trust_model: PlatformPluginTrustModel = PlatformPluginTrustModel.TRUSTED_IN_PROCESS
    trust_origin: PluginTrustOrigin | None = None
    package_name: str | None = None
    package_version: str | None = None
    domain: str | None = None
    capability_id: str | None = None
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    host_registration_path: str | None = None


@dataclass(frozen=True, slots=True)
class PluginQualificationResult:
    """Immutable qualification record suitable for audit and observability."""

    subject: PluginQualificationSubject
    status: QualificationStatus
    evidence: tuple[QualificationEvidence[PluginQualificationEvidenceKind], ...]
    reason: str
    domain_qualification_label: str | None = None

    @property
    def production_allowed(self) -> bool:
        return self.status is QualificationStatus.PRODUCTION_QUALIFIED


@dataclass(frozen=True, slots=True)
class PackageProductionAdmission:
    """Deterministic package-level production admission decision."""

    admitted: bool
    result: PluginQualificationResult
    compatibility: PlatformCompatibilityResult | None
    reason: str


def compatibility_evidence(
    compatibility: PlatformCompatibilityResult,
) -> QualificationEvidence[PluginQualificationEvidenceKind]:
    """Map a PLUGIN-6 compatibility result to qualification evidence metadata."""
    return QualificationEvidence(
        kind=PluginQualificationEvidenceKind.PLATFORM_COMPATIBILITY,
        code=compatibility.reason.value,
        ref=(
            f"declared={compatibility.declared_specifier};"
            f"tested={compatibility.tested_platform_version}"
        ),
    )


def build_external_package_subject(
    *,
    level: PluginQualificationLevel,
    package_name: str,
    package_version: str,
    domain: str | None = None,
    capability_id: str | None = None,
    entry_point_group: str | None = None,
    entry_point_name: str | None = None,
) -> PluginQualificationSubject:
    """Subject for wheel/distribution-delivered extensions."""
    return PluginQualificationSubject(
        level=level,
        delivery_source=PluginDeliverySource.EXTERNAL_PACKAGE,
        trust_model=PlatformPluginTrustModel.TRUSTED_IN_PROCESS,
        trust_origin=PluginTrustOrigin.INSTALLED_THIRD_PARTY_PACKAGE,
        package_name=package_name,
        package_version=package_version,
        domain=domain,
        capability_id=capability_id,
        entry_point_group=entry_point_group,
        entry_point_name=entry_point_name,
    )


def build_host_embedded_capability_subject(
    *,
    domain: str,
    capability_id: str,
    host_registration_path: str,
    level: PluginQualificationLevel = PluginQualificationLevel.CAPABILITY,
) -> PluginQualificationSubject:
    """Subject for explicitly host-registered local extensions (no wheel/EP required)."""
    return PluginQualificationSubject(
        level=level,
        delivery_source=PluginDeliverySource.HOST_EMBEDDED_EXTENSION,
        trust_model=PlatformPluginTrustModel.TRUSTED_IN_PROCESS,
        trust_origin=PluginTrustOrigin.HOST_LOCAL_CODE,
        domain=domain,
        capability_id=capability_id,
        host_registration_path=host_registration_path,
    )


def build_qualification_result(
    *,
    subject: PluginQualificationSubject,
    status: QualificationStatus,
    evidence: tuple[QualificationEvidence[PluginQualificationEvidenceKind], ...],
    reason: str,
    domain_qualification_label: str | None = None,
) -> PluginQualificationResult:
    """Construct an immutable qualification result."""
    return PluginQualificationResult(
        subject=subject,
        status=status,
        evidence=evidence,
        reason=reason,
        domain_qualification_label=domain_qualification_label,
    )


def is_production_qualified(result: PluginQualificationResult) -> bool:
    """Return whether ``result`` carries production-qualified status."""
    return result.production_allowed


def require_production_qualification(
    result: PluginQualificationResult,
) -> PluginQualificationResult:
    """Raise when ``result`` is not production-qualified (pure, side-effect free)."""
    if result.production_allowed:
        return result
    raise ProductionQualificationRequiredError(
        (
            "production qualification required; "
            f"subject level={result.subject.level.value} "
            f"status={result.status.value}: {result.reason}"
        ),
        result=result,
    )


def evaluate_package_production_admission(
    result: PluginQualificationResult,
    *,
    compatibility: PlatformCompatibilityResult | None = None,
) -> PackageProductionAdmission:
    """Evaluate production admission for a package-level qualification result.

    For external packages, incompatible platform metadata blocks admission even when
    other lifecycle states appear healthy. Host-embedded extensions do not require
    package compatibility metadata.
    """
    if result.subject.level is not PluginQualificationLevel.PACKAGE:
        return PackageProductionAdmission(
            admitted=False,
            result=result,
            compatibility=compatibility,
            reason="production admission helper requires package-level subject",
        )

    if result.subject.delivery_source is PluginDeliverySource.EXTERNAL_PACKAGE:
        if compatibility is None:
            return PackageProductionAdmission(
                admitted=False,
                result=result,
                compatibility=None,
                reason=(
                    "external package production admission requires "
                    "platform compatibility evidence"
                ),
            )
        if not compatibility.compatible:
            return PackageProductionAdmission(
                admitted=False,
                result=result,
                compatibility=compatibility,
                reason=(
                    "external package platform compatibility failed; "
                    f"{compatibility.reason.value}"
                ),
            )

    if not result.production_allowed:
        return PackageProductionAdmission(
            admitted=False,
            result=result,
            compatibility=compatibility,
            reason=result.reason,
        )

    return PackageProductionAdmission(
        admitted=True,
        result=result,
        compatibility=compatibility,
        reason="production-qualified evidence present",
    )


def resolve_host_platform_version() -> str:
    """Return the installed Intergrax platform version for compatibility checks."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("intergrax")
    except PackageNotFoundError:
        return version("Intergrax-ai")


def resolve_entry_point_distribution_identity(
    spec: EntryPointSpec,
) -> DistributionPackageIdentity | None:
    """Resolve canonical package identity for an entry point's distribution."""
    if spec.distribution is None:
        return None
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        installed = distribution(spec.distribution)
    except PackageNotFoundError:
        return None
    try:
        return DistributionPackageIdentity(name=spec.distribution, version=installed.version)
    except ValueError:
        return None


def qualification_matches_distribution_identity(
    qualification: PluginQualificationResult,
    identity: DistributionPackageIdentity,
) -> bool:
    """Return whether qualification evidence applies to ``identity``."""
    if qualification.subject.package_name is None or qualification.subject.package_version is None:
        return False
    try:
        qualified = DistributionPackageIdentity(
            name=qualification.subject.package_name,
            version=qualification.subject.package_version,
        )
    except ValueError:
        return False
    return qualified.name == identity.name and qualified.version == identity.version


def _try_parse_platform_plugin_manifest_from_distribution(
    dist: Distribution,
) -> PlatformPluginManifest | None:
    from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
    from intergrax.core.plugins.manifest_io import parse_platform_plugin_pyproject_toml

    if dist.files is None:
        return None
    for file in dist.files:
        if file.name != "pyproject.toml":
            continue
        try:
            source = dist.read_text(file)
        except OSError:
            continue
        if source is None:
            continue
        try:
            return parse_platform_plugin_pyproject_toml(source)
        except PlatformPluginManifestValidationError:
            return None
    return None


def resolve_installed_distribution_platform_compatibility(
    distribution_name: str,
    platform_version: str,
) -> PlatformCompatibilityResult | None:
    """Resolve platform compatibility from an installed distribution manifest."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        installed = distribution(distribution_name)
    except PackageNotFoundError:
        return None
    manifest = _try_parse_platform_plugin_manifest_from_distribution(installed)
    if manifest is None:
        return None
    return check_platform_compatibility(
        manifest.platform_compatibility,
        platform_version,
    )


def evaluate_external_package_entry_point_production_admission(
    spec: EntryPointSpec,
    qualification: PluginQualificationResult | None,
    *,
    platform_version: str,
) -> PackageProductionAdmission:
    """Evaluate production admission for an external-package entry point."""
    identity = resolve_entry_point_distribution_identity(spec)
    if identity is None:
        placeholder = build_qualification_result(
            subject=build_external_package_subject(
                level=PluginQualificationLevel.PACKAGE,
                package_name=spec.distribution or "unknown",
                package_version="0",
            ),
            status=QualificationStatus.NOT_QUALIFIED,
            evidence=(),
            reason="external package identity could not be resolved from entry point distribution",
        )
        return PackageProductionAdmission(
            admitted=False,
            result=placeholder,
            compatibility=None,
            reason=placeholder.reason,
        )

    compatibility = resolve_installed_distribution_platform_compatibility(
        identity.name,
        platform_version,
    )

    if qualification is None:
        missing = build_qualification_result(
            subject=build_external_package_subject(
                level=PluginQualificationLevel.PACKAGE,
                package_name=identity.name,
                package_version=identity.version,
                entry_point_group=spec.group,
                entry_point_name=spec.name,
            ),
            status=QualificationStatus.NOT_QUALIFIED,
            evidence=(),
            reason="production qualification evidence missing for external policy plugin package",
        )
        return evaluate_package_production_admission(missing, compatibility=compatibility)

    if not qualification_matches_distribution_identity(qualification, identity):
        mismatched = build_qualification_result(
            subject=qualification.subject,
            status=QualificationStatus.NOT_QUALIFIED,
            evidence=qualification.evidence,
            reason=(
                "production qualification package identity does not match entry point distribution "
                f"({identity.name}@{identity.version})"
            ),
            domain_qualification_label=qualification.domain_qualification_label,
        )
        return evaluate_package_production_admission(mismatched, compatibility=compatibility)

    return evaluate_package_production_admission(qualification, compatibility=compatibility)
