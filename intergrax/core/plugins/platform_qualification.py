# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform Plugin trust, qualification, and production gate contracts (PLATFORM-PLUGIN-7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.core.plugins.errors import ProductionQualificationRequiredError
from intergrax.core.plugins.platform_semantics import PlatformCompatibilityResult


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


class PluginQualificationStatus(StrEnum):
    """Qualification outcome — distinct from lifecycle/discovery states."""

    NOT_QUALIFIED = "not_qualified"
    QUALIFIED = "qualified"
    PRODUCTION_QUALIFIED = "production_qualified"
    REJECTED = "rejected"


class PluginQualificationEvidenceKind(StrEnum):
    """Auditable evidence categories; domains own required thresholds."""

    CONTRACT_VALIDATION = "contract_validation"
    PLATFORM_COMPATIBILITY = "platform_compatibility"
    FOCUSED_AUTOMATED_TESTS = "focused_automated_tests"
    DOMAIN_QUALIFICATION = "domain_qualification"
    LIVE_QUALIFICATION = "live_qualification"


@dataclass(frozen=True, slots=True)
class PluginQualificationEvidence:
    """Safe, immutable evidence metadata (no secrets or raw test logs)."""

    kind: PluginQualificationEvidenceKind
    code: str
    ref: str | None = None
    label: str | None = None


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
    status: PluginQualificationStatus
    evidence: tuple[PluginQualificationEvidence, ...]
    reason: str
    domain_qualification_label: str | None = None

    @property
    def production_allowed(self) -> bool:
        return self.status is PluginQualificationStatus.PRODUCTION_QUALIFIED


@dataclass(frozen=True, slots=True)
class PackageProductionAdmission:
    """Deterministic package-level production admission decision."""

    admitted: bool
    result: PluginQualificationResult
    compatibility: PlatformCompatibilityResult | None
    reason: str


def compatibility_evidence(
    compatibility: PlatformCompatibilityResult,
) -> PluginQualificationEvidence:
    """Map a PLUGIN-6 compatibility result to qualification evidence metadata."""
    return PluginQualificationEvidence(
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
    status: PluginQualificationStatus,
    evidence: tuple[PluginQualificationEvidence, ...],
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

    if (
        result.subject.delivery_source is PluginDeliverySource.EXTERNAL_PACKAGE
        and compatibility is not None
        and not compatibility.compatible
    ):
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
