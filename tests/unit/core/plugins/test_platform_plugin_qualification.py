# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.plugins.errors import ProductionQualificationRequiredError
from intergrax.core.plugins.package_contract import PlatformCompatibility
from intergrax.core.plugins.platform_qualification import (
    PlatformPluginTrustModel,
    PluginDeliverySource,
    PluginQualificationEvidence,
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    PluginQualificationResult,
    PluginQualificationStatus,
    PluginQualificationSubject,
    build_external_package_subject,
    build_host_embedded_capability_subject,
    build_qualification_result,
    compatibility_evidence,
    evaluate_package_production_admission,
    is_production_qualified,
    require_production_qualification,
)
from intergrax.core.plugins.platform_semantics import (
    PlatformPluginLifecycleState,
    check_platform_compatibility,
)

pytestmark = pytest.mark.unit


def _package_subject() -> PluginQualificationSubject:
    return build_external_package_subject(
        level=PluginQualificationLevel.PACKAGE,
        package_name="acme-intergrax",
        package_version="1.0.0",
    )


def _capability_subject(
    *,
    capability_id: str,
    status: PluginQualificationStatus,
) -> PluginQualificationResult:
    subject = build_external_package_subject(
        level=PluginQualificationLevel.CAPABILITY,
        package_name="acme-intergrax",
        package_version="1.0.0",
        domain="tools",
        capability_id=capability_id,
        entry_point_group="intergrax.tools",
        entry_point_name=capability_id,
    )
    return build_qualification_result(
        subject=subject,
        status=status,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="domain.tests.passed",
                ref="tests/unit/acme/test_tool.py",
            ),
        ),
        reason=f"{capability_id} domain qualification",
    )


def test_compatible_does_not_imply_qualified() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=1,<2"),
        "1.5",
    )
    assert compatibility.compatible is True
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.NOT_QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="platform compatible; qualification evidence pending",
    )
    assert result.status is PluginQualificationStatus.NOT_QUALIFIED
    assert result.production_allowed is False


def test_enabled_lifecycle_state_is_not_qualification() -> None:
    assert PlatformPluginLifecycleState.ENABLED.value == "enabled"
    assert "qualified" not in {item.value for item in PlatformPluginLifecycleState}


def test_qualified_is_not_production_qualified() -> None:
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.FOCUSED_AUTOMATED_TESTS,
                code="ci.package.install",
            ),
        ),
        reason="package qualified for non-production use",
    )
    assert result.status is PluginQualificationStatus.QUALIFIED
    assert result.production_allowed is False
    assert is_production_qualified(result) is False


def test_production_gate_rejects_merely_qualified_result() -> None:
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(),
        reason="qualified only",
    )
    with pytest.raises(ProductionQualificationRequiredError) as exc_info:
        require_production_qualification(result)
    assert exc_info.value.result is result


def test_production_gate_accepts_production_qualified_result() -> None:
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="domain.production_gate.passed",
            ),
        ),
        reason="approved for production host profiles",
    )
    assert require_production_qualification(result) is result
    assert is_production_qualified(result) is True


def test_package_level_result() -> None:
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="package production gate",
    )
    assert result.subject.level is PluginQualificationLevel.PACKAGE


def test_capability_level_result() -> None:
    result = _capability_subject(
        capability_id="acme_tool",
        status=PluginQualificationStatus.QUALIFIED,
    )
    assert result.subject.level is PluginQualificationLevel.CAPABILITY
    assert result.subject.capability_id == "acme_tool"


def test_domain_level_result_with_label() -> None:
    subject = build_external_package_subject(
        level=PluginQualificationLevel.DOMAIN,
        package_name="acme-intergrax",
        package_version="1.0.0",
        domain="vendor_knowledge",
        capability_id="acme_provider",
    )
    result = build_qualification_result(
        subject=subject,
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.LIVE_QUALIFICATION,
                code="vk.live_rollout.passed",
                label="live-qualified",
            ),
        ),
        reason="domain live qualification recorded",
        domain_qualification_label="live-qualified",
    )
    assert result.subject.level is PluginQualificationLevel.DOMAIN
    assert result.domain_qualification_label == "live-qualified"


def test_mixed_capability_outcomes_in_same_package() -> None:
    integration = _capability_subject(
        capability_id="acme_integration",
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
    )
    tool = _capability_subject(
        capability_id="acme_tool",
        status=PluginQualificationStatus.QUALIFIED,
    )
    skill = _capability_subject(
        capability_id="acme_skill",
        status=PluginQualificationStatus.REJECTED,
    )
    assert integration.production_allowed is True
    assert tool.production_allowed is False
    assert skill.status is PluginQualificationStatus.REJECTED
    assert integration.subject.package_name == tool.subject.package_name == skill.subject.package_name


def test_host_embedded_capability_subject_without_wheel_metadata() -> None:
    subject = build_host_embedded_capability_subject(
        domain="tools",
        capability_id="my_custom_tool",
        host_registration_path="applications/my_app/extensions/my_tool.py",
    )
    assert subject.delivery_source is PluginDeliverySource.HOST_EMBEDDED_EXTENSION
    assert subject.package_name is None
    assert subject.entry_point_group is None
    assert subject.host_registration_path is not None
    assert subject.host_registration_path.endswith("my_tool.py")


def test_host_embedded_production_qualification_without_entry_point() -> None:
    subject = build_host_embedded_capability_subject(
        domain="tools",
        capability_id="my_custom_tool",
        host_registration_path="applications/my_app/extensions/my_tool.py",
    )
    result = build_qualification_result(
        subject=subject,
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.CONTRACT_VALIDATION,
                code="tool.contract.valid",
            ),
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="host.production_gate.passed",
            ),
        ),
        reason="explicit host registration with domain production evidence",
    )
    assert require_production_qualification(result) is result


def test_trust_model_is_truthful_in_process_only() -> None:
    subject = build_external_package_subject(
        level=PluginQualificationLevel.PACKAGE,
        package_name="acme-intergrax",
        package_version="1.0.0",
    )
    assert subject.trust_model is PlatformPluginTrustModel.TRUSTED_IN_PROCESS
    forbidden = {"verified", "signed", "sandboxed", "isolated"}
    assert forbidden.isdisjoint({item.value for item in PlatformPluginTrustModel})


def test_evidence_result_immutable_and_explicit() -> None:
    evidence = PluginQualificationEvidence(
        kind=PluginQualificationEvidenceKind.CONTRACT_VALIDATION,
        code="integration.contract.valid",
        ref="manifest:acme_foo",
    )
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(evidence,),
        reason="contract validation passed",
    )
    with pytest.raises(AttributeError):
        result.status = PluginQualificationStatus.PRODUCTION_QUALIFIED  # type: ignore[misc]
    assert result.evidence[0].code == "integration.contract.valid"
    assert "secret" not in result.reason.lower()


def test_external_package_missing_compatibility_blocks_production_admission() -> None:
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="production-qualified without compatibility evidence",
    )
    admission = evaluate_package_production_admission(result, compatibility=None)
    assert admission.admitted is False
    assert admission.compatibility is None
    assert "compatibility" in admission.reason.lower()


def test_compatible_external_package_admitted_when_production_qualified() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=1"),
        "1.0.0",
    )
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="production-qualified with compatible platform metadata",
    )
    admission = evaluate_package_production_admission(result, compatibility=compatibility)
    assert admission.admitted is True
    assert admission.compatibility is compatibility


def test_incompatible_external_package_blocks_package_production_admission() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=2"),
        "1.0.0",
    )
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="would be production-qualified without compatibility check",
    )
    admission = evaluate_package_production_admission(result, compatibility=compatibility)
    assert admission.admitted is False
    assert admission.compatibility is compatibility
    assert "compatibility failed" in admission.reason


def test_compatible_external_package_requires_production_status() -> None:
    compatibility = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=1"),
        "1.0.0",
    )
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="compatible but not production-qualified",
    )
    admission = evaluate_package_production_admission(result, compatibility=compatibility)
    assert admission.admitted is False


def test_host_embedded_package_admission_skips_compatibility_metadata() -> None:
    subject = build_host_embedded_capability_subject(
        domain="tools",
        capability_id="my_custom_tool",
        host_registration_path="applications/my_app/extensions/my_tool.py",
        level=PluginQualificationLevel.PACKAGE,
    )
    result = build_qualification_result(
        subject=subject,
        status=PluginQualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(),
        reason="host-local production qualification",
    )
    admission = evaluate_package_production_admission(result, compatibility=None)
    assert admission.admitted is True


def test_live_qualification_is_optional_domain_metadata() -> None:
    result = build_qualification_result(
        subject=_package_subject(),
        status=PluginQualificationStatus.QUALIFIED,
        evidence=(
            PluginQualificationEvidence(
                kind=PluginQualificationEvidenceKind.LIVE_QUALIFICATION,
                code="rag.live.backend.passed",
                label="live-qualified",
            ),
        ),
        reason="domain recorded live qualification",
        domain_qualification_label="live-qualified",
    )
    assert result.status is PluginQualificationStatus.QUALIFIED
    assert result.domain_qualification_label == "live-qualified"
