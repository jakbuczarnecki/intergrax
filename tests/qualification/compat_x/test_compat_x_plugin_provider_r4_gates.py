# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R4 plugin/provider compatibility policy gates and adversarial probes."""

from __future__ import annotations

import pytest

from intergrax.core.distribution import PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
    evaluate_package_production_admission,
)
from intergrax.core.qualification import QualificationStatus
from intergrax.integrations.contracts.external_contract_compatibility import (
    DimensionCompatibilityStatus,
    ExternalContractCompatibilityDimension,
    ExternalContractCompatibilityOutcome,
    ExternalContractCompatibilityReasonCode,
)
from intergrax.integrations.examples.custom_memory_kv.compatibility import (
    CustomMemoryKvSchemaCompatibilityEvaluator,
)
from intergrax.integrations.external_contract_compatibility_extensions import (
    ExternalContractCompatibilityExtensionsError,
    external_contract_compatibility_extensions,
)
from tests.qualification.external_contract_compatibility.test_external_contract_compatibility_certification import (
    _AcceptAllPolicy,
    _expectation,
    _schema_evidence,
    _service,
    _request,
    _status_evaluator,
    _subject,
)
from tests.qualification.compat_x._compat_x_owner_discovery import COMPAT_X_OWNER_MATRIX
from tests.qualification.compat_x._compat_x_plugin_provider_r4_inventory import (
    PluginProviderCompatibilityDisposition,
    SYNTHETIC_COMPATIBILITY_ACTIVATION_SHORTCUT_SOURCE,
    SYNTHETIC_COMPATIBILITY_GOVERNANCE_SHORTCUT_SOURCE,
    SYNTHETIC_IMPLEMENTATION_COMPATIBILITY_SOURCE,
    SYNTHETIC_REGISTRY_V2_SHADOW_REGISTER_SOURCE,
    SYNTHETIC_ROGUE_LOCAL_PROVIDER_REGISTRY_SOURCE,
    SYNTHETIC_SANCTIONED_CATALOG_DELEGATE_SOURCE,
    SYNTHETIC_SHADOW_EXTERNAL_ASSESSMENT_SOURCE,
    SYNTHETIC_SHADOW_PACKAGE_EVALUATOR_SOURCE,
    SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE,
    _UNKNOWN_COMPAT_PROBE_MODULE,
    classify_synthetic_implementation_compatibility,
    classify_synthetic_registry_shadow_register,
    classify_synthetic_sanctioned_catalog_delegate,
    classify_synthetic_shadow_package_evaluator,
    classify_synthetic_unknown_helper,
    collect_plugin_provider_compatibility_r4_violations,
    count_provider_catalog_authorities,
    disposition_counts,
    plugin_provider_compatibility_r4_gate_passes,
    provider_r4_authority_report,
    qualification_composition_authority_violations,
    unclassified_records,
)
from tests.qualification.compat_x._compat_x_types import (
    POLICY_ONLY_DOMAIN_COMPOSITION_OWNER,
    CompatOwnerMatrixRow,
    OwnerResponsibilityState,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_r4_closed_world_gate_passes_on_current_head() -> None:
    passed, violations = plugin_provider_compatibility_r4_gate_passes()
    assert passed, violations
    assert unclassified_records() == ()
    assert disposition_counts().get(PluginProviderCompatibilityDisposition.UNCLASSIFIED.value, 0) == 0


def test_r4_owner_matrix_plugin_provider_policy_confirmed() -> None:
    policy = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "plugin_provider_compatibility_policy")
    assert policy.responsibility_state == OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER
    assert "COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md" in policy.semantic_owner_path
    assert policy.composition_owner_path == POLICY_ONLY_DOMAIN_COMPOSITION_OWNER
    manifest = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "platform_plugin_manifest_compatibility")
    assert manifest.responsibility_state == OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER
    provider = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "provider_contract_compatibility")
    assert provider.responsibility_state == OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER


def test_r4_probe_a_shadow_package_evaluator() -> None:
    assert (
        classify_synthetic_shadow_package_evaluator()
        == PluginProviderCompatibilityDisposition.PACKAGE_COMPATIBILITY_SHADOW_AUTHORITY
    )
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            ("intergrax/synthetic/shadow_package_eval.py", SYNTHETIC_SHADOW_PACKAGE_EVALUATOR_SOURCE),
        )
    )
    assert any("shadow evaluator" in v for v in violations)


def test_r4_probe_b_unknown_compatibility_helper() -> None:
    assert classify_synthetic_unknown_helper() == PluginProviderCompatibilityDisposition.UNCLASSIFIED
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            (_UNKNOWN_COMPAT_PROBE_MODULE, SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE),
        )
    )
    assert any("UNCLASSIFIED" in v for v in violations)


def test_r4_probe_c_registry_shadow_register() -> None:
    assert (
        classify_synthetic_registry_shadow_register()
        == PluginProviderCompatibilityDisposition.PROVIDER_REGISTRATION_SHADOW_AUTHORITY
    )
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            ("intergrax/synthetic/rogue_catalog_register.py", SYNTHETIC_REGISTRY_V2_SHADOW_REGISTER_SOURCE),
        )
    )
    assert any("shadow" in v.lower() for v in violations)


def test_r4_probe_d_implementation_class_compatibility() -> None:
    assert (
        classify_synthetic_implementation_compatibility()
        == PluginProviderCompatibilityDisposition.IMPLEMENTATION_BASED_COMPATIBILITY_AUTHORITY
    )
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            ("intergrax/synthetic/impl_class_compat.py", SYNTHETIC_IMPLEMENTATION_COMPATIBILITY_SOURCE),
        )
    )
    assert any("implementation-based" in v for v in violations)


def test_r4_probe_e_external_evaluator_ambiguity() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    e1 = _status_evaluator(
        "schema.a",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    e2 = _status_evaluator(
        "schema.b",
        ExternalContractCompatibilityDimension.SCHEMA,
        DimensionCompatibilityStatus.COMPATIBLE,
    )
    result = _service(e1, e2, evidence_policy=_AcceptAllPolicy()).assess(
        _request(exp, (_schema_evidence(exp.subject),))
    )
    assert result.reason_code is ExternalContractCompatibilityReasonCode.EVALUATOR_AMBIGUITY


def test_r4_probe_f_missing_external_evidence_fail_closed() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    result = _service(evidence_policy=_AcceptAllPolicy()).assess(_request(exp, ()))
    assert result.outcome is not ExternalContractCompatibilityOutcome.COMPATIBLE


def test_r4_probe_g_cross_tenant_evidence() -> None:
    exp = _expectation(
        subject=_subject(tenant_id="tenant-a"),
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA}),
    )
    wrong = _schema_evidence(_subject(tenant_id="tenant-b"))
    result = _service(evidence_policy=_AcceptAllPolicy()).assess(_request(exp, (wrong,)))
    assert result.outcome is not ExternalContractCompatibilityOutcome.COMPATIBLE


def test_r4_probe_h_provider_mismatch() -> None:
    exp = _expectation(
        required=frozenset({ExternalContractCompatibilityDimension.SCHEMA})
    )
    wrong = _subject(provider_id="other", integration_id="other:api")
    result = _service(evidence_policy=_AcceptAllPolicy()).assess(
        _request(exp, (_schema_evidence(wrong),))
    )
    assert result.reason_code is ExternalContractCompatibilityReasonCode.IDENTITY_MISMATCH


def test_r4_probe_i_compatibility_does_not_activate() -> None:
    compat = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=0.0.0"),
        "1.0.0",
    )
    assert compat.compatible
    subject = build_external_package_subject(
        level=PluginQualificationLevel.PACKAGE,
        package_name="demo.pkg",
        package_version="1.0.0",
    )
    result = build_qualification_result(
        subject=subject,
        status=QualificationStatus.NOT_QUALIFIED,
        evidence=(),
        reason="missing qualification evidence",
    )
    admission = evaluate_package_production_admission(result, compatibility=compat)
    assert admission.admitted is False


def test_r4_probe_j_external_extension_replaceability() -> None:
    replacement = CustomMemoryKvSchemaCompatibilityEvaluator()
    wired = external_contract_compatibility_extensions(evaluators=(replacement,))
    assert replacement in wired.evaluators


def test_r4_probe_k_duplicate_extension_ids() -> None:
    ev = CustomMemoryKvSchemaCompatibilityEvaluator()
    with pytest.raises(ExternalContractCompatibilityExtensionsError):
        external_contract_compatibility_extensions(evaluators=(ev, ev))


def test_r4_probe_l_unknown_production_candidate_fails_gate() -> None:
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            (_UNKNOWN_COMPAT_PROBE_MODULE, SYNTHETIC_UNKNOWN_PLUGIN_COMPATIBILITY_SOURCE),
        )
    )
    assert violations


def test_r4_probe_m_generic_unknown_compat_helper_production_path() -> None:
    assert classify_synthetic_unknown_helper() == PluginProviderCompatibilityDisposition.UNCLASSIFIED


def test_r4_probe_n_sanctioned_catalog_delegate() -> None:
    assert (
        classify_synthetic_sanctioned_catalog_delegate()
        == PluginProviderCompatibilityDisposition.PROVIDER_CATALOG_DELEGATE
    )
    report = provider_r4_authority_report()
    assert report["authority_count"] == 1


def test_r4_probe_o_rogue_local_provider_registry() -> None:
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            (
                "intergrax/applications/provider_bridge/rogue_local_registry.py",
                SYNTHETIC_ROGUE_LOCAL_PROVIDER_REGISTRY_SOURCE,
            ),
        )
    )
    assert any("shadow" in v.lower() for v in violations)


def test_r4_probe_p_qualification_as_composition_owner_fails_gate() -> None:
    policy = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "plugin_provider_compatibility_policy")
    mutated = CompatOwnerMatrixRow(
        concern=policy.concern,
        responsibility_state=policy.responsibility_state,
        semantic_owner_path=policy.semantic_owner_path,
        composition_owner_path="tests/qualification/compat_x/evil_composition_owner.py",
        evidence_paths=policy.evidence_paths,
    )
    matrix = tuple(
        mutated if row.concern == "plugin_provider_compatibility_policy" else row
        for row in COMPAT_X_OWNER_MATRIX
    )
    assert qualification_composition_authority_violations(matrix)


def test_r4_probe_q_shadow_external_assessment_authority() -> None:
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            (
                "intergrax/applications/provider_bridge/rogue_external_assessment.py",
                SYNTHETIC_SHADOW_EXTERNAL_ASSESSMENT_SOURCE,
            ),
        )
    )
    assert any("external" in v.lower() for v in violations)


def test_r4_probe_r_compatibility_activation_shortcut() -> None:
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            (
                "intergrax/applications/provider_bridge/compat_activation_shortcut.py",
                SYNTHETIC_COMPATIBILITY_ACTIVATION_SHORTCUT_SOURCE,
            ),
        )
    )
    assert any("activation" in v.lower() for v in violations)


def test_r4_probe_s_compatibility_governance_shortcut() -> None:
    violations = collect_plugin_provider_compatibility_r4_violations(
        extra_sources=(
            (
                "intergrax/applications/provider_bridge/compat_governance_shortcut.py",
                SYNTHETIC_COMPATIBILITY_GOVERNANCE_SHORTCUT_SOURCE,
            ),
        )
    )
    assert any("governance" in v.lower() or "execution" in v.lower() for v in violations)


def test_r4_provider_catalog_authority_exactly_one() -> None:
    assert count_provider_catalog_authorities() == 1
