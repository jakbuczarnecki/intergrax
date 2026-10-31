# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R5-P0 deprecation/removal policy gates and adversarial probes."""

from __future__ import annotations

import pytest
from tests.qualification.compat_x._compat_x_deprecation_r5_inventory import (
    _SYNTHETIC_BENIGN_DELEGATING_MODULE,
    _SYNTHETIC_COMPAT_UNKNOWN_MODULE,
    _SYNTHETIC_EXECUTION_SHORTCUT_MODULE,
    _SYNTHETIC_FAKE_PROVIDER_MODULE,
    _SYNTHETIC_GENERIC_UNKNOWN_MODULE,
    _SYNTHETIC_GOVERNANCE_SHORTCUT_MODULE,
    _SYNTHETIC_INDEPENDENT_COMPAT_MODULE,
    _SYNTHETIC_LEGACY_TRANSFORM_MODULE,
    _SYNTHETIC_PROVIDER_COMPATIBILITY_MODULE,
    _SYNTHETIC_TRANSLATION_ONLY_MODULE,
    _UNKNOWN_DEPRECATED_PROBE_MODULE,
    SYNTHETIC_COMPAT_FOLDER_UNKNOWN_AUTHORITY_SOURCE,
    SYNTHETIC_DELEGATING_ALIAS_SOURCE,
    SYNTHETIC_DEPRECATED_IMPORT_STATE_SOURCE,
    SYNTHETIC_DUAL_TRUTH_FIELD_SOURCE,
    SYNTHETIC_EXECUTION_FROM_DEPRECATED_SOURCE,
    SYNTHETIC_FAKE_PROVIDER_RETURN_SOURCE,
    SYNTHETIC_GENERIC_UNKNOWN_DEPRECATED_SOURCE,
    SYNTHETIC_GOVERNANCE_FROM_DEPRECATED_SOURCE,
    SYNTHETIC_HISTORICAL_READ_ONLY_SOURCE,
    SYNTHETIC_INDEPENDENT_COMPAT_LOGIC_SOURCE,
    SYNTHETIC_LEGACY_TRANSFORM_SOURCE,
    SYNTHETIC_PARALLEL_DEPRECATED_IMPL_SOURCE,
    SYNTHETIC_PERMANENT_NO_REASON_SOURCE,
    SYNTHETIC_PERMANENT_WITH_REASON_SOURCE,
    SYNTHETIC_PROVIDER_COMPATIBILITY_NO_SEMANTICS_SOURCE,
    SYNTHETIC_PROVIDER_DELEGATE_SOURCE,
    SYNTHETIC_PROVIDER_INDEPENDENT_ADAPTER_SOURCE,
    SYNTHETIC_REMOVAL_FALLBACK_SOURCE,
    SYNTHETIC_RETIRED_WRITABLE_SOURCE,
    SYNTHETIC_SHIM_VERSION_OWNER_SOURCE,
    SYNTHETIC_TEMPORARY_NO_REMOVAL_SOURCE,
    SYNTHETIC_TEMPORARY_NO_REPLACEMENT_SOURCE,
    SYNTHETIC_TRANSLATION_ONLY_WIRE_SOURCE,
    SYNTHETIC_UNKNOWN_DEPRECATED_HELPER_SOURCE,
    DeprecationDisposition,
    classify_synthetic_module,
    classify_synthetic_unknown_deprecated_helper,
    collect_deprecation_r5_violations,
    deprecation_r5_gate_passes,
    deprecation_r5_metrics,
    disposition_counts,
    unclassified_candidates,
)
from tests.qualification.compat_x._compat_x_deprecation_r5_shim_reconciliation import (
    current_compat_shim_paths,
    load_frozen_historical_shim_baseline,
    load_replay_x_accepted_shim_delta,
    production_shim_set_delta,
    shim_reconciliation_gate_violations,
)
from tests.qualification.compat_x._compat_x_owner_discovery import COMPAT_X_OWNER_MATRIX
from tests.qualification.compat_x._compat_x_shim_authority import (
    build_shim_authority_scope_reconciliation,
)
from tests.qualification.compat_x._compat_x_types import (
    POLICY_ONLY_DOMAIN_COMPOSITION_OWNER,
    OwnerResponsibilityState,
)
from tests.qualification.compat_x._compat_x_versioning_gates import (
    inventory_shim_canonical_authority_violations,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_r5_p0_closed_world_gate_passes_on_current_head() -> None:
    passed, violations = deprecation_r5_gate_passes()
    assert passed, violations
    assert unclassified_candidates() == ()
    metrics = deprecation_r5_metrics()
    assert metrics["unclassified_count"] == 0
    assert metrics["shim_inspected"] == metrics["shim_candidates"]
    assert metrics["shim_parallel_authority_count"] == 0
    assert metrics["current_version_authority_violations"] == 0
    assert metrics["governance_authority_violations"] == 0
    assert metrics["execution_authority_violations"] == 0
    assert metrics["retired_write_violations"] == 0
    assert metrics["silent_fallback_violations"] == 0
    assert metrics["parallel_deprecated_implementation_violations"] == 0
    assert metrics["dual_truth_field_violations"] == 0
    assert metrics["placeholder_replacement_count"] == 0
    shim_growth = metrics["shim_growth_reconciliation"]
    assert isinstance(shim_growth, dict)
    baseline = load_frozen_historical_shim_baseline()
    delta = production_shim_set_delta()
    replay_delta = load_replay_x_accepted_shim_delta()
    assert shim_growth["historical_accepted_shim_count"] == baseline.count == 474
    assert shim_growth["current_shim_count"] == metrics["shim_candidates"] == len(delta.current_paths)
    assert shim_growth["shim_removed_paths"] == list(delta.removed_paths)
    assert shim_growth["shim_added_paths"] == list(delta.added_paths)
    assert delta.removed_paths == replay_delta.removed_paths
    assert delta.added_paths == replay_delta.added_paths
    assert inventory_shim_canonical_authority_violations() == ()
    shim = build_shim_authority_scope_reconciliation()
    assert shim.uninspected_compatibility_candidates == 0


def test_r5_p0_owner_matrix_deprecation_policy_confirmed() -> None:
    row = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "deprecation_removal")
    assert row.responsibility_state == OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER
    assert "COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md" in row.semantic_owner_path
    assert row.composition_owner_path == POLICY_ONLY_DOMAIN_COMPOSITION_OWNER


def test_r5_p0_metrics_reportable() -> None:
    metrics = deprecation_r5_metrics()
    prod = metrics["production_files_scanned"]
    total = metrics["total_deprecation_candidates"]
    assert isinstance(prod, int) and prod > 0
    assert isinstance(total, int) and total > 0
    assert disposition_counts()


def test_r5_probe_a_unknown_deprecated_helper() -> None:
    assert classify_synthetic_unknown_deprecated_helper() == DeprecationDisposition.UNCLASSIFIED
    violations = collect_deprecation_r5_violations(
        extra_sources=((_UNKNOWN_DEPRECATED_PROBE_MODULE, SYNTHETIC_UNKNOWN_DEPRECATED_HELPER_SOURCE),)
    )
    assert any("UNCLASSIFIED" in v for v in violations)


def test_r5_probe_b_parallel_deprecated_implementation() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/synthetic/parallel_deprecated_impl.py", SYNTHETIC_PARALLEL_DEPRECATED_IMPL_SOURCE),)
    )
    assert any("PARALLEL_AUTHORITY" in v for v in violations)


def test_r5_probe_c_delegating_alias_passes() -> None:
    disposition = classify_synthetic_module(
        "intergrax/synthetic/delegating_alias.py",
        SYNTHETIC_DELEGATING_ALIAS_SOURCE,
    )
    assert disposition != DeprecationDisposition.UNCLASSIFIED
    assert disposition != DeprecationDisposition.PARALLEL_AUTHORITY
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/synthetic/delegating_alias.py", SYNTHETIC_DELEGATING_ALIAS_SOURCE),)
    )
    assert not any("PARALLEL_AUTHORITY" in v for v in violations)


def test_r5_probe_d_deprecated_import_path_owns_state() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/synthetic/deprecated_import_state.py",
                SYNTHETIC_DEPRECATED_IMPORT_STATE_SOURCE.replace("_STATE", "_STATE  # deprecated"),
            ),
        )
    )
    assert any("owns state" in v for v in violations)


def test_r5_probe_e_dual_truth_deprecated_field() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/synthetic/dual_truth_field.py", SYNTHETIC_DUAL_TRUTH_FIELD_SOURCE),)
    )
    assert any("dual truth" in v for v in violations)


def test_r5_probe_f_temporary_missing_replacement() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/synthetic/temporary_no_replacement.py",
                SYNTHETIC_TEMPORARY_NO_REPLACEMENT_SOURCE,
            ),
        )
    )
    assert any("missing replacement" in v for v in violations)


def test_r5_probe_g_temporary_missing_removal_condition() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/synthetic/temporary_no_removal.py",
                SYNTHETIC_TEMPORARY_NO_REMOVAL_SOURCE,
            ),
        )
    )
    assert any("missing removal condition" in v for v in violations)


def test_r5_probe_h_permanent_boundary_with_reason_passes() -> None:
    disposition = classify_synthetic_module(
        "intergrax/synthetic/permanent_with_reason.py",
        SYNTHETIC_PERMANENT_WITH_REASON_SOURCE,
    )
    assert disposition in {
        DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY,
        DeprecationDisposition.NON_DEPRECATION_USE,
    }


def test_r5_probe_i_permanent_boundary_without_reason() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/synthetic/permanent_no_reason.py",
                SYNTHETIC_PERMANENT_NO_REASON_SOURCE,
            ),
        )
    )
    assert any("permanent boundary without reason" in v for v in violations)


def test_r5_probe_j_legacy_provider_delegate_passes() -> None:
    disposition = classify_synthetic_module(
        "intergrax/integrations/providers/synthetic/register.py",
        SYNTHETIC_PROVIDER_DELEGATE_SOURCE,
    )
    assert disposition == DeprecationDisposition.LEGACY_PROVIDER_DELEGATE


def test_r5_probe_k_legacy_provider_independent_adapter() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/integrations/providers/synthetic/standalone_adapter.py",
                SYNTHETIC_PROVIDER_INDEPENDENT_ADAPTER_SOURCE,
            ),
        )
    )
    assert any("independent adapter" in v for v in violations)


def test_r5_probe_l_deprecated_governance_authority() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/synthetic/governance_from_deprecated.py",
                SYNTHETIC_GOVERNANCE_FROM_DEPRECATED_SOURCE,
            ),
        )
    )
    assert any("governance" in v for v in violations)


def test_r5_probe_m_deprecated_execution_authority() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                "intergrax/synthetic/execution_from_deprecated.py",
                SYNTHETIC_EXECUTION_FROM_DEPRECATED_SOURCE,
            ),
        )
    )
    assert any("execution" in v for v in violations)


def test_r5_probe_n_retired_write_path() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/synthetic/retired_writable.py", SYNTHETIC_RETIRED_WRITABLE_SOURCE),)
    )
    assert any("retired write" in v for v in violations)


def test_r5_probe_o_historical_read_only_passes() -> None:
    disposition = classify_synthetic_module(
        "intergrax/synthetic/historical_read_only.py",
        SYNTHETIC_HISTORICAL_READ_ONLY_SOURCE,
    )
    assert disposition != DeprecationDisposition.UNCLASSIFIED


def test_r5_probe_p_removal_fallback() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/synthetic/removal_fallback.py", SYNTHETIC_REMOVAL_FALLBACK_SOURCE),)
    )
    assert any("fallback" in v for v in violations)


def test_r5_probe_q_shim_version_owner() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/synthetic/shim_version_owner.py", SYNTHETIC_SHIM_VERSION_OWNER_SOURCE),)
    )
    assert any("current version owner" in v for v in violations)


def test_r5_probe_r_omitted_candidate_fails_gate() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=((_UNKNOWN_DEPRECATED_PROBE_MODULE, SYNTHETIC_UNKNOWN_DEPRECATED_HELPER_SOURCE),)
    )
    assert violations


def test_r5_probe_a2_legacy_transform_unclassified() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_LEGACY_TRANSFORM_MODULE,
        SYNTHETIC_LEGACY_TRANSFORM_SOURCE,
    )
    assert disposition == DeprecationDisposition.UNCLASSIFIED
    violations = collect_deprecation_r5_violations(
        extra_sources=((_SYNTHETIC_LEGACY_TRANSFORM_MODULE, SYNTHETIC_LEGACY_TRANSFORM_SOURCE),)
    )
    assert violations


def test_r5_probe_s_generic_unknown_production_candidate() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_GENERIC_UNKNOWN_MODULE,
        SYNTHETIC_GENERIC_UNKNOWN_DEPRECATED_SOURCE,
    )
    assert disposition == DeprecationDisposition.UNCLASSIFIED
    violations = collect_deprecation_r5_violations(
        extra_sources=((_SYNTHETIC_GENERIC_UNKNOWN_MODULE, SYNTHETIC_GENERIC_UNKNOWN_DEPRECATED_SOURCE),)
    )
    assert any("UNCLASSIFIED" in v for v in violations)


def test_r5_probe_t_compat_folder_unknown_not_permanent() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_COMPAT_UNKNOWN_MODULE,
        SYNTHETIC_COMPAT_FOLDER_UNKNOWN_AUTHORITY_SOURCE,
    )
    assert disposition == DeprecationDisposition.UNCLASSIFIED


def test_r5_probe_u_fake_provider_return_not_delegate() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_FAKE_PROVIDER_MODULE,
        SYNTHETIC_FAKE_PROVIDER_RETURN_SOURCE,
    )
    assert disposition == DeprecationDisposition.UNCLASSIFIED


def test_r5_probe_v_production_governance_shortcut() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                _SYNTHETIC_GOVERNANCE_SHORTCUT_MODULE,
                SYNTHETIC_GOVERNANCE_FROM_DEPRECATED_SOURCE,
            ),
        )
    )
    assert any("governance" in v for v in violations)


def test_r5_probe_w_production_execution_shortcut() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                _SYNTHETIC_EXECUTION_SHORTCUT_MODULE,
                SYNTHETIC_EXECUTION_FROM_DEPRECATED_SOURCE,
            ),
        )
    )
    assert any("execution" in v for v in violations)


def test_r5_probe_x_retired_writer() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/applications/host/retired_writer.py", SYNTHETIC_RETIRED_WRITABLE_SOURCE),)
    )
    assert any("retired write" in v for v in violations)


def test_r5_probe_y_silent_legacy_fallback() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(("intergrax/applications/host/removal_fallback.py", SYNTHETIC_REMOVAL_FALLBACK_SOURCE),)
    )
    assert any("fallback" in v for v in violations)


def test_r5_probe_aa_shim_reconciliation_hidden_removal_fails() -> None:
    baseline = load_frozen_historical_shim_baseline()
    current = current_compat_shim_paths()
    removed_path = next(iter(sorted(baseline.paths)))
    mutated = frozenset(p for p in current if p != removed_path)
    violations = shim_reconciliation_gate_violations(current_paths=mutated)
    assert violations


def test_r5_probe_ab_shim_reconciliation_balanced_substitution_fails() -> None:
    baseline = load_frozen_historical_shim_baseline()
    current = current_compat_shim_paths()
    removed_path = next(iter(sorted(baseline.paths)))
    mutated = frozenset(p for p in current if p != removed_path) | {
        "intergrax/synthetic/unexpected_substitution_shim.py"
    }
    violations = shim_reconciliation_gate_violations(current_paths=mutated)
    assert violations


def test_r5_probe_ac_shim_reconciliation_extra_addition_fails() -> None:
    current = current_compat_shim_paths() | {"intergrax/synthetic/third_unexpected_shim.py"}
    violations = shim_reconciliation_gate_violations(current_paths=current)
    assert violations


def test_r5_probe_ad_provider_compatibility_without_semantics_unclassified() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_PROVIDER_COMPATIBILITY_MODULE,
        SYNTHETIC_PROVIDER_COMPATIBILITY_NO_SEMANTICS_SOURCE,
    )
    assert disposition == DeprecationDisposition.UNCLASSIFIED
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                _SYNTHETIC_PROVIDER_COMPATIBILITY_MODULE,
                SYNTHETIC_PROVIDER_COMPATIBILITY_NO_SEMANTICS_SOURCE,
            ),
        )
    )
    assert any("UNCLASSIFIED" in v for v in violations)


def test_r5_probe_ae_independent_compat_not_permanent() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_INDEPENDENT_COMPAT_MODULE,
        SYNTHETIC_INDEPENDENT_COMPAT_LOGIC_SOURCE,
    )
    assert disposition != DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY


def test_r5_probe_af_translation_only_wire_permanent() -> None:
    disposition = classify_synthetic_module(
        _SYNTHETIC_TRANSLATION_ONLY_MODULE,
        SYNTHETIC_TRANSLATION_ONLY_WIRE_SOURCE,
    )
    assert disposition == DeprecationDisposition.PERMANENT_TRANSLATION_BOUNDARY


def test_r5_probe_ag_historical_read_disposition() -> None:
    disposition = classify_synthetic_module(
        "intergrax/synthetic/historical_read_only.py",
        SYNTHETIC_HISTORICAL_READ_ONLY_SOURCE,
    )
    assert disposition == DeprecationDisposition.HISTORICAL_READ_COMPATIBILITY


def test_r5_probe_z_benign_delegating_wrapper_passes() -> None:
    violations = collect_deprecation_r5_violations(
        extra_sources=(
            (
                _SYNTHETIC_BENIGN_DELEGATING_MODULE,
                SYNTHETIC_DELEGATING_ALIAS_SOURCE,
            ),
        )
    )
    assert not any("PARALLEL_AUTHORITY" in v for v in violations)
    assert not any("UNCLASSIFIED" in v for v in violations)
