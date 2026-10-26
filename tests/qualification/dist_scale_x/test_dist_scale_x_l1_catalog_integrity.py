# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P1 L1 gates — catalog coverage and structural invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._constants import (
    DIMENSION_IDS,
    EXIT_CRITERION_IDS,
    MANDATORY_BACKEND_IDS,
    WORKLOAD_IDS,
)
from tests.qualification.dist_scale_x.authority_surface_guard import scan_dist_scale_x_authority_surface
from tests.qualification.dist_scale_x.catalog import (
    DIST_SCALE_X_BACKEND_CATALOG,
    DIST_SCALE_X_DIMENSION_CATALOG,
    DIST_SCALE_X_EXIT_CRITERION_CATALOG,
    DIST_SCALE_X_HISTORICAL_EVIDENCE_CATALOG,
    DIST_SCALE_X_THRESHOLD_LOCK_STATE,
    DIST_SCALE_X_WORKLOAD_CATALOG,
    BackendQualificationClass,
    ThresholdLockState,
    validate_dist_scale_x_catalog_mappings,
)
from tests.qualification.dist_scale_x.selection import (
    DIST_SCALE_X_P1_SELECTION_CATALOG,
    DistScaleXQualificationLayer,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_PACKAGE_DIR = Path(__file__).resolve().parent


def test_l1_dimension_catalog_covers_a_through_n() -> None:
    ids = tuple(entry.dimension_id for entry in DIST_SCALE_X_DIMENSION_CATALOG)
    assert ids == DIMENSION_IDS


def test_l1_workload_catalog_covers_wl_1_through_wl_7() -> None:
    ids = tuple(entry.workload_id for entry in DIST_SCALE_X_WORKLOAD_CATALOG)
    assert ids == WORKLOAD_IDS


def test_l1_exit_criterion_catalog_covers_1_through_16() -> None:
    ids = tuple(entry.exit_id for entry in DIST_SCALE_X_EXIT_CRITERION_CATALOG)
    assert ids == EXIT_CRITERION_IDS


def test_l1_mandatory_backends_declare_qualification_owner() -> None:
    by_id = {entry.backend_id: entry for entry in DIST_SCALE_X_BACKEND_CATALOG}
    for mb_id in MANDATORY_BACKEND_IDS:
        owner = by_id[mb_id].qualification_owner.strip()
        assert owner, f"{mb_id} missing qualification_owner"
        assert by_id[mb_id].qualification_class == BackendQualificationClass.MANDATORY_SCALE


def test_l1_catalog_mappings_validate_without_violations() -> None:
    assert validate_dist_scale_x_catalog_mappings() == ()


def test_l1_backend_classes_use_m_r_f_e_u_only() -> None:
    allowed = {c.value for c in BackendQualificationClass}
    for entry in DIST_SCALE_X_BACKEND_CATALOG:
        assert entry.qualification_class.value in allowed


def test_l1_historical_evidence_uses_reconciliation_taxonomy() -> None:
    allowed = {"REUSE", "RE-VALIDATE", "NOT SUFFICIENT"}
    for entry in DIST_SCALE_X_HISTORICAL_EVIDENCE_CATALOG:
        assert entry.reconciliation in allowed


def test_l1_thresholds_remain_not_locked() -> None:
    assert DIST_SCALE_X_THRESHOLD_LOCK_STATE is ThresholdLockState.NOT_LOCKED


def test_l1_p1_selection_catalog_is_l1_l2_only() -> None:
    layers = {entry.layer for entry in DIST_SCALE_X_P1_SELECTION_CATALOG}
    assert layers == {
        DistScaleXQualificationLayer.L1_CATALOG,
        DistScaleXQualificationLayer.L2_EVIDENCE_CONTRACT,
    }
    for entry in DIST_SCALE_X_P1_SELECTION_CATALOG:
        assert not entry.requires_qual_infra
        assert not entry.requires_threshold_lock


def test_l1_no_shadow_scheduler_dispatcher_retry_authority_in_p1_package() -> None:
    violations = scan_dist_scale_x_authority_surface(_PACKAGE_DIR)
    assert violations == ()


def test_l1_every_exit_criterion_references_known_workloads_or_documentation_only() -> None:
    wl_ids = {e.workload_id for e in DIST_SCALE_X_WORKLOAD_CATALOG}
    dim_ids = {e.dimension_id for e in DIST_SCALE_X_DIMENSION_CATALOG}
    for entry in DIST_SCALE_X_EXIT_CRITERION_CATALOG:
        for wl in entry.workload_ids:
            assert wl in wl_ids
        for dim in entry.dimension_ids:
            assert dim in dim_ids
