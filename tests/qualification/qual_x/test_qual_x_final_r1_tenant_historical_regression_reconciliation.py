# © Artur Czarnecki. All rights reserved.

"""QUAL-X-FINAL-R1 — historical TENANT-X evidence vs current-HEAD regression classification."""

from __future__ import annotations

import pytest

from tests.qualification.qual_x._p2_ingress_catalog import QUAL_X_P2_INGRESS_CATALOG
from tests.qualification.qual_x._p5_freeze_selection_support import historical_linkage_cannot_mandatory_pass
from tests.qualification.qual_x._p5_freeze_selection_catalog import QUAL_X_P5_FREEZE_SELECTION_MANIFEST
from tests.qualification.tenant_x._final_certification import build_final_qualification_report
from tests.qualification.tenant_x._final_production_delta import (
    TENANT_X_FINAL_ACCEPTED_EVIDENCE_SHA,
    build_final_production_delta_evidence,
)
from tests.qualification.tenant_x._final_wave_evidence import current_git_head_sha

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_TENANT_FINAL_FAST = "tests/qualification/tenant_x/test_tenant_x_final_fast.py"
_TENANT_FINAL_CURRENT = "tests/qualification/tenant_x/test_tenant_x_final_current_head.py"


def test_qual_x_final_r1_p2_tenant_final_fast_nodes_remain_collectable() -> None:
    g8 = next(entry for entry in QUAL_X_P2_INGRESS_CATALOG if entry.ingress_id == "G8")
    assert any(_TENANT_FINAL_FAST in node for node in g8.primary_proof_node_ids)


def test_qual_x_final_r1_p5_f08_tenant_linkage_stays_historical_until_final_replay() -> None:
    f08 = next(e for e in QUAL_X_P5_FREEZE_SELECTION_MANIFEST if e.selection_id == "QUAL-X-P5-F08-ADVERSARIAL-NEGATIVES")
    assert "HISTORICAL" in f08.expected_evidence_class.value
    assert historical_linkage_cannot_mandatory_pass(f08)


def test_qual_x_final_r1_historical_delta_bound_to_acceptance_sha() -> None:
    evidence = build_final_production_delta_evidence()
    assert evidence is not None
    assert evidence.head_sha == TENANT_X_FINAL_ACCEPTED_EVIDENCE_SHA
    assert evidence.production_delta_count == 0


def test_qual_x_final_r1_current_head_behavioral_ready_without_zero_delta_since_rejected_base() -> None:
    drift = build_final_production_delta_evidence(head_sha=current_git_head_sha())
    assert drift is not None
    historical = build_final_production_delta_evidence()
    assert historical is not None and historical.production_delta_count == 0
    report = build_final_qualification_report()
    assert report.production_delta == 0
    if drift.production_delta_count > 0:
        assert drift.head_sha != TENANT_X_FINAL_ACCEPTED_EVIDENCE_SHA
    assert report.final_ready_for_audit


def test_qual_x_final_r1_mandatory_current_head_module_targets_include_tenant_final_surface() -> None:
    from tests.qualification.qual_x._p5_freeze_selection_catalog import (
        qual_x_p5_dimension_a_pytest_module_targets,
    )

    targets = qual_x_p5_dimension_a_pytest_module_targets()
    assert _TENANT_FINAL_FAST in targets
    assert _TENANT_FINAL_CURRENT in targets
