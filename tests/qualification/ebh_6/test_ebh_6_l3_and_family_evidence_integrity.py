# © Artur Czarnecki. All rights reserved.

"""Protocol 03 remediation — L3 reachability and family L9 evidence mechanical gates (R3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG
from tests.qualification.ebh_6._family_l9_evidence import (
    FAMILY_L9_EVIDENCE,
    FamilyL9EvidenceDisposition,
    assert_family_l9_bundles_complete,
    assert_family_l9_reconciliation_complete,
    family_l9_bundle_by_id,
)
from tests.qualification.ebh_6._findings_register import FindingStatus, all_hardcoding_findings
from tests.qualification.ebh_6._l3_aj_group_reconciliation import AjControlOutcome
from tests.qualification.ebh_6._l3_reachability_evidence import (
    all_l3_reachability_records,
    assert_l3_reachability_complete,
    l3_records_by_layer,
)
from tests.qualification.ebh_6._layer_evidence import all_layer_evidence

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_ONLY_GATE = (
    "tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py::"
    "test_ebh6_family_row_count_closed_world"
)
_FORBIDDEN_AJ = frozenset({"PASS"})


def test_l3_records_have_entrypoint_and_call_chain() -> None:
    assert_l3_reachability_complete()


def test_l3_samples_not_path_prefix_of_inventory() -> None:
    for layer_row in all_layer_evidence():
        inventory = layer_row.closed_world_paths
        samples = layer_row.production_effective_samples
        if len(inventory) <= 3 or not samples:
            continue
        assert tuple(samples) != tuple(inventory[: len(samples)]), (
            f"{layer_row.finding_id}: production_effective_samples must not be inventory prefix"
        )


def test_applicable_aj_not_auto_pass() -> None:
    from tests.qualification.ebh_6._l3_aj_group_reconciliation import assert_aj_group_reconciliation_complete

    assert_aj_group_reconciliation_complete()
    for rec in all_l3_reachability_records():
        for ctrl in rec.aj_controls:
            assert ctrl.outcome.value not in _FORBIDDEN_AJ, rec.mechanism_id
            if ctrl.outcome in (AjControlOutcome.PASS_CANDIDATE, AjControlOutcome.PARTIAL_CANDIDATE):
                assert "group=AJG-" in ctrl.evidence_ref, rec.mechanism_id
                assert "proves=" in ctrl.evidence_ref, rec.mechanism_id


def test_aj_evidence_required_honest_not_zero() -> None:
    from tests.qualification.ebh_6._l3_aj_group_reconciliation import count_evidence_required_groups
    from tests.qualification.ebh_6._l3_reachability_evidence import count_l3_aj_outcomes

    assert count_evidence_required_groups() == 0
    counts = count_l3_aj_outcomes()
    assert counts.get("EVIDENCE_REQUIRED", 0) == 0, counts
    assert counts.get("FINDING", 0) >= 1, counts


def test_layer_register_not_false_resolved() -> None:
    layer_rows = [row for row in all_hardcoding_findings() if row.finding_id.startswith("EBH6-LAYER-")]
    assert len(layer_rows) == 16
    for row in layer_rows:
        assert row.status == FindingStatus.CANDIDATE_RESOLVED, row.finding_id
        assert "_l3_reachability_evidence.py" in row.location


def test_family_blockers_have_non_catalog_gates() -> None:
    for bundle in FAMILY_L9_EVIDENCE:
        if not bundle.blocker_id:
            continue
        assert bundle.gate_test_ids, bundle.blocker_id
        for node in bundle.gate_test_ids:
            assert node != _CATALOG_ONLY_GATE, bundle.blocker_id


def test_family_gate_nodes_exist_and_are_tests() -> None:
    for bundle in FAMILY_L9_EVIDENCE:
        for node_id in bundle.gate_test_ids:
            path_part = node_id.split("::", 1)[0]
            assert (_REPO_ROOT / path_part).is_file(), node_id


def test_family_l9_reconciliation_complete() -> None:
    assert_family_l9_reconciliation_complete()


def test_family_metadata_gates_not_pass_candidate() -> None:
    for bundle in FAMILY_L9_EVIDENCE:
        if bundle.l9_evidence_disposition == FamilyL9EvidenceDisposition.PASS_CANDIDATE:
            for node in bundle.gate_test_ids:
                assert "mandatory_docs" not in node
                assert "findings_unclassified" not in node
                assert "prohibited_unresolved" not in node


def test_sys_inv_refs_present_per_layer_and_family() -> None:
    for layer in EBH6_AUDIT_LAYERS:
        records = l3_records_by_layer(layer)
        assert len(records) >= 3, layer
    for bundle in FAMILY_L9_EVIDENCE:
        if bundle.blocker_id:
            assert bundle.sys_inv_refs, bundle.blocker_id


def test_family_catalog_matches_l9_bundle_gates() -> None:
    assert_family_l9_bundles_complete()
    bundles = family_l9_bundle_by_id()
    for row in EBH6_FAMILY_CATALOG:
        bundle = bundles[row.family_id]
        expected = bundle.gate_test_ids if bundle.gate_test_ids else row.gate_test_ids
        assert row.gate_test_ids == expected, row.family_id
        assert row.implementer_proposed_disposition == bundle.implementer_proposed_disposition
