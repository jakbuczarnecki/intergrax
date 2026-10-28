# © Artur Czarnecki. All rights reserved.

"""EBH-6 B3 exit gates — 133 group reconciliation + layer status (EX-01 honest)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_6._aj_class_gate_scan import class_gate_node_id, iter_applicable_layer_class_pairs
from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._findings_register import (
    FindingStatus,
    all_hardcoding_findings,
    layer_register_status_for_layer,
)
from tests.qualification.ebh_6._l3_aj_group_reconciliation import (
    AjControlOutcome,
    aj_group_by_layer_class,
    count_evidence_required_groups,
)
from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh6_b3_evidence_required_groups_have_blockers() -> None:
    groups = aj_group_by_layer_class()
    ev = [g for g in groups.values() if g.outcome == AjControlOutcome.EVIDENCE_REQUIRED]
    assert ev, "Protocol 03 remediation must not mask mandatory gaps with zero EV"
    for group in ev:
        assert group.mandatory_blocker, group.group_id
        assert group.mandatory_blocker.startswith("EBH6-P03-AJ-"), group.group_id


def test_ebh6_b3_evidence_required_not_mechanically_zero() -> None:
    assert count_evidence_required_groups() >= 100


def test_ebh6_b3_all_applicable_groups_have_class_gate_ref() -> None:
    groups = aj_group_by_layer_class()
    assert len(groups) == 133
    for (layer, cls), group in groups.items():
        gate = class_gate_node_id(layer, cls)
        assert gate in group.evidence_refs, group.group_id


def test_ebh6_b3_q2d2_semantic_core_f_is_finding() -> None:
    group = aj_group_by_layer_class()[("semantic_core", "F")]
    assert group.outcome == AjControlOutcome.FINDING, group.outcome
    assert "Q2-D2" in group.invariant_proves or "EBH6-DEBT-Q2-D2" in group.invariant_proves


def test_ebh6_b3_partial_candidate_requires_explicit_adjudication() -> None:
    for group in aj_group_by_layer_class().values():
        if group.outcome == AjControlOutcome.PARTIAL_CANDIDATE:
            assert any("test_ebh6_aj_class_gate" in ref for ref in group.evidence_refs), group.group_id
            assert "mandatory blocker" not in group.invariant_proves.lower()


def test_ebh6_b3_sixteen_layers_status_matches_group_outcomes() -> None:
    layer_rows = [row for row in all_hardcoding_findings() if row.finding_id.startswith("EBH6-LAYER-")]
    assert len(layer_rows) == 16
    for layer in EBH6_AUDIT_LAYERS:
        applicable = [aj for aj in layer_aj_applicability(layer) if aj.applicable]
        layer_ev = any(
            aj_group_by_layer_class()[(layer, aj.detection_class)].outcome == AjControlOutcome.EVIDENCE_REQUIRED
            for aj in applicable
        )
        expected = FindingStatus.L9_EVIDENCE_SUBMITTED if layer_ev else FindingStatus.CANDIDATE_RESOLVED
        assert layer_register_status_for_layer(layer) == expected, layer


def test_ebh6_b3_pair_count_matches_l9a() -> None:
    assert len(iter_applicable_layer_class_pairs()) == 133
