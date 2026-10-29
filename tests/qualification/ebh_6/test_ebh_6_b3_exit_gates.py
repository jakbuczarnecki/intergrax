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

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh6_b3_evidence_required_groups_zero_at_handoff() -> None:
    groups = aj_group_by_layer_class()
    ev = [g for g in groups.values() if g.outcome == AjControlOutcome.EVIDENCE_REQUIRED]
    assert not ev, [g.group_id for g in ev]
    assert count_evidence_required_groups() == 0


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
        expected = layer_register_status_for_layer(layer)
        row = next(r for r in layer_rows if r.layer == layer)
        assert row.status == expected, layer
        has_t1 = any(
            g.outcome == AjControlOutcome.EVIDENCE_REQUIRED
            for (ly, _), g in aj_group_by_layer_class().items()
            if ly == layer
        )
        if has_t1:
            assert expected == FindingStatus.L9_EVIDENCE_SUBMITTED, layer
        else:
            assert expected == FindingStatus.CANDIDATE_RESOLVED, layer


def test_ebh6_protocol03_contract_tier_coverage() -> None:
    from tests.qualification.ebh_6._aj_class_gate_scan import scanner_effective
    from tests.qualification.ebh_6._tier_audit import EvidenceTier, evidence_tier_for_pair, tier_census

    groups = aj_group_by_layer_class()
    census = tier_census()
    assert len(groups) == 133
    assert census[EvidenceTier.T1.value] == count_evidence_required_groups()
    for (layer, cls), group in groups.items():
        tier = evidence_tier_for_pair(layer, cls)
        if tier == EvidenceTier.T1:
            assert group.outcome == AjControlOutcome.EVIDENCE_REQUIRED, group.group_id
        elif tier == EvidenceTier.T2:
            assert group.outcome == AjControlOutcome.PARTIAL_CANDIDATE, group.group_id
            assert scanner_effective(layer, cls), group.group_id
            assert group.scope_limitation.strip()
        else:
            assert group.outcome in (
                AjControlOutcome.PASS_CANDIDATE,
                AjControlOutcome.FINDING,
            ), group.group_id
        if group.outcome == AjControlOutcome.PARTIAL_CANDIDATE:
            assert scanner_effective(layer, cls), f"no-op PARTIAL: {group.group_id}"
        if group.outcome == AjControlOutcome.FINDING:
            assert any("register" in ref.lower() or "_findings_register" in ref for ref in group.evidence_refs), (
                group.group_id
            )


def test_ebh6_b3_pair_count_matches_l9a() -> None:
    assert len(iter_applicable_layer_class_pairs()) == 133
