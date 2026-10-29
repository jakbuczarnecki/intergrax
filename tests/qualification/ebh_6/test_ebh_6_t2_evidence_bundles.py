# © Artur Czarnecki. All rights reserved.

"""T2-AUD-03 — bundle review completeness vs 133× A–J pairs."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_6._aj_class_gate_scan import iter_applicable_layer_class_pairs
from tests.qualification.ebh_6._l3_aj_group_reconciliation import AjControlOutcome, aj_group_by_layer_class
from tests.qualification.ebh_6._pre_cert_evidence_export import pre_cert_export_metadata
from tests.qualification.ebh_6._t2_evidence_bundles import (
    ImplementerReview,
    all_evidence_bundles,
    bundle_has_non_register_behavioral_gate,
    bundle_review_disposition,
    mandatory_insufficient_pairs,
    pair_to_bundle_id_map,
    resolve_bundle_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh6_t2_bundle_review_complete() -> None:
    pairs = iter_applicable_layer_class_pairs()
    assert len(pairs) == 133
    mapping = pair_to_bundle_id_map()
    assert len(mapping) == 133
    for layer, cls in pairs:
        assert (layer, cls) in mapping
        review = bundle_review_disposition(layer, cls)
        assert review in ImplementerReview, (layer, cls)
        assert resolve_bundle_id(layer, cls) == mapping[(layer, cls)]
    for bundle in all_evidence_bundles():
        assert bundle.implementer_review in ImplementerReview
        assert bundle.primary_behavioral_gates
        assert bundle.invariant_ref.strip()
        assert bundle_has_non_register_behavioral_gate(bundle), bundle.bundle_id
    meta = pre_cert_export_metadata()
    assert "must not issue" in meta["independent_disposition"]
    insufficient = mandatory_insufficient_pairs()
    groups = aj_group_by_layer_class()
    for key in insufficient:
        group = groups[key]
        assert group.outcome == AjControlOutcome.EVIDENCE_REQUIRED, group.group_id
    for layer, cls in pairs:
        if bundle_review_disposition(layer, cls) == ImplementerReview.INSUFFICIENT:
            assert groups[(layer, cls)].outcome == AjControlOutcome.EVIDENCE_REQUIRED
        elif (layer, cls) == ("semantic_core", "F"):
            assert groups[(layer, cls)].outcome == AjControlOutcome.FINDING
