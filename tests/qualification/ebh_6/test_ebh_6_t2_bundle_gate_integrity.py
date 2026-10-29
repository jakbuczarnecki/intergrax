# © Artur Czarnecki. All rights reserved.

"""R4 — T2 bundle primary gate integrity (behavioral nodeids, not register-only)."""

from __future__ import annotations

from typing import Final

import pytest
from tests.qualification.ebh_6._t2_evidence_bundles import (
    EvidenceBundleId,
    ImplementerReview,
    all_evidence_bundles,
    assert_resolvable_pytest_node_id,
    bundle_has_non_register_behavioral_gate,
    is_register_only_pytest_gate,
    pair_to_bundle_id_map,
    pytest_primary_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_CRITICAL_BUNDLES: frozenset[EvidenceBundleId] = frozenset(
    {
        EvidenceBundleId.PLATFORM_TYPING_SCAN,
        EvidenceBundleId.PLATFORM_HARDCODING,
        EvidenceBundleId.PLATFORM_FAIL_CLOSED,
        EvidenceBundleId.REPLACEABILITY_J,
        EvidenceBundleId.PERSISTENCE_SQLITE,
        EvidenceBundleId.L3_REACHABILITY,
        EvidenceBundleId.SANCTIONED_COMPOSITION,
        EvidenceBundleId.GOVERNANCE_BOUNDARY,
    },
)

_RPLG_SUBSTR = "test_ebh5_rpl_g_config_x_active_blockers_remain_empty"


def test_ebh6_t2_all_bundles_have_behavioral_nodeids() -> None:
    for bundle in all_evidence_bundles():
        assert bundle.primary_behavioral_gates, bundle.bundle_id
        assert pytest_primary_gates(bundle), bundle.bundle_id


def test_ebh6_t2_all_bundles_reject_register_only_primary_substitution() -> None:
    for bundle in all_evidence_bundles():
        assert bundle_has_non_register_behavioral_gate(bundle), bundle.bundle_id


def test_ebh6_t2_critical_bundles_not_register_only() -> None:
    for bundle in all_evidence_bundles():
        if bundle.bundle_id not in _CRITICAL_BUNDLES:
            continue
        assert bundle_has_non_register_behavioral_gate(bundle), bundle.bundle_id


def test_ebh6_t2_primary_gates_resolve_to_real_tests() -> None:
    for bundle in all_evidence_bundles():
        for gate in pytest_primary_gates(bundle):
            assert_resolvable_pytest_node_id(gate)


def test_ebh6_t2_fake_nodeid_fails_resolution() -> None:
    with pytest.raises(AssertionError, match="missing gate"):
        assert_resolvable_pytest_node_id(
            "tests/qualification/ebh_6/test_ebh_6_t2_bundle_gate_integrity.py::test_ebh6_t2_fake_gate_does_not_exist",
        )


def test_ebh6_t2_register_only_marker_detects_substitution() -> None:
    reg = "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero"
    behavioral = (
        "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py"
        "::test_ebh5_rpl_g_config_x_active_blockers_remain_empty"
    )
    assert is_register_only_pytest_gate(reg)
    assert not is_register_only_pytest_gate(behavioral)


def test_ebh6_t2_replaceability_j_invokes_rpl_g() -> None:
    for bundle in all_evidence_bundles():
        if bundle.bundle_id != EvidenceBundleId.REPLACEABILITY_J:
            continue
        assert any(_RPLG_SUBSTR in g for g in bundle.primary_behavioral_gates)


def test_ebh6_t2_accepted_review_not_independent_pass() -> None:
    for bundle in all_evidence_bundles():
        assert bundle.implementer_review == ImplementerReview.ACCEPTED
        assert bundle.implementer_review.value != "PASS"
        assert bundle.implementer_review.value != "INDEPENDENT_PASS"


_E01_AUDIT_GATES: Final[frozenset[EvidenceBundleId]] = frozenset(
    {
        EvidenceBundleId.PLATFORM_HARDCODING,
        EvidenceBundleId.PERSISTENCE_SQLITE,
        EvidenceBundleId.SANCTIONED_COMPOSITION,
        EvidenceBundleId.PLATFORM_FAIL_CLOSED,
    },
)
_E01_SUBSTR = "test_ebh_6_e01_e02_audit_findings_evidence.py"


def test_ebh6_t2_e01_audit_finding_bundles_reference_behavioral_e01_gates() -> None:
    for bundle in all_evidence_bundles():
        if bundle.bundle_id not in _E01_AUDIT_GATES:
            continue
        assert any(_E01_SUBSTR in g for g in bundle.primary_behavioral_gates), bundle.bundle_id


def test_ebh6_t2_pair_map_covers_133_bundles() -> None:
    mapping = pair_to_bundle_id_map()
    assert len(mapping) == 133
    assert len({bid for bid in mapping.values()}) >= len(EvidenceBundleId) - 2
