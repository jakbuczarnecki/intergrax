# © Artur Czarnecki. All rights reserved.

"""R4 — T2 bundle primary gate integrity (behavioral nodeids, not register-only)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_6._t2_evidence_bundles import (
    EvidenceBundleId,
    all_evidence_bundles,
    pair_to_bundle_id_map,
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
        pytest_gates = [g for g in bundle.primary_behavioral_gates if "::" in g]
        assert pytest_gates, bundle.bundle_id


def test_ebh6_t2_critical_bundles_not_register_only() -> None:
    register_only = ("_findings_register.py", "hardcoding_register_integrity")
    for bundle in all_evidence_bundles():
        if bundle.bundle_id not in _CRITICAL_BUNDLES:
            continue
        gates = bundle.primary_behavioral_gates
        assert any("::" in g and register_only[0] not in g for g in gates), bundle.bundle_id


def test_ebh6_t2_replaceability_j_invokes_rpl_g() -> None:
    for bundle in all_evidence_bundles():
        if bundle.bundle_id != EvidenceBundleId.REPLACEABILITY_J:
            continue
        assert any(_RPLG_SUBSTR in g for g in bundle.primary_behavioral_gates)


def test_ebh6_t2_pair_map_covers_133_bundles() -> None:
    mapping = pair_to_bundle_id_map()
    assert len(mapping) == 133
    assert len({bid for bid in mapping.values()}) >= len(EvidenceBundleId) - 2
