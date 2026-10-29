# © Artur Czarnecki. All rights reserved.

"""E01/E02 — strengthened T2 evidence for confirmed audit finding gaps (behavioral, not register-only)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_5 import test_ebh_5_reachability_gates as reach_gates
from tests.qualification.ebh_6._aj_class_gate_scan import (
    assert_no_forbidden_vendor_literals_in_text,
    iter_applicable_layer_class_pairs,
    run_class_gate_scan,
    scan_class_b_hardcoding,
    scan_class_e_shadow_store,
)
from tests.qualification.ebh_6._closed_world_discovery import (
    assert_ebh6_sanctioned_composition_paths_resolvable,
    assert_ebh6_sqlite_annex_accountability,
)
from tests.qualification.ebh_6._t2_evidence_bundles import EvidenceBundleId, resolve_bundle_id

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh6_e01_platform_hardcoding_invokes_class_b_scans() -> None:
    for layer in ("execution", "semantic_core", "agent_distribution"):
        scan_class_b_hardcoding(layer)


def test_ebh6_e01_persistence_sqlite_closed_world_no_shadow_connects() -> None:
    assert_ebh6_sqlite_annex_accountability()
    for layer in (
        "state_persistence",
        "application_services",
        "events_traceability",
        "reliability_recovery",
    ):
        scan_class_e_shadow_store(layer)


def test_ebh6_e01_sanctioned_composition_paths_resolvable_on_disk() -> None:
    assert_ebh6_sanctioned_composition_paths_resolvable()


def test_ebh6_e01_platform_fail_closed_reach_gates() -> None:
    reach_gates.test_ebh5_reach_integration_unknown_slug_fail_closed()
    reach_gates.test_ebh5_reach_observability_fail_closed_without_role()
    for layer, cls in iter_applicable_layer_class_pairs():
        if cls != "C":
            continue
        if resolve_bundle_id(layer, cls) != EvidenceBundleId.PLATFORM_FAIL_CLOSED:
            continue
        run_class_gate_scan(layer, cls)


def test_ebh6_e01_negative_class_b_hardcoding_literal() -> None:
    with pytest.raises(AssertionError, match="forbidden vendor literal"):
        assert_no_forbidden_vendor_literals_in_text('provider = "openai"\n', "fake/execution/module.py")


def test_ebh6_e01_negative_sanctioned_composition_missing_path() -> None:
    fake = "intergrax/__ebh6_qualification_missing_composition_root__.py"
    with pytest.raises(AssertionError, match="missing composition paths on disk"):
        assert_ebh6_sanctioned_composition_paths_resolvable(
            qualification_extra_paths=frozenset({fake}),
        )
