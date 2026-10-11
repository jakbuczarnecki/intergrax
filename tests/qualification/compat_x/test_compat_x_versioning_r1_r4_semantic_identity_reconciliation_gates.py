# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1-R4 mechanical legacy P0 → current class-field semantic identity reconciliation."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_closed_world import (
    build_closed_world_report,
    build_closed_world_report_with_extra_candidates,
    discover_candidates_from_source,
)
from tests.qualification.compat_x._compat_x_discovery import discover_class_field_versions_from_source
from tests.qualification.compat_x._compat_x_semantic_identity_reconciliation import (
    ACCEPTED_P0_EXCLUSIONS,
    ACCEPTED_P0_RAW_CANDIDATES,
    ACCEPTED_P0_SEMANTIC_SURFACES,
    _NON_CLASS_FIELD_KINDS_PRESERVED,
    build_class_field_reconciliations_from_report,
    build_deterministic_semantic_identity_reconciliation,
    build_legacy_semantic_collision_groups,
    derived_semantic_split_delta,
    project_legacy_p0_class_field_semantic_identity,
    project_legacy_p0_semantic_identity_from_candidate,
    semantic_surface_counts_by_discovery_kind,
)
from tests.qualification.compat_x._compat_x_types import DiscoveryCandidate
from tests.qualification.compat_x._compat_x_versioning_gates import build_r1_qualification_report

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_MODULE_A = "synthetic/qualification/r1_r4_identity_probe_a.py"
_MODULE_B = "synthetic/qualification/r1_r4_identity_probe_b.py"


def test_cx_r1_r4_closure_mechanical_inventory_and_delta() -> None:
    report = build_closed_world_report()
    closure = build_deterministic_semantic_identity_reconciliation()

    assert len(report.raw_candidates) == ACCEPTED_P0_RAW_CANDIDATES
    assert len(report.exclusions) == ACCEPTED_P0_EXCLUSIONS
    assert report.unclassified_candidate_ids == frozenset()
    assert closure.legacy_projected_semantic_count == ACCEPTED_P0_SEMANTIC_SURFACES
    assert closure.current_semantic_count == len(report.semantic_surfaces)
    assert closure.semantic_split_delta == closure.current_semantic_count - closure.legacy_projected_semantic_count
    assert closure.semantic_split_delta == derived_semantic_split_delta(closure.collision_groups)
    assert closure.unexplained_added_semantic_identities == frozenset()
    assert closure.unexplained_removed_semantic_identities == frozenset()
    assert closure.collision_group_member_total == sum(g.member_count for g in closure.collision_groups)
    assert (
        closure.current_semantic_count - closure.legacy_projected_semantic_count
        == closure.semantic_split_delta
    )


def test_cx_r1_r4_collision_groups_deterministic() -> None:
    closure = build_deterministic_semantic_identity_reconciliation()
    assert closure.collision_group_count == len(closure.collision_groups)
    assert closure.collision_group_count > 0
    total_collapsed = sum(g.member_count - 1 for g in closure.collision_groups)
    assert total_collapsed == closure.semantic_split_delta


def test_cx_r1_r4_non_class_field_families_unchanged() -> None:
    counts = semantic_surface_counts_by_discovery_kind(build_closed_world_report())
    for kind in _NON_CLASS_FIELD_KINDS_PRESERVED:
        assert kind in counts
    assert counts["defect.persisted_without_version"] == 5


def test_cx_r1_r4_adversarial_a_same_literal_two_contracts() -> None:
    source = '''
class A:
    schema_version = "v1"

class B:
    schema_version = "v1"
'''
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    reconciliations = build_class_field_reconciliations_from_report(
        build_closed_world_report_with_extra_candidates(tuple(found))
    )
    probe = [r for r in reconciliations if r.owner_module_path == _MODULE_A]
    assert len(probe) == 2
    assert probe[0].legacy_identity == probe[1].legacy_identity == "schema.literal:v1"
    groups = build_legacy_semantic_collision_groups(tuple(probe))
    assert len(groups) == 1
    assert groups[0].member_count == 2
    assert derived_semantic_split_delta(groups) == 1


def test_cx_r1_r4_adversarial_b_three_contracts_same_inline_literal() -> None:
    source = '''
class A:
    schema_version = "v1"

class B:
    schema_version = "v1"

class C:
    schema_version = "v1"
'''
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    reconciliations = tuple(
        r
        for r in build_class_field_reconciliations_from_report(
            build_closed_world_report_with_extra_candidates(tuple(found))
        )
        if r.owner_module_path == _MODULE_A
    )
    assert len(reconciliations) == 3
    assert len({r.legacy_identity for r in reconciliations}) == 1
    assert len({r.current_identity for r in reconciliations}) == 3
    groups = build_legacy_semantic_collision_groups(reconciliations)
    assert derived_semantic_split_delta(groups) == 2


def test_cx_r1_r4_adversarial_c_different_versions_no_collision() -> None:
    source = '''
class A:
    schema_version = "v1"

class B:
    schema_version = "v2"
'''
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    reconciliations = tuple(
        r
        for r in build_class_field_reconciliations_from_report(
            build_closed_world_report_with_extra_candidates(tuple(found))
        )
        if r.owner_module_path == _MODULE_A
    )
    groups = build_legacy_semantic_collision_groups(reconciliations)
    assert groups == ()
    assert derived_semantic_split_delta(groups) == 0


def test_cx_r1_r4_adversarial_d_duplicate_discovery_single_semantic() -> None:
    source = '''
from typing import Literal

PROBE_SCHEMA_VERSION = "probe.v1"

class ProbeContract:
    schema_version: Literal["probe.v1"] = PROBE_SCHEMA_VERSION
'''
    module = "synthetic/qualification/r1_r4_dedup_probe.py"
    primary = tuple(discover_candidates_from_source(module, source))
    target = next(c for c in primary if c.discovery_kind == "class.field.version")
    duplicate = DiscoveryCandidate(
        candidate_id=f"{target.candidate_id}.dup_evidence",
        path=target.path,
        discovery_kind="class.field.version",
        discovered_signal=target.discovered_signal,
        semantic_identity=target.semantic_identity,
        current_version=target.current_version,
        version_source=target.version_source,
    )
    report = build_closed_world_report_with_extra_candidates((duplicate,))
    matches = [s for s in report.semantic_surfaces if s.semantic_identity == target.semantic_identity]
    assert len(matches) == 1
    reconciliations = [
        r for r in build_class_field_reconciliations_from_report(report) if r.owner_module_path == module
    ]
    assert len(reconciliations) == 1


def test_cx_r1_r4_adversarial_e_unresolved_field_deterministic_legacy() -> None:
    source = '''
class UnresolvedProbe:
    schema_version = MISSING_MODULE_CONSTANT
'''
    module = "synthetic/qualification/r1_r4_unresolved_probe.py"
    found = tuple(discover_class_field_versions_from_source(module, source))
    assert len(found) == 1
    candidate = next(iter(found))
    legacy = project_legacy_p0_semantic_identity_from_candidate(candidate)
    assert legacy == candidate.semantic_identity
    assert legacy.startswith("class.field:")
    assert project_legacy_p0_class_field_semantic_identity(candidate.semantic_identity, None) == legacy


def test_cx_r1_r4_public_compliance_and_persisted_findings() -> None:
    gate = build_r1_qualification_report()
    assert gate.public_stable_noncompliant_version_identity_count == 0
    assert gate.persisted_without_version_finding_count == 5


def test_cx_r1_r4_cross_contract_merge_zero() -> None:
    report = build_closed_world_report()
    identities = [s.semantic_identity for s in report.semantic_surfaces]
    assert len(identities) == len(set(identities))
    assert all(not identity.startswith("schema.literal:") for identity in identities)
