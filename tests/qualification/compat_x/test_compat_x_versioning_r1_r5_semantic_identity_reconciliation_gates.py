# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1-R5 exact historical P0 AST projection → current class-field semantic identity reconciliation."""

from __future__ import annotations

import ast

import pytest

from tests.qualification.compat_x._compat_x_closed_world import (
    build_closed_world_report,
    build_closed_world_report_with_extra_candidates,
    discover_candidates_from_source,
)
from tests.qualification.compat_x._compat_x_discovery import discover_class_field_versions_from_source
from tests.qualification.compat_x._compat_x_semantic_identity_reconciliation import (
    ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES,
    ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES,
    ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES,
    ACCEPTED_P0_EXCLUSIONS,
    ACCEPTED_P0_SEMANTIC_SURFACES,
    _NON_CLASS_FIELD_KINDS_PRESERVED,
    build_class_field_reconciliations_from_report,
    build_deterministic_semantic_identity_reconciliation,
    build_legacy_semantic_collision_groups,
    derived_semantic_split_delta,
    historical_p0_literal_version,
    locate_class_field_value_node,
    project_legacy_p0_class_field_semantic_identity,
    project_legacy_p0_semantic_identity_from_candidate,
    semantic_surface_counts_by_discovery_kind,
)
from tests.qualification.compat_x._compat_x_types import DiscoveryCandidate
from tests.qualification.compat_x._compat_x_versioning_gates import build_r1_qualification_report

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_MODULE_A = "synthetic/qualification/r1_r5_identity_probe_a.py"
_MODULE_B = "synthetic/qualification/r1_r5_identity_probe_b.py"


def test_cx_r1_r5_closure_mechanical_inventory_and_delta() -> None:
    report = build_closed_world_report()
    closure = build_deterministic_semantic_identity_reconciliation()

    assert len(report.raw_candidates) == ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES
    assert len(report.exclusions) == ACCEPTED_P0_EXCLUSIONS
    assert report.unclassified_candidate_ids == frozenset()
    assert closure.legacy_projected_semantic_count == ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES
    assert closure.legacy_projected_semantic_count == ACCEPTED_P0_SEMANTIC_SURFACES + 1
    assert closure.current_semantic_count == ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES
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


def test_cx_r1_r5_collision_groups_deterministic() -> None:
    closure = build_deterministic_semantic_identity_reconciliation()
    assert closure.collision_group_count == len(closure.collision_groups)
    assert closure.collision_group_count > 0
    total_collapsed = sum(g.member_count - 1 for g in closure.collision_groups)
    assert total_collapsed == closure.semantic_split_delta


def test_cx_r1_r5_non_class_field_families_unchanged() -> None:
    counts = semantic_surface_counts_by_discovery_kind(build_closed_world_report())
    for kind in _NON_CLASS_FIELD_KINDS_PRESERVED:
        assert kind in counts
    assert counts["defect.persisted_without_version"] == 5


def test_cx_r1_r5_adversarial_a_inline_literal() -> None:
    source = '''
class A:
    schema_version = "v1"

class B:
    schema_version = "v1"
'''
    overrides = {_MODULE_A: source}
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    reconciliations = build_class_field_reconciliations_from_report(
        build_closed_world_report_with_extra_candidates(tuple(found)),
        source_by_module=overrides,
    )
    probe = [r for r in reconciliations if r.owner_module_path == _MODULE_A]
    assert len(probe) == 2
    assert probe[0].legacy_identity == probe[1].legacy_identity == "schema.literal:v1"
    groups = build_legacy_semantic_collision_groups(tuple(probe))
    assert len(groups) == 1
    assert groups[0].member_count == 2
    assert derived_semantic_split_delta(groups) == 1


def test_cx_r1_r5_adversarial_b_same_module_constant() -> None:
    source = '''
V_SCHEMA_VERSION = "v1"

class A:
    schema_version = V_SCHEMA_VERSION
'''
    overrides = {_MODULE_A: source}
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    reconciliations = tuple(
        r
        for r in build_class_field_reconciliations_from_report(
            build_closed_world_report_with_extra_candidates(tuple(found)),
            source_by_module=overrides,
        )
        if r.owner_module_path == _MODULE_A
    )
    assert len(reconciliations) == 1
    row = reconciliations[0]
    assert row.legacy_identity.startswith("class.field:")
    assert row.current_version == "v1"
    assert build_legacy_semantic_collision_groups(reconciliations) == ()


def test_cx_r1_r5_adversarial_c_literal_annotation_constant_default() -> None:
    source = '''
from typing import Literal

V_SCHEMA_VERSION = "v1"

class Probe:
    schema_version: Literal["v1"] = V_SCHEMA_VERSION
'''
    module = "synthetic/qualification/r1_r5_literal_ann_probe.py"
    overrides = {module: source}
    found = discover_class_field_versions_from_source(module, source)
    reconciliations = build_class_field_reconciliations_from_report(
        build_closed_world_report_with_extra_candidates(tuple(found)),
        source_by_module=overrides,
    )
    row = next(r for r in reconciliations if r.owner_module_path == module)
    assert row.legacy_identity == f"class.field:{module}:Probe:schema_version"
    assert row.current_version == "v1"


def test_cx_r1_r5_adversarial_d_annotation_only_literal_value_none() -> None:
    source = '''
from typing import Literal

class AnnOnly:
    schema_version: Literal["ann.v1"]
'''
    module = "synthetic/qualification/r1_r5_ann_only_probe.py"
    overrides = {module: source}
    found = discover_class_field_versions_from_source(module, source)
    value_node = locate_class_field_value_node(
        module, "AnnOnly", "schema_version", source_by_module=overrides
    )
    assert value_node is None
    assert historical_p0_literal_version(value_node) is None
    row = next(
        r
        for r in build_class_field_reconciliations_from_report(
            build_closed_world_report_with_extra_candidates(tuple(found)),
            source_by_module=overrides,
        )
        if r.owner_module_path == module
    )
    assert row.legacy_identity.startswith("class.field:")


def test_cx_r1_r5_adversarial_e_spec_version_no_field_exception() -> None:
    source = '''
class Contract:
    spec_version: str = "1.2"
'''
    module = "synthetic/qualification/r1_r5_spec_version_probe.py"
    overrides = {module: source}
    found = discover_class_field_versions_from_source(module, source)
    row = next(
        r
        for r in build_class_field_reconciliations_from_report(
            build_closed_world_report_with_extra_candidates(tuple(found)),
            source_by_module=overrides,
        )
        if r.owner_module_path == module
    )
    assert row.legacy_identity == "schema.literal:1.2"


def test_cx_r1_r5_adversarial_f_unresolved_field_deterministic_legacy() -> None:
    source = '''
class UnresolvedProbe:
    schema_version = MISSING_MODULE_CONSTANT
'''
    module = "synthetic/qualification/r1_r5_unresolved_probe.py"
    overrides = {module: source}
    found = tuple(discover_class_field_versions_from_source(module, source))
    assert len(found) == 1
    candidate = next(iter(found))
    legacy = project_legacy_p0_semantic_identity_from_candidate(candidate, source_by_module=overrides)
    assert legacy == candidate.semantic_identity
    assert legacy.startswith("class.field:")
    assert (
        project_legacy_p0_class_field_semantic_identity(
            candidate.semantic_identity, source_by_module=overrides
        )
        == legacy
    )


def test_cx_r1_r5_historical_projection_independent_of_current_metadata() -> None:
    source = '''
class MetaProbe:
    schema_version = "meta.v1"
'''
    module = "synthetic/qualification/r1_r5_metadata_independence_probe.py"
    overrides = {module: source}
    found = tuple(discover_class_field_versions_from_source(module, source))
    base = next(iter(found))
    variants = (
        DiscoveryCandidate(
            candidate_id=base.candidate_id,
            path=base.path,
            discovery_kind=base.discovery_kind,
            discovered_signal=base.discovered_signal,
            semantic_identity=base.semantic_identity,
            current_version="tampered-version",
            version_source="synthetic:CONST→tampered",
        ),
        DiscoveryCandidate(
            candidate_id=f"{base.candidate_id}.v2",
            path=base.path,
            discovery_kind=base.discovery_kind,
            discovered_signal=f"{base.discovered_signal} tampered",
            semantic_identity=base.semantic_identity,
            current_version=None,
            version_source="no-arrow-source",
        ),
    )
    expected = project_legacy_p0_semantic_identity_from_candidate(base, source_by_module=overrides)
    for variant in variants:
        projected = project_legacy_p0_semantic_identity_from_candidate(
            variant, source_by_module=overrides
        )
        assert projected == expected == "schema.literal:meta.v1"


def test_cx_r1_r5_ast_value_node_used_not_annotation() -> None:
    source = '''
from typing import Literal

class Mixed:
    schema_version: Literal["from.ann"] = "from.value"
'''
    module = "synthetic/qualification/r1_r5_value_vs_ann_probe.py"
    overrides = {module: source}
    node = locate_class_field_value_node(
        module, "Mixed", "schema_version", source_by_module=overrides
    )
    assert isinstance(node, ast.Constant)
    assert historical_p0_literal_version(node) == "from.value"


def test_cx_r1_r5_public_compliance_and_persisted_findings() -> None:
    gate = build_r1_qualification_report()
    assert gate.public_stable_noncompliant_version_identity_count == 0
    assert gate.persisted_without_version_finding_count == 5


def test_cx_r1_r5_cross_contract_merge_zero() -> None:
    report = build_closed_world_report()
    identities = [s.semantic_identity for s in report.semantic_surfaces]
    assert len(identities) == len(set(identities))
    assert all(not identity.startswith("schema.literal:") for identity in identities)
