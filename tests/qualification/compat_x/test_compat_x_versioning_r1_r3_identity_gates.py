# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1-R3 semantic surface identity preservation (contract-specific, not version-value)."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_closed_world import (
    build_closed_world_report,
    build_closed_world_report_with_extra_candidates,
    discover_candidates_from_source,
)
from tests.qualification.compat_x._compat_x_discovery import discover_class_field_versions_from_source
from tests.qualification.compat_x._compat_x_types import DiscoveryCandidate

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_ACCEPTED_P0_SEMANTIC_SURFACES = 1879
_ACCEPTED_P0_RAW_CANDIDATES = 7266
_ACCEPTED_P0_EXCLUSIONS = 5289
# Contract-specific class.field identities split 98 erroneous accepted-P0 value-based merges.
_R1_R3_CORRECTED_SEMANTIC_SURFACES = _ACCEPTED_P0_SEMANTIC_SURFACES + 98

_MODULE_A = "synthetic/qualification/r1_r3_identity_probe_a.py"
_MODULE_B = "synthetic/qualification/r1_r3_identity_probe_b.py"


def test_cx_r1_r3_identity_a_two_classes_same_literal() -> None:
    source = '''
class A:
    schema_version = "v1"

class B:
    schema_version = "v1"
'''
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    class_fields = [c for c in found if c.discovery_kind == "class.field.version"]
    assert len(class_fields) == 2
    identities = {c.semantic_identity for c in class_fields}
    assert len(identities) == 2
    assert all(c.current_version == "v1" for c in class_fields)


def test_cx_r1_r3_identity_b_two_classes_same_module_constant() -> None:
    source = '''
COMMON_SCHEMA_VERSION = "v1"

class A:
    schema_version = COMMON_SCHEMA_VERSION

class B:
    schema_version = COMMON_SCHEMA_VERSION
'''
    found = discover_class_field_versions_from_source(_MODULE_A, source)
    class_fields = [c for c in found if c.discovery_kind == "class.field.version"]
    assert len(class_fields) == 2
    assert len({c.semantic_identity for c in class_fields}) == 2


def test_cx_r1_r3_identity_c_same_version_string_two_modules() -> None:
    source_a = '''
class A:
    schema_version = "shared.version.v1"
'''
    source_b = '''
class B:
    schema_version = "shared.version.v1"
'''
    a_fields = list(discover_class_field_versions_from_source(_MODULE_A, source_a))
    b_fields = list(discover_class_field_versions_from_source(_MODULE_B, source_b))
    assert len(a_fields) == 1
    assert len(b_fields) == 1
    assert a_fields[0].semantic_identity != b_fields[0].semantic_identity
    assert a_fields[0].current_version == b_fields[0].current_version


def test_cx_r1_r3_identity_d_same_field_multiple_evidence_merges_once() -> None:
    source = '''
from typing import Literal

PROBE_SCHEMA_VERSION = "probe.v1"

class ProbeContract:
    schema_version: Literal["probe.v1"] = PROBE_SCHEMA_VERSION
'''
    module = "synthetic/qualification/r1_r3_dedup_probe.py"
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
    assert len(matches[0].evidence_kinds) >= 1


def test_cx_r1_r3_cross_contract_version_value_collisions_zero() -> None:
    report = build_closed_world_report()
    identities = [s.semantic_identity for s in report.semantic_surfaces]
    assert len(identities) == len(set(identities))
    assert all(not identity.startswith("schema.literal:") for identity in identities)


def test_cx_r1_r3_p0_inventory_reconciliation() -> None:
    report = build_closed_world_report()
    assert len(report.raw_candidates) == _ACCEPTED_P0_RAW_CANDIDATES
    assert len(report.exclusions) == _ACCEPTED_P0_EXCLUSIONS
    assert report.unclassified_candidate_ids == frozenset()
    assert len(report.semantic_surfaces) == _R1_R3_CORRECTED_SEMANTIC_SURFACES
    split_from_p0_value_merge = len(report.semantic_surfaces) - _ACCEPTED_P0_SEMANTIC_SURFACES
    assert split_from_p0_value_merge == 98
