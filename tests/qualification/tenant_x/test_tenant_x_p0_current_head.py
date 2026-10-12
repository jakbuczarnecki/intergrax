# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0 current-HEAD inventory scan — run sparingly (heavy gate budget)."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._classifiers import uses_generic_safe_fallback
from tests.qualification.tenant_x._inventory import (
    build_closed_world_discovery_report,
    carrier_semantics_report,
    class_field_tenant_surface_ids,
    get_tenant_x_inventory,
    unknown_identity_source_count,
    unknown_propagation_role_count,
)
from tests.qualification.tenant_x._types import TenantCarrierFamily
from tests.qualification.tenant_x._p0_matrix import build_p0_matrix
from tests.qualification.tenant_x._propagation import validate_propagation_edges
from tests.qualification.tenant_x._types import TenantBlockerClass

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_tx_p0_head_q01_inventory_covers_all_candidates() -> None:
    report = build_closed_world_discovery_report()
    discovered = frozenset(c.surface_id for c in report.raw_candidates)
    classified = frozenset(r.surface_id for r in get_tenant_x_inventory())
    assert discovered == classified
    assert report.unrecorded_candidate_ids == frozenset()


def test_tx_p0_head_q02_integrity_no_generic_safe_fallback() -> None:
    violations = [r.surface_id for r in get_tenant_x_inventory() if uses_generic_safe_fallback(r)]
    assert violations == []


def test_tx_p0_head_q03_propagation_edges_reference_known_sources() -> None:
    known = frozenset(r.surface_id for r in get_tenant_x_inventory())
    assert validate_propagation_edges(known) == ()


def test_tx_p0_head_q04_matrix_truthful_unknown_allowed() -> None:
    matrix = build_p0_matrix()
    assert matrix.production_files_scanned > 0
    assert matrix.tenant_related_candidates > 0
    assert matrix.canonical_identity_sources >= 1
    assert unknown_propagation_role_count() == matrix.unknown_propagation_roles
    assert unknown_identity_source_count() == matrix.unknown_identity_sources


def test_tx_p0_head_q05_in_scope_blockers_require_positive_evidence() -> None:
    blockers = [r for r in get_tenant_x_inventory() if r.blocker_class == TenantBlockerClass.IN_SCOPE_BLOCKER]
    for record in blockers:
        joined = "".join(record.evidence_refs)
        assert "ast:missing_isolation" in joined or "ambient:TENANT_REQUIRED_SILENT_DEFAULT" in joined


def test_tx_p0_r2_head_q01_class_field_carriers_have_family() -> None:
    class_ids = class_field_tenant_surface_ids()
    inventory = get_tenant_x_inventory()
    class_records = [r for r in inventory if r.surface_id in class_ids]
    assert len(class_records) == len(class_ids)
    for record in class_records:
        assert isinstance(record.carrier_family, TenantCarrierFamily)


def test_tx_p0_r2_head_q02_class_field_identity_carrier_not_authority() -> None:
    report = carrier_semantics_report()
    assert report.generic_class_field_authority_count == 0
    assert report.class_field_carrier_total > 0


def test_tx_p0_r2_head_q03_unknown_propagation_reduced_vs_class_fields() -> None:
    report = carrier_semantics_report()
    assert report.unknown_propagation_roles < report.class_field_carrier_total
