# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 current-HEAD closed-world inventory scan."""

from __future__ import annotations

import pytest

from tests.qualification.prod_q._p0_certification import build_prod_q_p0_certification_report
from tests.qualification.prod_q._p0_classification import (
    classification_by_candidate_id,
    classified_inventory,
)
from tests.qualification.prod_q._p0_discovery import (
    anchor_resolution_failures,
    discovery_scan_failures,
    raw_discovery_candidates,
    raw_discovery_duplicate_ids,
)
from tests.qualification.prod_q._p0_integrity import assert_inventory_identity_sound
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation, required_frz_ids
from tests.qualification.prod_q._p0_inventory import (
    build_closed_world_discovery_report,
    build_family_coverage,
    build_ownership_matrix,
)
from tests.qualification.prod_q._p0_ownership import (
    build_ownership_reconciliation,
    ownership_soundness_violations,
)
from tests.qualification.prod_q._p0_types import (
    ALL_PRODUCTION_READINESS_FAMILIES,
    ProdQFindingDisposition,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_prod_q_p0_head_q01_discovery_parse_fail_closed() -> None:
    assert discovery_scan_failures() == ()
    assert anchor_resolution_failures() == ()
    assert raw_discovery_duplicate_ids() == ()


def test_prod_q_p0_head_q02_inventory_covers_all_candidates() -> None:
    report = build_closed_world_discovery_report()
    raw = raw_discovery_candidates()
    by_id = classification_by_candidate_id()
    assert_inventory_identity_sound(raw, by_id)
    assert report.unclassified_candidate_ids == frozenset()
    assert len(report.raw_candidates) == len(classified_inventory())


def test_prod_q_p0_head_q03_all_pq_families_classified() -> None:
    coverage = {row.family for row in build_family_coverage()}
    assert coverage == ALL_PRODUCTION_READINESS_FAMILIES
    for row in build_family_coverage():
        assert row.candidate_count >= 0


def test_prod_q_p0_head_q04_frz_exactly_once() -> None:
    ids = [r.criterion_id for r in build_frz_reconciliation()]
    assert tuple(ids) == required_frz_ids()


def test_prod_q_p0_head_q05_ownership_duplicates_absent() -> None:
    matrix = build_ownership_matrix()
    recon = build_ownership_reconciliation()
    assert matrix.duplicate_semantic_owner_keys == ()
    assert recon.duplicate_semantic_concern_keys == ()
    assert ownership_soundness_violations() == ()
    assert (
        recon.candidate_count
        == recon.semantic_mechanism_count
        + recon.auxiliary_discovery_count
        + recon.unresolved_classification_count
    )
    assert (
        recon.ownership_resolved
        + recon.ownership_unresolved
        + recon.ownership_ambiguous
        + recon.ownership_not_applicable_with_evidence
        + recon.unresolved_classification_count
        == recon.candidate_count
    )


def test_prod_q_p0_head_q06_certification_report_ready_for_audit_shape() -> None:
    report = build_prod_q_p0_certification_report()
    assert report.matrix.unclassified_candidates == 0
    assert report.matrix.frz_criteria_classified == 33
    assert report.matrix.families_covered == 23
    assert report.tenant_audit.verdict == "DERIVED_PASS_REUSE_TENANT_X"
    assert report.r1_sqlite.reproduction_executed is False
    assert report.r1_sqlite.reproduction_classification.value == "NOT_EXECUTED"
    assert report.r1_sqlite.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    assert report.ownership_reconciliation.semantic_mechanism_count > 0
    assert len(report.recommended_children) >= 3


def test_prod_q_p0_head_q07_provider_qualification_external_evidence() -> None:
    report = build_prod_q_p0_certification_report()
    ids = {p.provider_id for p in report.provider_dispositions}
    assert ids == {"E2B", "Modal", "Daytona"}
    assert all(not p.production_approved for p in report.provider_dispositions)
