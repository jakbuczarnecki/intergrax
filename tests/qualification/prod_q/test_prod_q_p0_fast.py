# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 fast gates — types, FRZ registry, historical ancestry (no full scan budget)."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.prod_q._p0_certification import (
    HISTORICAL_EVIDENCE,
    PROD_Q_P0_CORRECTIVE_STAGE_ID,
    PROD_Q_P0_MECHANICAL_EVIDENCE_LOCK_SHA,
    _git_is_ancestor,
    build_r1_sqlite_disposition,
    current_git_head,
    historical_evidence_sha_mismatches,
    probe_default_relational_db_header,
)
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation, required_frz_ids
from tests.qualification.prod_q._p0_types import (
    ALL_PRODUCTION_READINESS_FAMILIES,
    ProdQFindingDisposition,
    ProductionReadinessEvidenceState,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_prod_q_p0_fast_q01_head_is_git_object() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert len(head) == 40


def test_prod_q_p0_fast_q02_frz_registry_exact_33() -> None:
    records = build_frz_reconciliation()
    ids = [r.criterion_id for r in records]
    assert ids == list(required_frz_ids())
    assert len(ids) == 33
    assert len(frozenset(ids)) == 33


def test_prod_q_p0_fast_q03_frz_evidence_states_finite() -> None:
    allowed = frozenset(ProductionReadinessEvidenceState)
    for record in build_frz_reconciliation():
        assert record.evidence_state in allowed
        assert record.finding_disposition in frozenset(ProdQFindingDisposition)


def test_prod_q_p0_fast_q04_historical_evidence_ancestry() -> None:
    ctrl = next(item for item in HISTORICAL_EVIDENCE if item.stage_id == "CTRL-X")
    assert ctrl.accepted_sha == "318a4d835534ef5c48575f4649551c957dacd8ea"
    tenant = next(item for item in HISTORICAL_EVIDENCE if item.stage_id == "TENANT-X")
    assert tenant.accepted_sha == "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558"
    p0_lock = next(
        item for item in HISTORICAL_EVIDENCE if item.stage_id == PROD_Q_P0_CORRECTIVE_STAGE_ID
    )
    assert p0_lock.accepted_sha == PROD_Q_P0_MECHANICAL_EVIDENCE_LOCK_SHA
    head = current_git_head()
    assert _git_is_ancestor(PROD_Q_P0_MECHANICAL_EVIDENCE_LOCK_SHA, head)
    assert historical_evidence_sha_mismatches() == ()


def test_prod_q_p0_fast_q05_family_registry_complete() -> None:
    assert len(ALL_PRODUCTION_READINESS_FAMILIES) == 23


def test_prod_q_p0_fast_q06_r1_sqlite_disposition_blocker() -> None:
    record = build_r1_sqlite_disposition()
    assert record.finding_id == "R1-SQLITE-ENV-01"
    assert record.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    _ = probe_default_relational_db_header()


def test_prod_q_p0_fast_q07_prd04_linked_to_sqlite() -> None:
    prd04 = next(r for r in build_frz_reconciliation() if r.criterion_id == "FRZ-PRD-04")
    assert prd04.evidence_state == ProductionReadinessEvidenceState.BLOCKER
    assert "R1-SQLITE-ENV-01" in prd04.evidence_refs[0]
