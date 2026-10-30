# © Artur Czarnecki. All rights reserved.

"""B6 R2 bidirectional reconciliation + scoped regression evidence gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.roadmap_replay_x._formal_reopen_register import FormalReopenRecord
from tests.qualification.roadmap_replay_x._replay_bidirectional_reconciliation import (
    ecap_counter_bidirectional_violations,
    formal_reopen_bidirectional_violations,
    replay_blocker_bidirectional_violations,
    replay_r2_reconciliation_violations,
)
from tests.qualification.roadmap_replay_x._replay_blocker_register import ReplayBlockerRecord
from tests.qualification.roadmap_replay_x._replay_source_adapters import discover_roadmap_category_a_reopen_claims
from tests.qualification.roadmap_replay_x._scoped_regression_evidence import (
    scoped_regression_evidence_violations,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "roadmap_replay_x"
_REPO = Path(__file__).resolve().parents[3]


def test_r2_reconciliation_clean_at_head() -> None:
    assert replay_r2_reconciliation_violations() == ()


def test_r2_ecap_counter_matches_mechanical_counters() -> None:
    assert ecap_counter_bidirectional_violations() == ()


def test_r2_scoped_regression_evidence_valid_at_head() -> None:
    assert scoped_regression_evidence_violations(_REPO) == ()


def test_r2_negative_ecap_counter_mismatch_uses_fixture_shape() -> None:
    ecap = (_FIXTURES / "ecap_six_counter_excerpt.md").read_text(encoding="utf-8")
    violations = ecap_counter_bidirectional_violations(ecap_text=ecap)
    assert any(v.startswith("ecap_counter_vs_mechanical:unresolved_replay_blockers:") for v in violations)


def test_r2_negative_roadmap_reopen_without_formal_record_fixture() -> None:
    roadmap = (_FIXTURES / "roadmap_section3_reopen_excerpt.md").read_text(encoding="utf-8")
    violations = formal_reopen_bidirectional_violations((), roadmap_text=roadmap)
    assert any(v.startswith("roadmap_reopen_without_formal_record:CONFIG-X") for v in violations)
    claims = discover_roadmap_category_a_reopen_claims(roadmap)
    assert any(c.parent_stage_id == "CONFIG-X" for c in claims)


def test_r2_negative_closure_matrix_blocker_omitted_from_register_fixture() -> None:
    matrix = (_FIXTURES / "closure_matrix_blocker_excerpt.md").read_text(encoding="utf-8")
    violations = replay_blocker_bidirectional_violations((), closure_text=matrix)
    assert any("discovered_blocker_without_register:" in v for v in violations)


def test_r2_negative_closed_blocker_without_task_fixture() -> None:
    register = (
        ReplayBlockerRecord(
            blocker_id="RX-BLK-FIX-NEG",
            source_ref="ECAP-C5",
            status="CLOSED",
            authority="audit",
            evidence_ref="docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md",
            closed_task=None,
        ),
    )
    violations = replay_blocker_bidirectional_violations(register)
    assert any(v.startswith("closed_without_task:") for v in violations)


def test_r2_negative_formal_reopen_open_without_roadmap_fixture() -> None:
    register = (
        FormalReopenRecord(
            reopen_id="RX-REOPEN-FIX-NEG",
            parent_stage_id="CONFIG-X",
            status="OPEN",
            authority="replay",
            evidence_ref="ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md",
        ),
    )
    roadmap = (_FIXTURES / "roadmap_section3_closed_excerpt.md").read_text(encoding="utf-8")
    violations = formal_reopen_bidirectional_violations(register, roadmap_text=roadmap)
    assert any(v.startswith("formal_reopen_without_roadmap_signal:CONFIG-X") for v in violations)
