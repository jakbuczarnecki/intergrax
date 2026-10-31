# © Artur Czarnecki. All rights reserved.

"""Anti-tautology matrix for ROADMAP-REPLAY-X six B6 counters (B6-06)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.roadmap_replay_x._closure_counters import (
    compute_historical_evidence_invalidated,
    compute_reopened_historical_stages,
    compute_roadmap_drift,
    compute_unresolved_replay_blockers,
    compute_vendor_neutrality_replay_failures,
)
from tests.qualification.roadmap_replay_x._formal_reopen_register import (
    FormalReopenRecord,
)
from tests.qualification.roadmap_replay_x._prod_delta_disposition import ProdDeltaDisposition
from tests.qualification.roadmap_replay_x._replay_blocker_register import (
    ReplayBlockerRecord,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_b6_01_unresolved_blocker_nonzero_on_open_register(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.qualification.roadmap_replay_x._replay_blocker_register as reg

    monkeypatch.setattr(
        reg,
        "REPLAY_BLOCKER_REGISTER",
        (
            ReplayBlockerRecord(
                blocker_id="RX-BLK-NEG-01",
                source_ref="ECAP-C5",
                status="OPEN",
                authority="independent-audit",
                evidence_ref="docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md",
            ),
        ),
    )
    assert compute_unresolved_replay_blockers().value >= 1


def test_b6_02_reopen_nonzero_without_reopen_required_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.qualification.roadmap_replay_x._formal_reopen_register as reopen

    repo_root = Path(__file__).resolve().parents[3]
    roadmap = repo_root / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
    roadmap_text = roadmap.read_text(encoding="utf-8")
    assert "REOPEN REQUIRED" not in roadmap_text
    monkeypatch.setattr(
        reopen,
        "FORMAL_REOPEN_REGISTER",
        (
            FormalReopenRecord(
                reopen_id="RX-REOPEN-NEG-01",
                parent_stage_id="CONFIG-X",
                status="OPEN",
                authority="replay-parent",
                evidence_ref="ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md",
            ),
        ),
    )
    assert compute_reopened_historical_stages().value >= 1


def test_b6_03_invalid_scoped_disposition_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.qualification.roadmap_replay_x._prod_delta_disposition as disp

    bad = (
        ProdDeltaDisposition(
            path="intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
            disposition="",
            regression_evidence="tests/qualification/roadmap_replay_x/__init__.py",
            qualification_head_sha="0000000000000000000000000000000000000000",
        ),
    )
    monkeypatch.setattr(disp, "REPLAY_X_L2_SCOPED_PROD_DELTA", bad)
    monkeypatch.setattr(disp, "REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS", frozenset(row.path for row in bad))
    assert compute_historical_evidence_invalidated().value >= 1


def test_b6_04_unknown_inventory_finding_nonzero() -> None:
    from tests.qualification.roadmap_replay_x._replay_finding_ledger import count_unclassified_replay_findings

    assert count_unclassified_replay_findings(extra_inventory_ids=frozenset({"RX-FINDING-UNKNOWN-ADVERSARIAL"})) >= 1


def test_b6_05_prohibited_unresolved_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.qualification.roadmap_replay_x._vbn_replay_counter as vbn

    monkeypatch.setattr(
        vbn,
        "count_prohibited_unresolved",
        lambda: 1,
    )
    assert compute_vendor_neutrality_replay_failures().value >= 1


def test_b6_06_roadmap_drift_nonzero_on_forbidden_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.qualification.roadmap_replay_x._closure_counters.roadmap_drift_violations",
        lambda: ["forbidden:RX-TAUTO-DRIFT-INJECT"],
    )
    assert compute_roadmap_drift().value >= 1


def test_b6_03_scoped_disposition_valid_at_head() -> None:
    from tests.qualification.roadmap_replay_x._prod_delta_disposition import (
        scoped_prod_delta_disposition_violations,
    )

    repo = Path(__file__).resolve().parents[3]
    assert scoped_prod_delta_disposition_violations(repo) == ()


def test_b6_04_finding_ledger_covers_inventories() -> None:
    from tests.qualification.roadmap_replay_x._replay_finding_ledger import (
        replay_finding_inventories,
        replay_finding_ledger,
    )

    ledger = replay_finding_ledger()
    for _source, ids in replay_finding_inventories().items():
        for finding_id in ids:
            assert finding_id in ledger
