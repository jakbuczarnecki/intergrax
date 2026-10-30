# © Artur Czarnecki. All rights reserved.

"""ROADMAP-REPLAY-X mechanical exit criteria — roadmap §3 ROADMAP-REPLAY-X @ W10 closure."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.roadmap_replay_x._frz_rows import FRZ_REPLAY_ROWS
from tests.qualification.roadmap_replay_x._historical_rows import HISTORICAL_LEDGER_ROWS
from tests.qualification.roadmap_replay_x._replay_constants import (
    DUP_X_ACCEPTANCE_SHA,
    FRZ_OPEN_COUNT,
    FRZ_PASS_COUNT,
    FRZ_ROW_COUNT,
    HISTORICAL_LEDGER_ROW_COUNT,
    MANDATORY_QUAL_DOCS,
    NPSC_42_MODULE_PATHS,
    ROADMAP_REPLAY_X_AUDITED_HEAD,
    ROADMAP_REPLAY_X_INDEPENDENT_ACCEPTANCE_BASELINE_SHA,
)
from tests.qualification.roadmap_replay_x._roadmap_drift import assert_roadmap_drift_zero

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_replay_x_frz_row_count() -> None:
    assert len(FRZ_REPLAY_ROWS) == FRZ_ROW_COUNT


def test_replay_x_frz_pass_open_split() -> None:
    passes = sum(1 for r in FRZ_REPLAY_ROWS if r[1] == "PASS")
    opens = sum(1 for r in FRZ_REPLAY_ROWS if r[1] == "OPEN")
    assert passes == FRZ_PASS_COUNT
    assert opens == FRZ_OPEN_COUNT


def test_replay_x_frz_unique_ids() -> None:
    ids = [r[0] for r in FRZ_REPLAY_ROWS]
    assert len(ids) == len(set(ids))


def test_replay_x_historical_ledger_row_count() -> None:
    assert len(HISTORICAL_LEDGER_ROWS) == HISTORICAL_LEDGER_ROW_COUNT


def test_replay_x_historical_unique_stage_labels() -> None:
    labels = {r[0] for r in HISTORICAL_LEDGER_ROWS}
    assert len(labels) >= 225


def test_replay_x_c0_roadmap_drift_zero() -> None:
    assert_roadmap_drift_zero()


def test_replay_x_mandatory_docs_present() -> None:
    for rel in MANDATORY_QUAL_DOCS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_replay_x_audited_head_in_sha_ladder() -> None:
    ladder = (_REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_SHA_LADDER.md").read_text(
        encoding="utf-8"
    )
    assert ROADMAP_REPLAY_X_AUDITED_HEAD in ladder


def test_replay_x_closed_in_roadmap() -> None:
    roadmap = (_REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md").read_text(
        encoding="utf-8"
    )
    assert ROADMAP_REPLAY_X_INDEPENDENT_ACCEPTANCE_BASELINE_SHA in roadmap
    assert "CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)" in roadmap
    assert "OD-REPLAYX-08" in roadmap


def test_replay_x_npsc_module_paths_exist() -> None:
    for rel in NPSC_42_MODULE_PATHS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_replay_x_prod_delta_empty_since_dup_x() -> None:
    import subprocess

    from tests.qualification.roadmap_replay_x._prod_delta_disposition import (
        REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS,
    )

    for path in ("intergrax", "agents", "applications"):
        names = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                f"{DUP_X_ACCEPTANCE_SHA}..HEAD",
                "--",
                path,
            ],
            cwd=_REPO_ROOT,
            text=True,
        ).splitlines()
        unscoped = [
            n.strip().replace("\\", "/")
            for n in names
            if n.strip() and n.strip().replace("\\", "/") not in REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS
        ]
        assert unscoped == [], f"{path}: {unscoped}"


def test_replay_x_operator_closure_recorded() -> None:
    reg = (
        _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_OPERATOR_DECISION_REGISTER.md"
    ).read_text(encoding="utf-8")
    assert "OD-REPLAYX-08" in reg and "APPROVED" in reg
    dossier = (
        _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PARENT_IMPLEMENTATION_DOSSIER.md"
    ).read_text(encoding="utf-8")
    assert "CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)" in dossier
    pass_rec = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md"
    ).read_text(encoding="utf-8")
    assert "ISSUED — PASS (SCOPED)" in pass_rec


def test_replay_x_protocol_03_b6_closed_scoped() -> None:
    pkg = (
        _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PACKAGE.md"
    ).read_text(encoding="utf-8")
    assert "PASS (SCOPED)" in pkg
    assert "OD-REPLAYX-08" in pkg


def test_replay_x_closure_counters_computed() -> None:
    from tests.qualification.roadmap_replay_x._closure_counters import all_counters
    from tests.qualification.roadmap_replay_x._replay_blocker_register import (
        replay_blocker_register_violations,
    )

    counters = all_counters()
    assert len(counters) == 6
    by_name = {c.name: c for c in counters}
    assert replay_blocker_register_violations() == ()
    assert by_name["roadmap_drift"].value == 0
    assert by_name["historical_evidence_invalidated_by_later_changes"].value == 0
    assert by_name["reopened_historical_stages"].value == 0
    assert by_name["unresolved_replay_blockers"].value == 0
    assert by_name["unclassified_replay_findings"].value == 0
    assert by_name["vendor_neutrality_replay_failures"].value == 0


def test_replay_x_b6_r2_reconciliation_zero() -> None:
    from tests.qualification.roadmap_replay_x._replay_bidirectional_reconciliation import (
        replay_r2_reconciliation_violations,
    )

    assert replay_r2_reconciliation_violations() == ()


def test_replay_x_vbn_seven_artifacts_present() -> None:
    from tests.qualification.roadmap_replay_x._vbn_replay_counter import REPLAY_VBN_ARTIFACT_REL_PATHS

    for rel in REPLAY_VBN_ARTIFACT_REL_PATHS:
        assert (_REPO_ROOT / rel).is_file(), rel
