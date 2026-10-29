# © Artur Czarnecki. All rights reserved.

"""ROADMAP-REPLAY-X mechanical exit criteria — roadmap §3 ROADMAP-REPLAY-X (handoff, not CLOSED)."""

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


def test_replay_x_dup_x_closed_in_roadmap() -> None:
    roadmap = (_REPO_ROOT / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md").read_text(
        encoding="utf-8"
    )
    assert DUP_X_ACCEPTANCE_SHA in roadmap
    assert "ROADMAP-REPLAY-X" in roadmap and "READY FOR INDEPENDENT AUDIT" in roadmap


def test_replay_x_npsc_module_paths_exist() -> None:
    for rel in NPSC_42_MODULE_PATHS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_replay_x_prod_delta_empty_since_dup_x() -> None:
    import subprocess

    for path in ("intergrax", "agents", "applications"):
        out = subprocess.check_output(
            [
                "git",
                "diff",
                "--stat",
                f"{DUP_X_ACCEPTANCE_SHA}..HEAD",
                "--",
                path,
            ],
            cwd=_REPO_ROOT,
            text=True,
        ).strip()
        assert out == "", path


def test_replay_x_not_self_closed() -> None:
    dossier = (
        _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PARENT_IMPLEMENTATION_DOSSIER.md"
    ).read_text(encoding="utf-8")
    assert "READY FOR INDEPENDENT AUDIT" in dossier
    assert "NOT CLOSED" in dossier or "B6 OPEN" in dossier


def test_replay_x_protocol_03_b6_open() -> None:
    pkg = (
        _REPO_ROOT / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PACKAGE.md"
    ).read_text(encoding="utf-8")
    assert "B6" in pkg
    assert "OPEN" in pkg or "NOT ISSUED" in pkg or "READY" in pkg


def test_replay_x_closure_counters_computed() -> None:
    from tests.qualification.roadmap_replay_x._closure_counters import all_counters

    counters = all_counters()
    assert len(counters) == 6
    by_name = {c.name: c for c in counters}
    assert by_name["roadmap_drift"].value == 0
    assert by_name["historical_evidence_invalidated_by_later_changes"].value == 0
    assert by_name["reopened_historical_stages"].value == 0


def test_replay_x_vbn_seven_artifacts_present() -> None:
    vbn = (
        "ROADMAP_REPLAY_X_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md",
        "ROADMAP_REPLAY_X_CORE_HARDCODING_FINDINGS_REGISTER.md",
        "ROADMAP_REPLAY_X_CROSS_LAYER_REPLACEABILITY_MATRIX.md",
        "ROADMAP_REPLAY_X_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md",
        "ROADMAP_REPLAY_X_VENDOR_SELECTION_REACHABILITY_EVIDENCE.md",
        "ROADMAP_REPLAY_X_ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF.md",
        "ROADMAP_REPLAY_X_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md",
    )
    for name in vbn:
        assert (_REPO_ROOT / "docs/project/maintainers/qualification" / name).is_file(), name
