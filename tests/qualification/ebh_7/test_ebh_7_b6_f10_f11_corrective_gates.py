# © Artur Czarnecki. All rights reserved.

"""B6-F10/F11 corrective gates — ECAP normative matrix + dossier reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7._constants import (
    EBH7_B5_GHA_RUN_ID,
    EBH7_B5_REPLAY_HEAD_SHA,
    EBH7_TEST_EXECUTION_SHA,
)
from tests.qualification.ebh_7._ebh7_f10_ecap_normative_matrix import (
    assert_ebh7_f10_ecap_normative_ready,
    sync_f10_matrix_markdown,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json"


def test_b6_f10_ecap_normative_matrix_ready() -> None:
    assert_ebh7_f10_ecap_normative_ready()


def test_b6_f10_matrix_markdown_synced() -> None:
    sync_f10_matrix_markdown()
    md = (_REPO / "docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md").read_text(
        encoding="utf-8"
    )
    assert "ECAP_GATE" in md and "C1" in md and "C5" in md


def test_b6_f11_dossier_replay_profile_current() -> None:
    dossier = (_REPO / "docs/project/maintainers/qualification/EBH_7_INDEPENDENT_B6_AUDIT_DOSSIER.md").read_text(
        encoding="utf-8"
    )
    manifest = _MANIFEST.read_text(encoding="utf-8")
    assert EBH7_TEST_EXECUTION_SHA in dossier
    assert EBH7_B5_GHA_RUN_ID in dossier
    assert EBH7_B5_REPLAY_HEAD_SHA in dossier
    assert "test_ebh_5_dual_slug_rpl_e_operational_gates.py" in dossier
    assert "41212701986" not in dossier or "superseded" in dossier.lower()
    assert "B6 PASS" not in dossier or "NOT ISSUED" in dossier
    assert "d1bdcfa46e973cf9f3bb9ee14611036980951c91" not in dossier
    assert EBH7_TEST_EXECUTION_SHA in manifest


def test_b6_f11_traceability_effective_sha_roles() -> None:
    text = (_REPO / "docs/project/maintainers/qualification/EBH_7_FINAL_CLOSURE_TRACEABILITY.md").read_text(
        encoding="utf-8"
    )
    assert EBH7_TEST_EXECUTION_SHA in text
    assert EBH7_B5_GHA_RUN_ID in text
    assert "READY FOR INDEPENDENT B6 RE-AUDIT" in text
    assert "8afff322cb2525f0106c2bbccf2d44f682d3e0df" in text  # historical B6 corrective preserved


def test_b6_f11_handoff_links_resolve() -> None:
    handoff = (_REPO / "docs/project/maintainers/qualification/EBH_7_B6_EXTERNAL_AUDIT_HANDOFF.md").read_text(
        encoding="utf-8"
    )
    assert "EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md" in handoff
    assert (_REPO / "docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md").is_file()
