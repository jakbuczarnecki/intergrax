# © Artur Czarnecki. All rights reserved.

"""EBH-7 W9 evidence dossier + manifest gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7._constants import EBH7_PLANNING_BASELINE_SHA

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]


def test_ebh7_w9_dossier_index_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_EVIDENCE_DOSSIER_INDEX.md"
    assert path.is_file()
    assert EBH7_PLANNING_BASELINE_SHA in path.read_text(encoding="utf-8")


def test_ebh7_w9_proof_staging_dir_exists() -> None:
    proof_root = _REPO / ".tmp/proof/ebh-7-materialization"
    assert proof_root.is_dir()
    manifest = proof_root / "MANIFEST.md"
    assert manifest.is_file()
