# © Artur Czarnecki. All rights reserved.

"""EBH-7 W5 VBN artifacts + RPL-H disposition gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7._rpl_h_disposition import assert_ebh7_rpl_h_disposition_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]

_VBN_ARTIFACT_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md",
    "docs/project/maintainers/qualification/EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md",
    "docs/project/maintainers/qualification/EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md",
    "docs/project/maintainers/qualification/EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md",
    "docs/project/maintainers/qualification/EBH_5_VENDOR_SELECTION_REACHABILITY.md",
    "docs/project/maintainers/qualification/EBH_5_ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF.md",
    "docs/project/maintainers/qualification/EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md",
)


def test_ebh7_w5_vbn_seven_artifacts_present() -> None:
    for rel in _VBN_ARTIFACT_DOCS:
        assert (_REPO / rel).is_file(), rel


def test_ebh7_w5_rpl_h_matrix_56_at_head() -> None:
    assert_ebh7_rpl_h_disposition_integrity()


def test_ebh7_w5_rpl_disposition_record_present() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_RPL_H_BATCH_DISPOSITION_RECORD.md"
    assert path.is_file()
    assert path.read_text(encoding="utf-8").count("| integration.") >= 20
