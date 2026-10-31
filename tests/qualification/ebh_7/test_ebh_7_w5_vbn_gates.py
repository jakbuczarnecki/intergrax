# © Artur Czarnecki. All rights reserved.

"""EBH-7 W5 VBN — structural vs behavioral gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7._rpl_h_disposition import (
    assert_ebh7_rpl_h_behavioral_qualification,
    assert_ebh7_rpl_h_disposition_integrity,
    assert_ebh7_rpl_h_semantic_qualification_for_b6,
)
from tests.qualification.ebh_7._rpl_h_semantic_review import assert_ebh7_rpl_h_semantic_review_complete

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


def test_ebh7_w5_vbn_artifacts_materialized() -> None:
    for rel in _VBN_ARTIFACT_DOCS:
        assert (_REPO / rel).is_file(), rel


def test_ebh7_w5_rpl_h_structural_inventory_56() -> None:
    assert_ebh7_rpl_h_disposition_integrity()


def test_ebh7_w5_rpl_disposition_record_materialized() -> None:
    path = _REPO / "docs/project/maintainers/qualification/EBH_7_RPL_H_BATCH_DISPOSITION_RECORD.md"
    assert path.is_file()


def test_ebh7_w5_rpl_h_behavioral_all_qualified() -> None:
    assert_ebh7_rpl_h_behavioral_qualification()


def test_ebh7_w5_rpl_h_semantic_review_complete() -> None:
    assert_ebh7_rpl_h_semantic_review_complete()


def test_ebh7_b6_readiness_rpl_h_semantic_qualified() -> None:
    assert_ebh7_rpl_h_semantic_qualification_for_b6()
