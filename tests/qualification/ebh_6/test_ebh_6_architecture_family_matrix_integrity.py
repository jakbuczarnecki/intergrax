# © Artur Czarnecki. All rights reserved.

"""Architecture-family recertification matrix integrity (catalog SSOT)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._family_catalog import (
    EBH6_APPLICABLE_FAMILY_COUNT,
    EBH6_FAMILY_CATALOG,
    FamilyIndependentDisposition,
    family_by_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_MD = _REPO_ROOT / "docs/project/maintainers/qualification/EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md"


def test_ebh6_family_catalog_covers_applicable_themes() -> None:
    assert EBH6_APPLICABLE_FAMILY_COUNT >= 22
    assert "VBN" in family_by_id()
    assert "BND" in family_by_id()
    assert "CFG" in family_by_id()


def test_ebh6_family_row_count_closed_world() -> None:
    assert len(EBH6_FAMILY_CATALOG) == EBH6_APPLICABLE_FAMILY_COUNT


def test_ebh6_family_independent_disposition_not_closed() -> None:
    pending = [
        row for row in EBH6_FAMILY_CATALOG if row.independent_disposition == FamilyIndependentDisposition.PENDING
    ]
    assert pending, "L9 must review pending families"


def test_ebh6_family_gate_test_paths_exist() -> None:
    for row in EBH6_FAMILY_CATALOG:
        for node_id in row.gate_test_ids:
            if node_id.startswith("tests/"):
                path_part = node_id.split("::", 1)[0]
                assert (_REPO_ROOT / path_part).is_file(), node_id


def test_ebh6_matrix_md_lists_catalog_families() -> None:
    text = _MATRIX_MD.read_text(encoding="utf-8")
    for row in EBH6_FAMILY_CATALOG:
        assert row.family_id in text, row.family_id
    assert "_family_catalog.py" in text
