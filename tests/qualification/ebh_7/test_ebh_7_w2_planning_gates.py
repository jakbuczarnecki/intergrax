# © Artur Czarnecki. All rights reserved.

"""EBH-7 W2 planning artifact gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]

_W2_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/EBH_7_GAP_REGISTER.md",
    "docs/project/maintainers/qualification/EBH_7_HISTORICAL_EVIDENCE_REUSE_MATRIX.md",
    "docs/project/maintainers/qualification/EBH_7_CROSS_DOMAIN_RISK_MATRIX.md",
    "docs/project/maintainers/qualification/EBH_7_ENTERPRISE_SURFACE_INVENTORY.md",
)


def test_ebh7_w2_mandatory_planning_docs() -> None:
    for rel in _W2_DOCS:
        assert (_REPO / rel).is_file(), rel
