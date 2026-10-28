# © Artur Czarnecki. All rights reserved.

"""SCOPE-03 dossier content controls (DOC family gate)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._evidence_manifest import evidence_git_head_sha
from tests.qualification.ebh_6._l3_aj_group_reconciliation import assert_aj_group_reconciliation_complete
from tests.qualification.ebh_6._l3_reachability_evidence import count_l3_aj_outcomes

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_DOSSIER = _REPO / "docs/project/maintainers/qualification/EBH_6_SCOPE_RECONCILIATION_DOSSIER.md"
_ROADMAP = _REPO / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"


def test_ebh6_scope_dossier_content_controls() -> None:
    assert _DOSSIER.is_file()
    text = _DOSSIER.read_text(encoding="utf-8")
    sha = evidence_git_head_sha()
    assert "IMPLEMENTATION_SHA" in text
    assert sha in text or "*(set @ qual commit)*" in text or "CERTIFICATION_HANDOFF_SHA" in text
    assert "_l3_aj_group_reconciliation.py" in text
    assert "_family_l9_evidence.py" in text
    assert "CONFIG-X" in text
    assert "OD-EBH6-01" in text
    assert "INDEPENDENT_AUDITOR_DISPOSITION" not in text or "must not set" in text.lower() or "23 × PENDING" in text
    assert _ROADMAP.is_file()
    roadmap_head = _ROADMAP.read_text(encoding="utf-8")[:120000]
    assert "EBH" in roadmap_head and "3.0.8" in roadmap_head


def test_ebh6_scope_aj_groups_complete() -> None:
    assert_aj_group_reconciliation_complete()
    counts = count_l3_aj_outcomes()
    assert counts.get("EVIDENCE_REQUIRED", 0) == 0
