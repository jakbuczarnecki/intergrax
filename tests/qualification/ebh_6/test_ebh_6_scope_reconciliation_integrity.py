# © Artur Czarnecki. All rights reserved.

"""SCOPE-03 dossier content controls (DOC family gate)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._evidence_manifest import evidence_git_head_sha
from tests.qualification.ebh_6._l3_aj_group_reconciliation import assert_aj_group_reconciliation_complete

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
    assert "Evidence tier audit @ HEAD" in text
    assert "_t2_evidence_bundles.py" in text or "T2-AUD" in text
    assert "_tier_audit.py" in text
    assert "_l3_aj_group_reconciliation.py" in text
    assert "_family_l9_evidence.py" in text
    assert "CONFIG-X" in text
    assert "OD-EBH6-01" in text
    assert "INDEPENDENT_AUDITOR_DISPOSITION" not in text or "must not set" in text.lower() or "23 × PENDING" in text
    assert "EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md" in text
    semantic = _REPO / "docs/project/maintainers/qualification/EBH_6_FINAL_SEMANTIC_AUDIT_DOSSIER.md"
    assert semantic.is_file()
    prod = _REPO / "docs/project/maintainers/qualification/EBH_6_FINAL_PRODUCTION_CODE_AUDIT_DOSSIER.md"
    assert prod.is_file()
    prod_text = prod.read_text(encoding="utf-8")
    assert "PARTIAL / EXACT BLOCKERS" in prod_text or "EBH-6 — CLOSED" in prod_text
    assert _ROADMAP.is_file()
    roadmap_head = _ROADMAP.read_text(encoding="utf-8")[:120000]
    assert "EBH" in roadmap_head and "3.0.8" in roadmap_head


def test_ebh6_scope_aj_groups_complete() -> None:
    from tests.qualification.ebh_6._t2_evidence_bundles import mandatory_insufficient_pairs

    assert_aj_group_reconciliation_complete()
    from tests.qualification.ebh_6._l3_aj_group_reconciliation import (
        AjControlOutcome,
        aj_group_by_layer_class,
        count_evidence_required_groups,
    )

    groups = aj_group_by_layer_class()
    ev_req = sum(1 for g in groups.values() if g.outcome == AjControlOutcome.EVIDENCE_REQUIRED)
    expected = len(mandatory_insufficient_pairs())
    assert len(groups) == 133
    assert ev_req == count_evidence_required_groups()
    assert ev_req == expected


def test_ebh6_tier_audit_optional_session_artifact() -> None:
    from tests.qualification.ebh_6._tier_audit import write_optional_session_tier_audit

    write_optional_session_tier_audit(".tmp/session/ebh-6-cert-handoff", evidence_git_head_sha())


def test_ebh6_l9a_matrix_parity_with_layer_evidence() -> None:
    """T-NORM-01 — L9-A table @ EBH_6_FINAL_L9_EVIDENCE_COMPLETION_PLAN.md vs _layer_evidence."""
    from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
    from tests.qualification.ebh_6._l3_aj_group_reconciliation import (
        AjControlOutcome,
        aj_group_by_layer_class,
        count_evidence_required_groups,
    )
    from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability

    l9a: dict[str, str] = {
        "contracts": "YNNNNYNYNY",
        "semantic_core": "YYYNNYNYYY",
        "application_services": "YYYYNYNYYY",
        "composition_roots": "YYYYNYNYYY",
        "governance": "NNYYYYYYYN",
        "execution": "YYYNNYYYYY",
        "integrations": "YYYYYYYYYY",
        "state_persistence": "YYYYYYNYYY",
        "reliability_recovery": "YYYNYYYYYY",
        "observability_diagnostics": "NNYYNYYYYY",
        "events_traceability": "YYYYYYYYYY",
        "agent_distribution": "YYYYNYNYYY",
        "rag_memory": "YYYYYYYYYY",
        "runtime_hosting": "YYYYNYYYYY",
        "tools_skills_capabilities": "YYYYNYYYYY",
        "background_workers_infra_adapters": "YYYYYYYYYY",
    }
    classes = "ABCDEFGHIJ"
    applicable = 0
    for layer in EBH6_AUDIT_LAYERS:
        exp = l9a[layer]
        assert len(exp) == len(classes)
        got = layer_aj_applicability(layer)
        for idx, letter in enumerate(classes):
            want = exp[idx] == "Y"
            assert got[idx].applicable is want, (layer, letter, want, got[idx].applicable)
            if got[idx].applicable:
                applicable += 1
    from tests.qualification.ebh_6._t2_evidence_bundles import mandatory_insufficient_pairs

    groups = aj_group_by_layer_class()
    ev_req = sum(1 for g in groups.values() if g.outcome == AjControlOutcome.EVIDENCE_REQUIRED)
    expected = len(mandatory_insufficient_pairs())
    assert applicable == 133
    assert len(groups) == 133
    assert ev_req == count_evidence_required_groups()
    assert ev_req == expected
