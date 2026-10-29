# © Artur Czarnecki. All rights reserved.

"""EBH-6 mechanical exit criteria (roadmap §3.0.8 VBN slice + dossier presence)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._closed_world_discovery import (
    assert_ebh6_closed_world_surfaces_non_empty,
    assert_ebh6_sqlite_annex_accountability,
    discover_sanctioned_production_composition_roots,
)
from tests.qualification.ebh_6._config_x_reconciliation import assert_ebh6_config_x_fully_reconciled
from tests.qualification.ebh_6._evidence_manifest import (
    EBH6_CERTIFICATION_HANDOFF_SHA,
    EBH6_PUBLICATION_SHA,
    assert_ebh5_audit_baseline_resolvable,
    evidence_git_head_sha,
)
from tests.qualification.ebh_6._findings_register import (
    count_prohibited_unresolved,
    count_unclassified,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MANDATORY_QUAL_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/EBH_6_OPERATOR_DECISION_REGISTER.md",
    "docs/project/maintainers/qualification/EBH_6_ENTERPRISE_SURFACE_INVENTORY.md",
    "docs/project/maintainers/qualification/EBH_6_EVIDENCE_FRESHNESS_LEDGER.md",
    "docs/project/maintainers/qualification/EBH_6_ARCHITECTURE_FAMILY_RECERTIFICATION_MATRIX.md",
    "docs/project/maintainers/qualification/EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md",
    "docs/project/maintainers/qualification/EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md",
    "docs/project/maintainers/qualification/EBH_6_PARENT_CERTIFICATION_DOSSIER.md",
    "docs/project/maintainers/qualification/EBH_6_L9_INDEPENDENT_AUDIT_HANDOFF.md",
    "docs/project/maintainers/qualification/EBH_6_STAGE_CERTIFICATION_READINESS.md",
    "docs/project/maintainers/qualification/EBH_6_INDEPENDENT_PRE_CERTIFICATION_FINDINGS.md",
    "docs/project/maintainers/qualification/EBH_6_T2_EVIDENCE_INDEPENDENT_REVIEW.md",
    "docs/project/maintainers/qualification/EBH_6_SCOPE_RECONCILIATION_DOSSIER.md",
    "docs/project/maintainers/qualification/EBH_6_PROTOCOL_03_REJECTION_REMEDIATION.md",
)


def test_ebh6_exit_findings_register_complete() -> None:
    assert count_unclassified() == 0
    assert count_prohibited_unresolved() == 0


def test_ebh6_exit_config_x_reconciled() -> None:
    assert_ebh6_config_x_fully_reconciled()


def test_ebh6_exit_sqlite_annex_accountability() -> None:
    assert_ebh6_sqlite_annex_accountability()


def test_ebh6_exit_closed_world_surfaces() -> None:
    assert_ebh6_closed_world_surfaces_non_empty()


def test_ebh6_exit_sanctioned_composition_roots_non_empty() -> None:
    assert discover_sanctioned_production_composition_roots()


def test_ebh6_exit_mandatory_docs_present() -> None:
    for rel in _MANDATORY_QUAL_DOCS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_ebh6_exit_p03_handoff_export_complete() -> None:
    from tests.qualification.ebh_6._p03_handoff_export import p03_handoff_bundle
    from tests.qualification.ebh_6._t2_evidence_bundles import mandatory_insufficient_pairs

    bundle = p03_handoff_bundle()
    insufficient = mandatory_insufficient_pairs()
    assert bundle["aj_group_count"] == 133
    assert bundle["family_count"] == 23
    assert bundle["l9_blocker_count"] == 8
    assert bundle["aj_evidence_required_count"] == len(insufficient)
    assert "PARTIAL" in bundle["certification_banner"]
    assert bundle["tier_census"]["T1"] == len(insufficient)
    assert bundle["layer_candidate_resolved_count"] == 16 - len(
        {layer for layer, _ in insufficient},
    )
    assert bundle["layer_l9_evidence_submitted_count"] == len({layer for layer, _ in insufficient})
    assert bundle["independent_disposition_pending"] == 23
    assert bundle["certification_handoff_sha"] == evidence_git_head_sha()


def test_ebh6_exit_evidence_sha_recorded() -> None:
    assert_ebh5_audit_baseline_resolvable()
    sha = evidence_git_head_sha()
    assert len(sha) == 40
    dossier = (_REPO_ROOT / _MANDATORY_QUAL_DOCS[6]).read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in dossier or "IMPLEMENTATION_SHA" in dossier
    handoff_ok = EBH6_CERTIFICATION_HANDOFF_SHA in dossier and _git_is_ancestor(
        EBH6_CERTIFICATION_HANDOFF_SHA,
        sha,
    )
    assert sha in dossier or EBH6_PUBLICATION_SHA in dossier or handoff_ok


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    import subprocess

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
    )
    return result.returncode == 0
