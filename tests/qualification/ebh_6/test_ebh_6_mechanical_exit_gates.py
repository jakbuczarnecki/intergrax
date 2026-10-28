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


def test_ebh6_exit_evidence_sha_recorded() -> None:
    assert_ebh5_audit_baseline_resolvable()
    sha = evidence_git_head_sha()
    assert len(sha) == 40
    dossier = (_REPO_ROOT / _MANDATORY_QUAL_DOCS[6]).read_text(encoding="utf-8")
    assert "READY FOR AUDIT" in dossier or "IMPLEMENTATION_SHA" in dossier
    assert sha in dossier or EBH6_PUBLICATION_SHA in dossier
