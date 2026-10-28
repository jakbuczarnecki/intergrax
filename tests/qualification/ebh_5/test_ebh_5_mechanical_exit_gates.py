# © Artur Czarnecki. All rights reserved.

"""EBH-5 mechanical exit criteria gates (roadmap §3.0.8 parent closure prep)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.config_x._config_x_discovery import discover_active_blocker_path_keys
from tests.qualification.ebh_5._evidence_manifest import (
    AUDIT_BASELINE_SHA,
    assert_audit_baseline_resolvable,
    evidence_git_head_sha,
)
from tests.qualification.ebh_5._replaceability_matrix import (
    CROSS_LAYER_REPLACEABILITY_MATRIX,
    matrix_by_concern_id,
    undocumented_qualified_rows,
)
from tests.qualification.ebh_5._vendor_backend_inventory import (
    VENDOR_BACKEND_DEPENDENCY_INVENTORY,
    concerns_with_extensibility_yes,
    count_unclassified,
)
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MANDATORY_QUAL_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/EBH_5_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md",
    "docs/project/maintainers/qualification/EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md",
    "docs/project/maintainers/qualification/EBH_5_VENDOR_SELECTION_REACHABILITY.md",
    "docs/project/maintainers/qualification/EBH_5_ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF.md",
    "docs/project/maintainers/qualification/EBH_5_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md",
    "docs/project/maintainers/qualification/EBH_5_SQLITE_ANNEX.md",
    "docs/project/maintainers/qualification/EBH_5_PARENT_QUALIFICATION_DOSSIER.md",
    "docs/project/maintainers/qualification/EBH_5_L9_INDEPENDENT_AUDIT_HANDOFF.md",
    "docs/project/maintainers/qualification/EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md",
)


def test_ebh5_exit_inventory_complete() -> None:
    assert count_unclassified() == 0
    assert len(VENDOR_BACKEND_DEPENDENCY_INVENTORY) >= 54


def test_ebh5_exit_matrix_complete_for_extensibility() -> None:
    promised = {c.concern_id for c in concerns_with_extensibility_yes()}
    matrix_ids = {row.concern_id for row in CROSS_LAYER_REPLACEABILITY_MATRIX}
    assert promised == matrix_ids


def test_ebh5_exit_qualified_rows_have_behavioral_refs() -> None:
    assert undocumented_qualified_rows() == ()
    by_id = matrix_by_concern_id()
    for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        if row.extensibility_classification != ExtensibilityClassification.REPLACEABLE_QUALIFIED:
            continue
        matrix_row = by_id.get(row.concern_id)
        assert matrix_row is not None
        assert matrix_row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
        assert matrix_row.behavioral_proof_ref not in ("", "PENDING")


def test_ebh5_exit_rpl_g_config_blockers_empty() -> None:
    assert discover_active_blocker_path_keys() == frozenset()


def test_ebh5_exit_mandatory_qualification_docs_present() -> None:
    for rel in _MANDATORY_QUAL_DOCS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_ebh5_exit_evidence_sha_recorded() -> None:
    assert_audit_baseline_resolvable()
    sha = evidence_git_head_sha()
    assert len(sha) == 40
    dossier = (_REPO_ROOT / _MANDATORY_QUAL_DOCS[6]).read_text(encoding="utf-8")
    assert "INDEPENDENT AUDIT PENDING" in dossier or "READY FOR INDEPENDENT" in dossier
    assert AUDIT_BASELINE_SHA in dossier or "83a376b3" in dossier
    assert (_REPO_ROOT / _MANDATORY_QUAL_DOCS[8]).is_file()
