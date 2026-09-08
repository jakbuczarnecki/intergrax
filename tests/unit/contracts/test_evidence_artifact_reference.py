# © Artur Czarnecki. All rights reserved.

"""MP-3G — evidence-side WorkArtifactVersion reference contract tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.evidence_artifact_reference import EvidenceArtifactVersionLink

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVIDENCE_REF_PATH = _REPO_ROOT / "intergrax" / "contracts" / "evidence_artifact_reference.py"


def test_evidence_reference_imports_work_artifact_version_ref_only() -> None:
    tree = ast.parse(_EVIDENCE_REF_PATH.read_text(encoding="utf-8"), filename=str(_EVIDENCE_REF_PATH))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("intergrax."):
            imports.append(node.module)
    assert imports == ["intergrax.contracts.collaborative_work"]


def test_evidence_reference_has_no_proof_receipt_import() -> None:
    source = _EVIDENCE_REF_PATH.read_text(encoding="utf-8")
    assert "ProofReceipt" not in source
    assert "proofs.receipts" not in source


def test_evidence_reference_strict_shape() -> None:
    version_ref = WorkArtifactVersionRef(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        work_item_id="work-item-1",
        work_artifact_id="artifact-1",
        work_artifact_version_id="artifact-version-1",
    )
    link = EvidenceArtifactVersionLink(
        evidence_id="evidence-1",
        artifact_version=version_ref,
    )
    assert link.evidence_id == "evidence-1"
    assert link.artifact_version == version_ref
    with pytest.raises(ValidationError):
        EvidenceArtifactVersionLink(
            evidence_id="evidence-1",
            artifact_version=version_ref,
            metadata={"spoof": True},
        )
