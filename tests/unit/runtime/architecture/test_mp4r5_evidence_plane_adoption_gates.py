# © Artur Czarnecki. All rights reserved.

"""MP-4R5 — canonical Evidence Plane adoption boundaries for Multiplayer."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COLLABORATIVE_ROOT = _REPO_ROOT / "intergrax" / "collaborative_work"
_COLLABORATIVE_CONTRACTS = (
    _REPO_ROOT / "intergrax" / "contracts" / "collaborative_decision_binding.py",
    _REPO_ROOT / "intergrax" / "contracts" / "collaborative_functional_evidence_projection.py",
)
_FUNCTIONAL_EVIDENCE_CONTRACT_ROOT = _REPO_ROOT / "intergrax" / "contracts" / "functional_evidence"
_SERVICE_PATH = _COLLABORATIVE_ROOT / "decision_binding_service.py"
_FORBIDDEN_EVIDENCE_STORE_NAMES = frozenset(
    {
        "MultiplayerEvidenceRepository",
        "CollaborativeEvidenceStore",
        "DecisionBindingEvidenceRepository",
        "WorkspaceEvidenceDatabase",
        "MultiplayerEvidence",
        "CollaborativeEvidenceRecord",
        "DecisionBindingEvidenceRecord",
    },
)
_FORBIDDEN_RUNTIME_OBS_PREFIX = "intergrax.runtime.observability"
_FORBIDDEN_DIAG_PREFIX = "intergrax.runtime.diagnostics"
_FORBIDDEN_VENDOR_TOKENS = (
    "opentelemetry",
    "otlp",
    "datadog",
    "MongoObjectId",
)
_FORBIDDEN_OUTCOME_TOKENS = (
    "DecisionLifecycle",
    "DecisionOutcome",
    "DecisionResolution",
    "DecisionStatus",
    "HumanApprovalOutcome",
    "ExecutionAuthorization",
    "PolicyAction",
    "GovernanceAuthorization",
)
_FORBIDDEN_NEXUS_PREFIX = "intergrax.runtime.nexus"
_SOURCE_OF_TRUTH_MATRIX: tuple[tuple[str, str], ...] = (
    ("CollaborativeDecisionBinding", "Multiplayer"),
    ("Decision resolution", "Decision System"),
    ("Authorization", "Governance / HITL"),
    ("Execution lifecycle", "Execution Runtime"),
    ("Evidence record", "Evidence Plane"),
    ("Diagnostics", "Diagnostics"),
)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _collaborative_production_modules() -> list[Path]:
    modules = sorted(_COLLABORATIVE_ROOT.rglob("*.py"))
    return [path for path in modules if path.name != "__init__.py"]


def test_no_multiplayer_evidence_repository_or_store_types() -> None:
    violations: list[str] = []
    for module in _collaborative_production_modules():
        source = module.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_EVIDENCE_STORE_NAMES:
            if name in source:
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {name}")
    assert violations == []


def test_collaborative_work_does_not_import_observability_runtime_implementation() -> None:
    violations: list[str] = []
    for module in _collaborative_production_modules():
        for imported in _collect_imports(module):
            if imported.startswith(_FORBIDDEN_RUNTIME_OBS_PREFIX):
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_collaborative_work_does_not_import_diagnostics() -> None:
    violations: list[str] = []
    for module in _collaborative_production_modules():
        for imported in _collect_imports(module):
            if imported.startswith(_FORBIDDEN_DIAG_PREFIX):
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_collaborative_work_has_no_vendor_or_otlp_coupling() -> None:
    violations: list[str] = []
    for module in _collaborative_production_modules():
        source = module.read_text(encoding="utf-8-sig").lower()
        for token in _FORBIDDEN_VENDOR_TOKENS:
            if token.lower() in source:
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {token}")
    assert violations == []


def test_decision_binding_service_has_no_decision_or_governance_outcome_ownership() -> None:
    source = _SERVICE_PATH.read_text(encoding="utf-8-sig")
    for token in _FORBIDDEN_OUTCOME_TOKENS:
        assert token not in source


def test_collaborative_evidence_adoption_uses_functional_evidence_persistence_contract_only() -> None:
    evidence_module = _COLLABORATIVE_ROOT / "decision_binding_evidence.py"
    imports = _collect_imports(evidence_module)
    assert any(item.endswith("functional_evidence.persistence") for item in imports)
    assert not any("sqlite" in item or "postgresql" in item for item in imports)
    assert not any(item.startswith(_FORBIDDEN_RUNTIME_OBS_PREFIX) for item in imports)


def test_functional_evidence_contracts_do_not_import_collaborative_work() -> None:
    violations: list[str] = []
    for module in _FUNCTIONAL_EVIDENCE_CONTRACT_ROOT.rglob("*.py"):
        for imported in _collect_imports(module):
            if "collaborative_work" in imported or "collaborative_decision_binding" in imported:
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_collaborative_work_has_no_nexus_dependency() -> None:
    violations: list[str] = []
    for module in _collaborative_production_modules():
        for imported in _collect_imports(module):
            if imported.startswith(_FORBIDDEN_NEXUS_PREFIX):
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_decision_binding_service_unchanged_mp4r4_authority_surface() -> None:
    source = _SERVICE_PATH.read_text(encoding="utf-8-sig")
    assert "CollaborativeDecisionBindingRepository" in source
    assert "functional_evidence" not in source
    assert "FunctionalEvidencePersistence" not in source


def test_projection_contract_documents_association_gap() -> None:
    contract = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_functional_evidence_projection.py"
    source = contract.read_text(encoding="utf-8-sig")
    assert "CollaborativeDecisionBindingAssociationNotRepresentable" in source
    assert "PipelineEvidenceKind" not in source or "frozen" in source.lower()


def test_source_of_truth_matrix_documented_in_gate_module() -> None:
    source = Path(__file__).read_text(encoding="utf-8-sig")
    for concern, owner in _SOURCE_OF_TRUTH_MATRIX:
        assert concern in source
        assert owner in source


def test_no_direct_db_evidence_provider_imports_in_collaborative_work() -> None:
    forbidden = (
        "document_store_functional_evidence",
        "in_memory_functional_evidence",
        "sqlite_functional_evidence",
        "postgresql_functional_evidence",
    )
    violations: list[str] = []
    for module in _collaborative_production_modules():
        for imported in _collect_imports(module):
            if any(fragment in imported for fragment in forbidden):
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_collaborative_contracts_reference_decision_proposal_ref_not_outcomes() -> None:
    for path in _COLLABORATIVE_CONTRACTS:
        source = path.read_text(encoding="utf-8-sig")
        for token in _FORBIDDEN_OUTCOME_TOKENS:
            assert token not in source


def test_decision_binding_service_does_not_emit_evidence_on_read_paths() -> None:
    source = _SERVICE_PATH.read_text(encoding="utf-8-sig")
    for method_name in ("get_binding", "list_bindings_for_work_item", "list_bindings_for_decision_proposal"):
        block = source.split(f"def {method_name}", maxsplit=1)[1].split("\n    def ", maxsplit=1)[0]
        assert "append" not in block
        assert "evidence" not in block.lower()


_APPLICATION_PATH = _COLLABORATIVE_ROOT / "decision_binding_application.py"
_COMPOSITION_PATH = _COLLABORATIVE_ROOT / "decision_binding_composition.py"
_PROJECTION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "collaborative_functional_evidence_projection.py"
)


def test_create_outcome_projection_does_not_require_binding_for_failed() -> None:
    source = _PROJECTION_CONTRACT.read_text(encoding="utf-8-sig")
    assert "binding: CollaborativeDecisionBinding | None = None" in source
    assert "tenant_id: str" in source


def test_application_boundary_emits_create_outcome_evidence() -> None:
    source = _APPLICATION_PATH.read_text(encoding="utf-8-sig")
    assert "append_decision_binding_create_outcome_evidence" in source
    assert "DefaultCollaborativeFunctionalEvidenceProjection" not in source
    for imported in _collect_imports(_APPLICATION_PATH):
        assert not imported.startswith(_FORBIDDEN_RUNTIME_OBS_PREFIX)


def test_composition_wires_application_create_binding_path() -> None:
    source = _COMPOSITION_PATH.read_text(encoding="utf-8-sig")
    assert "build_collaborative_decision_binding_application_from_artifacts_bundle" in source
    assert "DefaultCollaborativeFunctionalEvidenceProjection" not in source
    imports = _collect_imports(_COMPOSITION_PATH)
    assert not any("in_memory_functional_evidence" in item for item in imports)
    assert not any("document_store_functional_evidence" in item for item in imports)


def test_pipeline_evidence_kind_enum_unchanged_by_mp4r5_task_surface() -> None:
    models_path = _FUNCTIONAL_EVIDENCE_CONTRACT_ROOT / "models.py"
    source = models_path.read_text(encoding="utf-8-sig")
    assert "class PipelineEvidenceKind(StrEnum):" in source
    for kind in (
        "ARTIFACT_LINEAGE",
        "OPERATION_OUTCOME",
        "CANDIDATE_RANK",
        "SELECTION",
        "OUTPUT_RELATION",
        "VALIDATION",
    ):
        assert kind in source
