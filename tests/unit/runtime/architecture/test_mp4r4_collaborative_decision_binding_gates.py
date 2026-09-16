# © Artur Czarnecki. All rights reserved.

"""MP-4R4 — CollaborativeDecisionBinding ownership and authority boundary gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BINDING_ROOT = _REPO_ROOT / "intergrax" / "collaborative_work"
_BINDING_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_decision_binding.py"
_DECISION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "decision"
_SERVICE_PATH = _BINDING_ROOT / "decision_binding_service.py"
_WIRE_PATH = _REPO_ROOT / "intergrax" / "contracts" / "decision_proposal_ref_wire.py"
_FORBIDDEN_FEATURE_DOMAIN_PREFIXES = (
    "intergrax.knowledge",
    "intergrax.memory",
    "intergrax.rag",
    "intergrax.marketplace",
    "intergrax.applications",
)
_FORBIDDEN_RUNTIME_PREFIXES = (
    "intergrax.runtime.execution",
    "intergrax.runtime.nexus",
)
_FORBIDDEN_OUTCOME_TOKENS = (
    "DecisionLifecycle",
    "DecisionOutcome",
    "DecisionResolution",
    "DecisionStatus",
    "HumanApprovalOutcome",
    "ExecutionAuthorization",
    "PolicyAction",
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


def _binding_modules() -> list[Path]:
    modules = [_BINDING_CONTRACT, _SERVICE_PATH]
    modules.extend(sorted(_BINDING_ROOT.glob("**/decision_binding*.py")))
    modules.extend(
        path
        for path in _BINDING_ROOT.glob("**/*repository*.py")
        if path.is_file() and "decision_binding" in path.read_text(encoding="utf-8-sig", errors="ignore")
    )
    return list(dict.fromkeys(modules))


def test_binding_contract_uses_decision_proposal_ref() -> None:
    source = _BINDING_CONTRACT.read_text(encoding="utf-8-sig")
    assert "DecisionProposalRef" in source
    assert "decision_proposal: DecisionProposalRef" in source


def test_binding_modules_do_not_import_execution_or_nexus_runtime() -> None:
    violations: list[str] = []
    for module in _binding_modules():
        for imported in _collect_imports(module):
            if any(imported.startswith(prefix) for prefix in _FORBIDDEN_RUNTIME_PREFIXES):
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_binding_contract_has_no_decision_outcome_fields() -> None:
    source = _BINDING_CONTRACT.read_text(encoding="utf-8-sig")
    for token in _FORBIDDEN_OUTCOME_TOKENS:
        assert token not in source


def test_decision_runtime_does_not_import_collaborative_binding() -> None:
    if not _DECISION_ROOT.is_dir():
        return
    violations: list[str] = []
    for module in _DECISION_ROOT.rglob("*.py"):
        for imported in _collect_imports(module):
            if "collaborative_decision_binding" in imported:
                violations.append(f"{module.relative_to(_REPO_ROOT)}: {imported}")
    assert violations == []


def test_decision_proposal_ref_wire_has_no_feature_domain_imports() -> None:
    violations: list[str] = []
    for imported in _collect_imports(_WIRE_PATH):
        if any(imported.startswith(prefix) for prefix in _FORBIDDEN_FEATURE_DOMAIN_PREFIXES):
            violations.append(imported)
    assert violations == []


def test_service_depends_on_repository_protocol_not_sqlite() -> None:
    imports = _collect_imports(_SERVICE_PATH)
    assert not any("sqlite" in item for item in imports)
    assert not any("postgresql" in item for item in imports)
    source = _SERVICE_PATH.read_text(encoding="utf-8-sig")
    assert "CollaborativeDecisionBindingRepository" in source


def test_work_item_contract_has_no_decision_fields() -> None:
    work_item_source = (_REPO_ROOT / "intergrax" / "contracts" / "collaborative_work.py").read_text(
        encoding="utf-8-sig",
    )
    assert "class WorkItem" in work_item_source
    work_item_block = work_item_source.split("class WorkItem", maxsplit=1)[1].split("class ", maxsplit=1)[0]
    assert "decision" not in work_item_block.lower()
