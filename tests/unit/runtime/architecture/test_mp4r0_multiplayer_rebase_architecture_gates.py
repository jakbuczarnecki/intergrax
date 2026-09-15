# © Artur Czarnecki. All rights reserved.

"""MP-4R0 — Multiplayer production modules must not create duplicate platform authorities."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COLLABORATIVE_WORK = _REPO_ROOT / "intergrax" / "collaborative_work"
_FORBIDDEN_NEXUS_PREFIXES = ("intergrax.runtime.nexus",)
_FORBIDDEN_NEXUS_SYMBOLS = frozenset(
    {
        "GraphExecutor",
        "NexusLoop",
        "NexusIntakeRunner",
    },
)
_FORBIDDEN_AUTHORITY_CLASS_NAMES = frozenset(
    {
        "DecisionId",
        "DecisionLifecycleState",
        "DecisionRuntime",
        "MultiplayerDecisionEngine",
        "MultiplayerHitlEngine",
        "ApprovalRuntime",
        "HumanReviewRuntime",
        "EvidenceStore",
        "ExecutionReconstructor",
        "DiagnosticEngine",
        "ProblemLifecycleEngine",
    },
)
_LEGACY_DECISION_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "decision.py"
_LEGACY_APPROVAL_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "approval.py"
_LEGACY_APPROVAL_PACKAGE = _REPO_ROOT / "intergrax" / "approval"


def _production_modules(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_class_definitions(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append((node.lineno, node.name))
    return classes


def _nexus_import_violations(modules: list[Path]) -> list[str]:
    violations: list[str] = []
    for module_path in modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, module in _collect_imports(module_path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_NEXUS_PREFIXES
            ):
                violations.append(f"{rel}:{lineno} imports {module}")
            for symbol in _FORBIDDEN_NEXUS_SYMBOLS:
                if symbol in module:
                    violations.append(f"{rel}:{lineno} imports {module}")
    return violations


def test_mp4r0_collaborative_work_has_no_public_nexus_dependency() -> None:
    violations = _nexus_import_violations(_production_modules(_COLLABORATIVE_WORK))
    assert not violations, "\n".join(violations)


def test_mp4r0_collaborative_work_defines_no_duplicate_platform_authority_classes() -> None:
    violations: list[str] = []
    for module_path in _production_modules(_COLLABORATIVE_WORK):
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, name in _collect_class_definitions(module_path):
            if name in _FORBIDDEN_AUTHORITY_CLASS_NAMES:
                violations.append(f"{rel}:{lineno} defines forbidden class {name}")
    assert not violations, "\n".join(violations)


def test_mp4r0_collaborative_work_does_not_import_legacy_mp4_decision_contract() -> None:
    violations: list[str] = []
    for module_path in _production_modules(_COLLABORATIVE_WORK):
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, module in _collect_imports(module_path):
            if module == "intergrax.contracts.decision" or module.startswith(
                "intergrax.contracts.decision."
            ):
                violations.append(f"{rel}:{lineno} imports legacy MP-4B decision contract")
    assert not violations, "\n".join(violations)


def test_mp4r0_legacy_mp4_surfaces_remain_quarantined_pending_convergence() -> None:
    """Caller-proof retirement path — legacy modules exist but are not MP-4R0 expansion targets."""
    assert _LEGACY_DECISION_CONTRACT.is_file()
    assert _LEGACY_APPROVAL_CONTRACT.is_file()
    assert _LEGACY_APPROVAL_PACKAGE.is_dir()
