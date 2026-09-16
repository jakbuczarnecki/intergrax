# © Artur Czarnecki. All rights reserved.

"""MP-4R0 — Multiplayer production modules must not create duplicate platform authorities."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MULTIPLAYER_PRODUCTION_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "intergrax" / "collaborative_work",
)
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
_DECISION_INTEGRATION_PACKAGE = _REPO_ROOT / "intergrax" / "contracts" / "decision"
_INTEGRATION_NAMESPACE_PREFIXES = (
    "intergrax.contracts.decision",
    "intergrax.contracts.decision.integration",
)
_LEGACY_APPROVAL_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "approval.py"
_LEGACY_APPROVAL_PACKAGE = _REPO_ROOT / "intergrax" / "approval"
_CANONICAL_DECISION_MODULE_PREFIXES = (
    "intergrax.contracts.decision_identity",
    "intergrax.contracts.decision_lifecycle",
    "intergrax.contracts.decision_human_review",
    "intergrax.contracts.decision_record",
    "intergrax.contracts.decision_finalization",
    "intergrax.contracts.decision_checkpoint",
    "intergrax.contracts.decision_revision",
    "intergrax.contracts.decision_resolution",
    "intergrax.contracts.decision_verification",
    "intergrax.contracts.decision_verification_stage",
    "intergrax.contracts.decision_authorization",
    "intergrax.contracts.decision_authoritative_exposure",
    "intergrax.contracts.decision_artifact_registry",
    "intergrax.contracts.decision_strategy",
    "intergrax.contracts.decision_exposure_selection",
    "intergrax.contracts.decision_coordination",
    "intergrax.contracts.decision_disagreement",
)


def _production_modules(roots: Iterable[Path]) -> list[Path]:
    modules: list[Path] = []
    for root in roots:
        modules.extend(
            sorted(path for path in root.rglob("*.py") if path.is_file()),
        )
    return modules


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


def _is_legacy_mp4b_decision_import(module: str) -> bool:
    if any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _INTEGRATION_NAMESPACE_PREFIXES
    ):
        return False
    if not module.startswith("intergrax.contracts.decision."):
        return module == "intergrax.contracts.decision"
    return not any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _CANONICAL_DECISION_MODULE_PREFIXES
    )


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


def _duplicate_authority_class_violations(modules: list[Path]) -> list[str]:
    violations: list[str] = []
    for module_path in modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, name in _collect_class_definitions(module_path):
            if name in _FORBIDDEN_AUTHORITY_CLASS_NAMES:
                violations.append(f"{rel}:{lineno} defines forbidden class {name}")
    return violations


def _legacy_decision_import_violations(modules: list[Path]) -> list[str]:
    violations: list[str] = []
    for module_path in modules:
        rel = module_path.relative_to(_REPO_ROOT)
        for lineno, module in _collect_imports(module_path):
            if _is_legacy_mp4b_decision_import(module):
                violations.append(f"{rel}:{lineno} imports legacy MP-4B decision contract")
    return violations


@pytest.fixture(name="multiplayer_production_modules")
def fixture_multiplayer_production_modules() -> list[Path]:
    return _production_modules(_MULTIPLAYER_PRODUCTION_ROOTS)


def test_mp4r0_protected_multiplayer_roots_are_declared() -> None:
    assert _MULTIPLAYER_PRODUCTION_ROOTS
    for root in _MULTIPLAYER_PRODUCTION_ROOTS:
        assert root.is_dir(), f"missing Multiplayer production root: {root}"


def test_mp4r0_multiplayer_production_has_no_public_nexus_dependency(
    multiplayer_production_modules: list[Path],
) -> None:
    violations = _nexus_import_violations(multiplayer_production_modules)
    assert not violations, "\n".join(violations)


def test_mp4r0_multiplayer_production_defines_no_duplicate_platform_authority_classes(
    multiplayer_production_modules: list[Path],
) -> None:
    violations = _duplicate_authority_class_violations(multiplayer_production_modules)
    assert not violations, "\n".join(violations)


def test_mp4r0_multiplayer_production_does_not_import_legacy_mp4_decision_contract(
    multiplayer_production_modules: list[Path],
) -> None:
    violations = _legacy_decision_import_violations(multiplayer_production_modules)
    assert not violations, "\n".join(violations)


def test_mp4r0_legacy_mp4_surfaces_post_mp4r2_convergence() -> None:
    """MP-4R1 retired MP-4B module; MP-4R2 retired legacy Approval surfaces."""
    legacy_decision_module = _REPO_ROOT / "intergrax" / "contracts" / "decision.py"
    assert not legacy_decision_module.is_file()
    assert _DECISION_INTEGRATION_PACKAGE.is_dir()
    assert not _LEGACY_APPROVAL_CONTRACT.is_file()
    assert not _LEGACY_APPROVAL_PACKAGE.is_dir()
