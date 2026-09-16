# © Artur Czarnecki. All rights reserved.

"""MP-4R7 — enterprise integration architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_R7_ROOT = _REPO_ROOT / "testing_support" / "mp4r7_enterprise_integration"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.integrations.providers",
    "intergrax.runtime.nexus",
    "intergrax.runtime.execution.continuation.service",
    "intergrax.runtime.execution.continuation.lifecycle_driver",
)
_FORBIDDEN_REFLECTION = ("getattr(", "setattr(", "hasattr(", "vars(", ".__dict__")
_FORBIDDEN_AUTHORITY_NAMES = (
    "MultiplayerDecisionExecutionManager",
    "UnifiedApprovalExecutor",
    "R7Coordinator",
)


def _r7_modules() -> list[Path]:
    return sorted(path for path in _R7_ROOT.rglob("*.py") if path.is_file())


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def test_mp4r7_package_exists() -> None:
    assert _R7_ROOT.is_dir()


def test_mp4r7_no_vendor_internals_in_composition_layer() -> None:
    violations: list[str] = []
    for module in _r7_modules():
        rel = module.relative_to(_REPO_ROOT)
        for lineno, imported in _collect_imports(module):
            if any(imported.startswith(prefix) for prefix in _FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{rel}:{lineno} imports {imported}")
    assert not violations, "\n".join(violations)


def test_mp4r7_no_reflection_in_qualification_code() -> None:
    violations: list[str] = []
    for module in _r7_modules():
        text = module.read_text(encoding="utf-8-sig")
        rel = module.relative_to(_REPO_ROOT)
        for token in _FORBIDDEN_REFLECTION:
            if token in text:
                violations.append(f"{rel} uses forbidden reflection token {token!r}")
    assert not violations, "\n".join(violations)


def test_mp4r7_no_facade_authority_types() -> None:
    violations: list[str] = []
    for module in _r7_modules():
        text = module.read_text(encoding="utf-8-sig")
        rel = module.relative_to(_REPO_ROOT)
        for name in _FORBIDDEN_AUTHORITY_NAMES:
            if f"class {name}" in text:
                violations.append(f"{rel} defines forbidden authority facade {name}")
    assert not violations, "\n".join(violations)


def test_mp4r7_scenario_uses_public_continuation_port() -> None:
    scenario = _R7_ROOT / "scenario.py"
    text = scenario.read_text(encoding="utf-8-sig")
    assert "continuation_port" in text
    assert "ExecutionContinuationPort" not in text or "composition.continuation_port" in text
    assert "ExecutionContinuationService(" not in text


def test_mp4r7_scenario_uses_canonical_human_review_continuation_bridge() -> None:
    scenario = _R7_ROOT / "scenario.py"
    text = scenario.read_text(encoding="utf-8-sig")
    assert (
        "execution_continuation_resolution_command_from_decision_human_review_decision"
        in text
    )
    assert "_HUMAN_REQUEST_ID" not in text
    assert "ExecutionHumanVerdict.APPROVE" not in text


def test_mp4r7_scenario_does_not_fabricate_primary_error_in_evidence_handler() -> None:
    scenario = _R7_ROOT / "scenario.py"
    text = scenario.read_text(encoding="utf-8-sig")
    assert "Mp4R7ProtectedOperationError" in text
    assert 'primary_error = RuntimeError("protected operation failed")' not in text


def test_mp4r7_canonical_contracts_importable() -> None:
    from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
    from intergrax.contracts.decision_human_review import DecisionHumanReviewPort
    from intergrax.contracts.execution_continuation import ExecutionContinuationPort
    from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence

    assert CollaborativeDecisionBinding is not None
    assert DecisionHumanReviewPort is not None
    assert ExecutionContinuationPort is not None
    assert FunctionalEvidencePersistence is not None
