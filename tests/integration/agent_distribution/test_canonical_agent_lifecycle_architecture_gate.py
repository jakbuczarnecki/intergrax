# © Artur Czarnecki. All rights reserved.

"""Architecture gates for Stage 15 canonical lifecycle E2E proof."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROOF_DIR = REPO_ROOT / "tests" / "integration" / "agent_distribution"
COMPOSITION_PATH = REPO_ROOT / "testing_support" / "canonical_agent_lifecycle_composition.py"
GATE_FILE = "test_canonical_agent_lifecycle_architecture_gate.py"

_FORBIDDEN_FRAGMENTS = (
    "AgentRegistry(",
    "AgentRegistry.from_agents",
    "registry.register(",
    "NexusLoop(",
    "._private",
)

_FORBIDDEN_DIRECT_IMPORTS = (
    "InstallationService",
    "BindingService",
    "RuntimeRevisionService",
    "ActivationService",
)


def _proof_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in PROOF_DIR.glob("test_*.py")
        if path.name != GATE_FILE
    )


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names.add(alias.name)
    return names


def test_stage15_proof_files_avoid_forbidden_lifecycle_bypass_fragments() -> None:
    targets = _proof_files() + (COMPOSITION_PATH,)
    violations: list[str] = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in text:
                violations.append(f"{path.name} contains forbidden fragment {fragment!r}")
    assert not violations, "\n".join(violations)


def test_stage15_proof_tests_avoid_direct_lifecycle_service_imports() -> None:
    violations: list[str] = []
    for path in _proof_files():
        imported = _imported_names(path)
        for name in sorted(imported):
            if name in _FORBIDDEN_DIRECT_IMPORTS:
                violations.append(f"{path.name} imports forbidden lifecycle service {name!r}")
    assert not violations, "\n".join(violations)
