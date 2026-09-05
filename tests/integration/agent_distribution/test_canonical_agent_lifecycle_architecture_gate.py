# © Artur Czarnecki. All rights reserved.

"""Architecture gates for Stage 15 canonical lifecycle E2E proof."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROOF_DIR = REPO_ROOT / "tests" / "integration" / "agent_distribution"
COMPOSITION_PATH = REPO_ROOT / "testing_support" / "canonical_agent_lifecycle_composition.py"
PING_AGENT_PATH = REPO_ROOT / "testing_support" / "canonical_lifecycle_ping_agent.py"
GATE_FILE = "test_canonical_agent_lifecycle_architecture_gate.py"

_FORBIDDEN_FRAGMENTS = (
    "AgentRegistry(",
    "AgentRegistry.from_agents",
    "registry.register(",
    "NexusLoop(",
    "._private",
    "# noqa: SLF001",
    "projection_input_store.register(",
)

_VALIDATION_BYPASS_TARGETS = (
    "assert_manifest_package_closure",
    "assert_diagnostic_assembly_valid",
)

_FORBIDDEN_DIRECT_IMPORTS = (
    "InstallationService",
    "BindingService",
    "RuntimeRevisionService",
    "ActivationService",
)

_MONKEYPATCH_SETATTR = re.compile(
    r"monkeypatch\.setattr\s*\(\s*[\"']([^\"']+)[\"']",
    re.MULTILINE,
)

_FORBIDDEN_LIFECYCLE_MUTATIONS: tuple[tuple[str, str], ...] = (
    ("activation_service", "prepare_candidate"),
    ("activation_service", "commit_activation"),
    ("revision_service", "persist_candidate_revision"),
    ("revision_service", "mark_validated"),
)


def _proof_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in PROOF_DIR.glob("test_*.py")
        if path.name != GATE_FILE
    )


def _gate_targets() -> tuple[Path, ...]:
    return _proof_files() + (COMPOSITION_PATH, PING_AGENT_PATH)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names.add(alias.name)
    return names


def _forbidden_lifecycle_mutation_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        receiver = node.func.value.id
        method = node.func.attr
        if (receiver, method) in _FORBIDDEN_LIFECYCLE_MUTATIONS:
            violations.append(f"{path.name} calls forbidden lifecycle mutation {receiver}.{method}(...)")
        elif receiver == "serving_store" and method.startswith("set_"):
            violations.append(f"{path.name} calls forbidden serving store mutation {receiver}.{method}(...)")
    return violations


def _validation_bypass_violations(path: Path, text: str) -> list[str]:
    violations: list[str] = []
    for target in _MONKEYPATCH_SETATTR.findall(text):
        for validator in _VALIDATION_BYPASS_TARGETS:
            if validator in target:
                violations.append(
                    f"{path.name} monkeypatches validation boundary {validator!r}",
                )
    return violations


def test_stage15_proof_files_avoid_forbidden_lifecycle_bypass_fragments() -> None:
    violations: list[str] = []
    for path in _gate_targets():
        text = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in text:
                violations.append(f"{path.name} contains forbidden fragment {fragment!r}")
        violations.extend(_validation_bypass_violations(path, text))
        violations.extend(_forbidden_lifecycle_mutation_violations(path))
    assert not violations, "\n".join(violations)


def test_stage15_proof_tests_avoid_direct_lifecycle_service_imports() -> None:
    violations: list[str] = []
    for path in _proof_files():
        imported = _imported_names(path)
        for name in sorted(imported):
            if name in _FORBIDDEN_DIRECT_IMPORTS:
                violations.append(f"{path.name} imports forbidden lifecycle service {name!r}")
    assert not violations, "\n".join(violations)
