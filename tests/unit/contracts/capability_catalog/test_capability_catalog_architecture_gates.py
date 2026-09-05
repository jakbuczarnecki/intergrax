# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 1 architecture boundary regression gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_MODULE = "intergrax.contracts.capability_catalog"

_FORBIDDEN_IMPORT_PREFIXES = (
    "applications",
    "intergrax.applications",
    "intergrax.agent_distribution",
    "intergrax.autonomous_work",
    "intergrax.runtime",
    "intergrax.skills",
    "intergrax.tools",
)

_FORBIDDEN_REGISTRY_TOKENS = (
    "AgentRegistry",
    "SkillRegistry",
    "ToolRegistry",
    "UniversalCapabilityEngine",
    "UniversalRegistry",
    "CapabilityRegistry",
)

_FORBIDDEN_RUNTIME_MUTATION_TOKENS = (
    "def install",
    "def enable",
    "def activate",
    "def materialize",
    "def register",
    "def mutate",
)


def _package_paths() -> list[Path]:
    package = importlib.import_module(_PACKAGE_MODULE)
    assert package.__path__ is not None
    return sorted(Path(path) for path in package.__path__)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_capability_catalog_package_has_no_forbidden_imports() -> None:
    for path in _package_paths():
        if path.name == "__pycache__":
            continue
        if not path.suffix == ".py":
            continue
        joined = "\n".join(_collect_imports(path))
        for prefix in _FORBIDDEN_IMPORT_PREFIXES:
            for imported in joined.splitlines():
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.name} imports forbidden dependency: {imported}",
                    )


def test_capability_catalog_package_has_no_registry_unification_tokens() -> None:
    for path in _package_paths():
        if not path.suffix == ".py":
            continue
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_REGISTRY_TOKENS:
            assert token not in source, f"{path.name} defines forbidden token {token}"


def test_capability_catalog_package_has_no_runtime_mutation_api() -> None:
    for path in _package_paths():
        if not path.suffix == ".py":
            continue
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_RUNTIME_MUTATION_TOKENS:
            assert token not in source, f"{path.name} exposes forbidden API {token}"


def test_capability_catalog_import_smoke_subprocess() -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[4]
    statement = """
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityStageVocabulary,
    normalize_discovery_identity_set,
)
assert CapabilityKind.AGENT.value == "agent"
assert CapabilityStageVocabulary.DISCOVERED.value == "discovered"
assert normalize_discovery_identity_set(()) == ()
print("capability catalog import smoke OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
