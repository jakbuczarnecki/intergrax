# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C-R1 — runtime behavior and weak typing gates on application contracts."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_R1_CONTRACT_MODULES = (
    _REPO_ROOT / "intergrax/applications/contracts/settings.py",
    _REPO_ROOT / "intergrax/applications/contracts/agent_ref.py",
    _REPO_ROOT / "intergrax/applications/contracts/manifest.py",
    _REPO_ROOT / "intergrax/applications/contracts/factory.py",
    _REPO_ROOT / "intergrax/skills/contracts/skill_registry_read.py",
)

_SKILL_READ_COMPAT = _REPO_ROOT / "intergrax/skills/registry/read.py"

_FORBIDDEN_SETTINGS_IMPORTS = {"os", "pathlib", "dotenv"}
_FORBIDDEN_RESOLUTION_NAMES = {
    "importlib",
    "import_module",
    "__import__",
}
_WEAK_TYPE_PATTERNS = (
    re.compile(r"\btyping\.Any\b"),
    re.compile(r"\bdict\[str,\s*Any\]"),
    re.compile(r"\bCallable\[\.\.\.,\s*Any\]"),
    re.compile(r"->\s*object\b"),
    re.compile(r"\bdict\[str,\s*object\]"),
)


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(_read_source(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_settings_contract_has_no_environment_io() -> None:
    path = _REPO_ROOT / "intergrax/applications/contracts/settings.py"
    source = _read_source(path)
    assert "from_env" not in source
    assert "EnvReader" not in source
    assert _imported_top_level_modules(path).isdisjoint(_FORBIDDEN_SETTINGS_IMPORTS)


def test_agent_ref_contract_has_no_dynamic_resolution() -> None:
    path = _REPO_ROOT / "intergrax/applications/contracts/agent_ref.py"
    source = _read_source(path)
    assert "resolve_agent_type" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in _FORBIDDEN_RESOLUTION_NAMES
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in _FORBIDDEN_RESOLUTION_NAMES


def test_manifest_has_no_resolved_agent_type_method() -> None:
    source = _read_source(_REPO_ROOT / "intergrax/applications/contracts/manifest.py")
    assert "def resolved_agent_type" not in source


def test_target_contracts_avoid_weak_any_object_escapes() -> None:
    problems: list[str] = []
    for path in _R1_CONTRACT_MODULES:
        if not path.is_file():
            continue
        source = _read_source(path)
        for pattern in _WEAK_TYPE_PATTERNS:
            if pattern.search(source):
                problems.append(f"{path.relative_to(_REPO_ROOT)}: {pattern.pattern}")
    assert not problems, "\n".join(problems)


def test_skill_registry_read_returns_typed_surface() -> None:
    source = _read_source(_REPO_ROOT / "intergrax/skills/contracts/skill_registry_read.py")
    assert "-> Any" not in source
    assert "Any |" not in source


def test_skill_registry_read_compat_is_pure_reexport() -> None:
    source = _read_source(_SKILL_READ_COMPAT)
    assert "from intergrax.skills.registry.runtime import" not in source
    assert "as_skill_registry_read" not in source
    assert "def " not in source
