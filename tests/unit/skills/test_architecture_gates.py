# © Artur Czarnecki. All rights reserved.

"""Skill domain architecture boundary gates (Stage 6)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_RESOLVER_MODULE = "intergrax.skills.resolver"
_AGENT_REGISTRY_MODULE = "intergrax.runtime.registry.agent_registry"


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    return Path(module.__file__)


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_skill_resolver_does_not_import_capability_catalog() -> None:
    tree = ast.parse(_module_path(_RESOLVER_MODULE).read_text(encoding="utf-8"))
    for imported in _collect_imports(tree):
        assert not imported.startswith("intergrax.capability_catalog"), (
            f"SkillResolver must not import capability catalog: {imported}"
        )


def test_agent_registry_retains_resolved_skill_pack() -> None:
    source = _module_path(_AGENT_REGISTRY_MODULE).read_text(encoding="utf-8")
    assert "get_resolved_skill_pack" in source
    assert "_ = resolved_pack" not in source


def test_resolved_skill_pack_is_frozen() -> None:
    from intergrax.skills.resolver import ResolvedSkillPack

    assert getattr(ResolvedSkillPack, "__dataclass_params__").frozen is True
