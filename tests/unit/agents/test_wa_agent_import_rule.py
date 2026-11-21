# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA): Tier-2 agent import rule of DP-14 (NT-WA-13, NT-WA-14, NT-WA-17).

Tier-2 agents import ``intergrax.integrations`` only through ``.contracts.*`` leaves; the
integration providers, ``integrations._shared`` and ``tools.providers`` are forbidden. This is a
ticket-local test of the rule; the gate itself (``PT-G01``) is not edited here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_AGENTS = _REPO / "agents"

# Copied from scripts/maintenance/check_agent_registry_bypass.py (the script is not imported).
_OLD_GATE_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+intergrax\.(?:integrations|tools)\.",
    re.MULTILINE,
)

_FORBIDDEN_PREFIXES = (
    "intergrax.integrations._shared",
    "intergrax.integrations.providers",
    "intergrax.tools.providers",
)
_CONTRACT_LEAF_PREFIX = "intergrax.integrations.contracts"
_CONTRACT_LEAF_FILES = frozenset(
    {
        "agents/external_contractor_adapter/external_contractor_adapter_agent.py",
        "agents/external_contractor_adapter/external_work_adapter.py",
        "agents/external_contractor_adapter/external_work_reconciliation_plugin.py",
        "agents/external_contractor_adapter/steps/domain_job.py",
        "agents/external_contractor_adapter/strict_host_dependency_injection.py",
        "agents/web_search_qualifier/steps/web_search_job.py",
    },
)


def _production_agent_files() -> list[Path]:
    files = [
        path
        for path in sorted(_AGENTS.rglob("*.py"))
        if "tests" not in path.relative_to(_REPO).parts
        and not path.name.startswith("test_")
        and path.name != "conftest.py"
    ]
    assert files
    return files


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    modules: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append((node.lineno, node.module))
            if node.module in {"intergrax", "intergrax.integrations", "intergrax.tools"}:
                modules.extend((node.lineno, f"{node.module}.{alias.name}") for alias in node.names)
    return modules


def test_wa_13_agents_tree_uses_only_integration_contract_leaves() -> None:
    violations: list[str] = []
    for path in _production_agent_files():
        rel = path.relative_to(_REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        for line, module in _imported_modules(tree):
            if module.startswith(_FORBIDDEN_PREFIXES):
                violations.append(f"{rel}:{line} -> {module}")
            elif module.startswith("intergrax.integrations") and not (
                module == _CONTRACT_LEAF_PREFIX or module.startswith(_CONTRACT_LEAF_PREFIX + ".")
            ):
                violations.append(f"{rel}:{line} -> {module}")
    assert violations == []


def test_wa_14_qualifier_agents_read_no_environment() -> None:
    violations: list[str] = []
    for qualifier in ("web_search_qualifier", "tool_selection_qualifier"):
        for path in sorted((_AGENTS / qualifier).rglob("*.py")):
            if "tests" in path.relative_to(_REPO).parts or path.name.startswith("test_"):
                continue
            rel = path.relative_to(_REPO).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id == "os" and node.attr in {"environ", "getenv", "putenv"}:
                        violations.append(f"{rel}:{node.lineno} -> os.{node.attr}")
                if isinstance(node, ast.ImportFrom) and node.module == "os":
                    for alias in node.names:
                        if alias.name in {"environ", "getenv", "putenv"}:
                            violations.append(f"{rel}:{node.lineno} -> from os import {alias.name}")
    assert violations == []


def test_wa_17_old_gate_pattern_matches_only_contract_leaf_files() -> None:
    matching: dict[str, list[str]] = {}
    for path in _production_agent_files():
        rel = path.relative_to(_REPO).as_posix()
        if rel == "agents/__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if _OLD_GATE_PATTERN.match(line)]
        if lines:
            matching[rel] = lines
    assert set(matching) == _CONTRACT_LEAF_FILES
    for rel, lines in matching.items():
        for line in lines:
            assert re.match(
                r"(?:from|import)\s+intergrax\.integrations\.contracts\.",
                line,
            ), (rel, line)
    assert "agents/local_search/steps/search_job.py" not in matching
    assert "agents/tool_selection_qualifier/tool_selection.py" not in matching
    assert "agents/tool_selection_qualifier/steps/tool_selection_job.py" not in matching
    assert not (_AGENTS / "web_search_qualifier" / "search_provider_resolver.py").exists()
