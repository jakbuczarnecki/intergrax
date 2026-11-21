# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA): Tier-1 runtime modules import no Tier-3 module (NT-WA-01..04)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from testing_support.architecture.public_contract_boundary.policy import (
    classify_intergrax_dependency,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_EXECUTION = "intergrax/runtime/execution/"

# The 12 modules that imported ``intergrax.applications`` at the audit baseline (EQOA-A4-003).
_TWELVE_SEEDS = (
    _EXECUTION + "application_environment_context_composition.py",
    _EXECUTION + "application_graph_spec_to_plan.py",
    _EXECUTION + "application_host_orchestration_composition.py",
    _EXECUTION + "environment_host_task_execution.py",
    _EXECUTION + "harness_host_internal_composition.py",
    _EXECUTION + "host_orchestration_application_wiring_applier.py",
    _EXECUTION + "host_orchestration_assembly_validation.py",
    _EXECUTION + "host_orchestration_environment_spec_builder.py",
    _EXECUTION + "host_orchestration_planner_classifier_wiring.py",
    _EXECUTION + "host_orchestration_wiring_bundle.py",
    _EXECUTION + "scenario_host_diagnostic_wiring.py",
    "intergrax/runtime/architecture/obs_diag_x3_ast_gates.py",
)
_REMOVED_FROM_PRODUCT_TREE = (
    _EXECUTION + "host_orchestration_wiring_bundle.py",
    _EXECUTION + "scenario_host_diagnostic_wiring.py",
    "intergrax/runtime/architecture/obs_diag_x3_ast_gates.py",
)

_NEW_CONTRACTS = (
    "intergrax/contracts/application_graph_topology.py",
    "intergrax/contracts/delegation_budget_policy.py",
    "intergrax/contracts/host_orchestration_wiring_port.py",
    "intergrax/contracts/host_orchestration_environment.py",
    "intergrax/contracts/rag_retrieval_source_scope.py",
    "intergrax/contracts/workspace_tool_ids.py",
)

_IMPORT_CALLS = frozenset({"import_module", "__import__"})


def _is_tier3(module: str) -> bool:
    return module == "applications" or module.startswith(
        ("applications.", "intergrax.applications"),
    )


def _tier3_references(tree: ast.AST) -> list[tuple[int, str]]:
    """Every Tier-3 import, function-level, ``TYPE_CHECKING`` and importlib literals included."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_tier3(alias.name):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _is_tier3(node.module):
                found.append((node.lineno, node.module))
            elif node.module == "intergrax" and any(a.name == "applications" for a in node.names):
                found.append((node.lineno, "intergrax.applications"))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _IMPORT_CALLS and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if _is_tier3(first.value):
                        found.append((node.lineno, first.value))
    return found


def _violations(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for line, module in _tier3_references(tree):
            out.append(f"{path.relative_to(_REPO).as_posix()}:{line} -> {module}")
    return out


def test_wa_01_twelve_modules_have_no_tier3_import_ast() -> None:
    present = [_REPO / rel for rel in _TWELVE_SEEDS if (_REPO / rel).is_file()]
    assert _violations(present) == []
    still_in_product_tree = [rel for rel in _REMOVED_FROM_PRODUCT_TREE if (_REPO / rel).exists()]
    assert still_in_product_tree == []


def test_wa_02_runtime_tree_has_no_tier3_import_ast() -> None:
    files = sorted((_REPO / "intergrax" / "runtime").rglob("*.py"))
    assert files
    assert _violations(files) == []


def test_wa_03_tier1_modules_import_without_loading_tier3() -> None:
    modules = [
        "intergrax.runtime.execution.application_environment_context_composition",
        "intergrax.runtime.execution.application_graph_spec_to_plan",
        "intergrax.runtime.execution.application_host_orchestration_composition",
        "intergrax.runtime.execution.environment_host_task_execution",
        "intergrax.runtime.execution.environment_orchestration_materialization",
        "intergrax.runtime.execution.harness_host_internal_composition",
        "intergrax.runtime.execution.host_orchestration_application_wiring_applier",
        "intergrax.runtime.execution.host_orchestration_assembly_validation",
        "intergrax.runtime.execution.host_orchestration_environment_spec_builder",
        "intergrax.runtime.execution.host_orchestration_loop_init_spec",
        "intergrax.runtime.execution.host_orchestration_planner_classifier_wiring",
    ]
    code = textwrap.dedent(
        """
        import importlib, importlib.abc, sys

        class _RefuseTier3(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "applications" or fullname.startswith(
                    ("applications.", "intergrax.applications")
                ):
                    raise ImportError("tier-3 import refused: " + fullname)
                return None

        sys.meta_path.insert(0, _RefuseTier3())
        for name in %r:
            importlib.import_module(name)
        loaded = sorted(
            m for m in sys.modules
            if m == "applications" or m.startswith(("applications.", "intergrax.applications"))
        )
        print("TIER3_LOADED=" + ",".join(loaded))
        """,
    ) % (modules,)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "TIER3_LOADED=" in completed.stdout
    assert completed.stdout.strip().splitlines()[-1] == "TIER3_LOADED="


def _annotation_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def test_wa_04_new_contracts_import_only_contract_modules() -> None:
    for rel in _NEW_CONTRACTS:
        path = _REPO / rel
        assert path.is_file(), rel
        text = path.read_text(encoding="utf-8")
        assert "type: ignore" not in text, rel
        tree = ast.parse(text, filename=rel)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            for module in modules:
                if not module.startswith("intergrax."):
                    continue
                assert not module.startswith("intergrax.runtime."), (rel, module)
                assert ".registry" not in module and ".bootstrap" not in module, (rel, module)
                assert classify_intergrax_dependency(module) is None, (rel, module)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("HostOrchestration"):
                assert not {"Any", "object"} & _annotation_names(node), (rel, node.name)
