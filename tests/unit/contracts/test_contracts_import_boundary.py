# © Artur Czarnecki. All rights reserved.

"""Contracts package import boundary regression gate (HARNESS-01-R4)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PUBLIC_CONTRACT_IMPORTS = (
    "from intergrax.contracts.execution_phase import ExecutionPhase; assert ExecutionPhase is not None",
    "from intergrax.contracts import AgentContract; assert AgentContract is not None",
    "from intergrax.contracts import ExecutionInterrupt; assert ExecutionInterrupt is not None",
    "from intergrax.contracts.context_budget import ContextBudgetPolicy; assert ContextBudgetPolicy is not None",
    "from intergrax.contracts.evaluator_loop import EvaluatorLoopGraphBinding; assert EvaluatorLoopGraphBinding is not None",
)


def _run_import_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _direct_runtime_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.runtime"
        ):
            hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
    return hits


@pytest.mark.parametrize("statement", _PUBLIC_CONTRACT_IMPORTS)
def test_public_contract_imports(statement: str) -> None:
    completed = _run_import_subprocess(statement)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_contracts_package_has_no_direct_nexus_imports() -> None:
    root = _REPO_ROOT / "intergrax" / "contracts"
    hits: list[str] = []
    for path in root.rglob("*.py"):
        for mod in _direct_runtime_imports(path):
            if mod == "intergrax.runtime.nexus" or mod.startswith("intergrax.runtime.nexus."):
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{mod}")
    assert hits == [], "direct contracts → Nexus imports:\n" + "\n".join(hits)


def test_contracts_package_init_does_not_import_runtime() -> None:
    init_path = _REPO_ROOT / "intergrax" / "contracts" / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    forbidden_prefixes = ("intergrax.runtime", "intergrax.decision")
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("from ") and not stripped.startswith("import "):
            continue
        for prefix in forbidden_prefixes:
            assert prefix not in stripped, (
                f"contracts/__init__.py imports forbidden dependency: {line}"
            )
        assert "runtime_mapping" not in stripped


def test_runtime_answer_mapping_lives_outside_contracts() -> None:
    assert not (_REPO_ROOT / "intergrax" / "contracts" / "runtime_mapping.py").exists()
    mapping = _REPO_ROOT / "intergrax" / "agents" / "runtime_answer_mapping.py"
    assert mapping.is_file()
    source = mapping.read_text(encoding="utf-8")
    assert "RuntimeAnswer" in source
    assert "runtime_answer_to_agent_result" in source
