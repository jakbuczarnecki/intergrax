# © Artur Czarnecki. All rights reserved.

"""Architecture gates for canonical orchestration topology submission (Execution/Nexus)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUBMISSION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "orchestration_topology_submission.py"
)
_CONTRACT_MODULE = _REPO_ROOT / "intergrax" / "contracts" / "orchestration_topology.py"
_NODE_EXECUTION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "execution"
    / "orchestration_node_execution.py"
)
_GRAPH_EXECUTOR_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py"
)
_FANOUT_ORCHESTRATION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "multi_agent_fanout_orchestration.py"
)

_FORBIDDEN_CONSUMER_SCHEDULER_PATTERNS = (
    re.compile(r"\bGraphExecutor\s*\("),
    re.compile(r"\bNexusLoop\s*\("),
    re.compile(r"\bAgentEngine\s*\("),
    re.compile(r"\bAgentRegistry\s*\("),
)

_FORBIDDEN_IDENTITY_PATTERNS = (
    re.compile(r'metadata\[\s*["\']graph_node_id["\']\s*\]'),
    re.compile(r"fanout-slot-"),
    re.compile(r'tenant_id\s*=\s*["\']npsc-fanout["\']'),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _instantiation_calls(path: Path, class_names: tuple[str, ...]) -> list[str]:
    tree = ast.parse(_read(path), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in class_names:
            hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
    return hits


@pytest.mark.gate
def test_gate_a_submission_port_does_not_construct_consumer_graph_executor() -> None:
    source = _read(_SUBMISSION_MODULE)
    violations = [
        pattern.pattern
        for pattern in _FORBIDDEN_CONSUMER_SCHEDULER_PATTERNS
        if pattern.search(source)
    ]
    assert violations == []


@pytest.mark.gate
def test_gate_c_contract_uses_typed_orchestration_slot_id() -> None:
    source = _read(_CONTRACT_MODULE)
    assert "OrchestrationSlotId = NewType" in source
    for pattern in _FORBIDDEN_IDENTITY_PATTERNS:
        assert pattern.search(source) is None


@pytest.mark.gate
def test_gate_d_contract_has_no_shared_mutable_outcome_side_channel() -> None:
    source = _read(_CONTRACT_MODULE)
    assert "asyncio.Lock" not in source
    assert "outcomes: dict" not in source


@pytest.mark.gate
def test_gate_h_submission_port_wires_canonical_graph_executor() -> None:
    source = _read(_SUBMISSION_MODULE)
    assert "nexus_loop.graph_executor" in source
    assert "build_orchestration_topology_submission_port" in source


@pytest.mark.gate
def test_gate_i_topology_contract_modules_do_not_create_agents() -> None:
    for path in (_CONTRACT_MODULE, _SUBMISSION_MODULE, _NODE_EXECUTION_MODULE):
        hits = _instantiation_calls(path, ("Agent",))
        assert hits == [], f"{path.name} must not construct Agent: {hits}"


@pytest.mark.gate
def test_gate_j_topology_contract_modules_do_not_reference_llm() -> None:
    forbidden = ("LLMAdapter", "PrefixStubLLMAdapter", "stub_llm", "StubLLM")
    for path in (_CONTRACT_MODULE, _SUBMISSION_MODULE, _NODE_EXECUTION_MODULE):
        source = _read(path)
        violations = [name for name in forbidden if name in source]
        assert violations == [], f"{path.name} must not reference LLM stubs: {violations}"


@pytest.mark.gate
def test_shared_topology_submission_does_not_import_r2_fanout_mini_runtime() -> None:
    tree = ast.parse(_read(_SUBMISSION_MODULE), filename=str(_SUBMISSION_MODULE))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert "intergrax.runtime.execution.multi_agent_fanout_orchestration" not in modules


@pytest.mark.gate
def test_graph_executor_orchestration_work_path_uses_child_execution_runner() -> None:
    source = _read(_GRAPH_EXECUTOR_MODULE)
    assert "execute_orchestration_topology" in source
    assert "_child_runner.execute" in source
    assert "orchestration_slot_id" in source
