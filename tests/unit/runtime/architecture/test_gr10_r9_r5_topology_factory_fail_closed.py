# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-R5 — canonical topology factories fail-closed; lab path explicitly isolated."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.execution.orchestration_topology_submission import (
    CanonicalOrchestrationTopologySubmissionPort,
    OrchestrationTopologyMseCompositionError,
    build_lab_orchestration_topology_continuation_port,
    build_lab_orchestration_topology_submission_port,
    build_orchestration_topology_continuation_port,
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SUBMISSION = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "orchestration_topology_submission.py"
)
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"


def _canonical_factory_calls_lab_builder(name: str, lab_builder: str) -> bool:
    tree = ast.parse(_SUBMISSION.read_text(encoding="utf-8-sig"), filename=str(_SUBMISSION))
    func: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            func = node
            break
    assert func is not None
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == lab_builder:
            return True
        if isinstance(target, ast.Attribute) and target.attr == lab_builder:
            return True
    return False


def test_r9_r5_canonical_submission_missing_policy_fail_closed() -> None:
    with pytest.raises(OrchestrationTopologyMseCompositionError):
        build_orchestration_topology_submission_port(NexusLoop(AgentRegistry()))


def test_r9_r5_canonical_continuation_missing_policy_fail_closed() -> None:
    with pytest.raises(OrchestrationTopologyMseCompositionError):
        build_orchestration_topology_continuation_port(NexusLoop(AgentRegistry()))


def test_r9_r5_lab_submission_allows_missing_policy() -> None:
    port = build_lab_orchestration_topology_submission_port(NexusLoop(AgentRegistry()))
    assert port is not None


def test_r9_r5_lab_continuation_allows_missing_policy() -> None:
    port = build_lab_orchestration_topology_continuation_port(NexusLoop(AgentRegistry()))
    assert port is not None


def test_r9_r5_no_implicit_lab_fallback_in_canonical_submission_factory() -> None:
    assert not _canonical_factory_calls_lab_builder(
        "build_orchestration_topology_submission_port",
        "build_lab_orchestration_topology_submission_port",
    )


def test_r9_r5_no_implicit_lab_fallback_in_canonical_continuation_factory() -> None:
    assert not _canonical_factory_calls_lab_builder(
        "build_orchestration_topology_continuation_port",
        "build_lab_orchestration_topology_continuation_port",
    )


def test_r9_r5_direct_production_construction_rejects_missing_policy() -> None:
    graph_executor = NexusLoop(AgentRegistry()).graph_executor
    with pytest.raises(OrchestrationTopologyMseCompositionError):
        CanonicalOrchestrationTopologySubmissionPort(
            _graph_executor=graph_executor,
            _slot_mse_policy=None,
            _production_topology=True,
        )


def test_r9_r5_production_package_surfaces_do_not_import_lab_builders() -> None:
    forbidden = (
        "build_lab_orchestration_topology_submission_port",
        "build_lab_orchestration_topology_continuation_port",
    )
    for path in _INTERGRAX_ROOT.rglob("*.py"):
        if path.name == "orchestration_topology_submission.py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        for name in forbidden:
            assert name not in text, f"{path.relative_to(_REPO_ROOT)} must not reference {name}"
