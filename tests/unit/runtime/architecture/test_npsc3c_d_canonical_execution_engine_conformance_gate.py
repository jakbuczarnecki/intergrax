# © Artur Czarnecki. All rights reserved.

"""NPSC-3C-D: canonical execution engine conformance gate."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.runtime.interactions.intake_service import InteractionIntakeService

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TASK_EXECUTOR_PATH = _REPO_ROOT / "intergrax" / "runtime" / "interactions" / "task_executor.py"
_HOST_TASK_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_FACADE_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "facade.py"
_INTERACTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "interactions"


def test_npsc3c_d_interaction_intake_has_no_direct_nexus_path() -> None:
    source = Path(inspect.getfile(InteractionIntakeService)).read_text(encoding="utf-8")
    assert "NexusLoop" not in source
    assert "NexusLoopTaskExecutor" not in source
    assert "UnifiedTaskRunner" not in source
    signature = inspect.signature(InteractionIntakeService.__init__)
    assert "nexus_loop" not in signature.parameters


def test_npsc3c_d_task_executor_exposes_host_execution_port() -> None:
    source = _TASK_EXECUTOR_PATH.read_text(encoding="utf-8")
    assert "class HostTaskExecutionExecutor" in source
    assert "HostTaskExecutionPort" in source
    assert "NexusLoop" not in source


def test_npsc3c_d_host_task_routes_through_execution_facade() -> None:
    host_source = _HOST_TASK_PATH.read_text(encoding="utf-8")
    facade_source = _FACADE_PATH.read_text(encoding="utf-8")
    assert "Execution(" in host_source
    assert "StrategyExecutionRouter" in host_source
    assert "ExecutionRuntime" in host_source
    assert "mint_root_execution_identity(" not in host_source
    assert "resolve_root_execution_context(" in facade_source


def test_npsc3c_d_interaction_layer_does_not_own_identity_or_strategy() -> None:
    for path in _INTERACTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "mint_run_id" not in text
        assert "mint_attempt_id" not in text
        assert "mint_execution_id" not in text
        assert "StrategyExecutionRouter" not in text
