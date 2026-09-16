# © Artur Czarnecki. All rights reserved.

"""OBS-EVIDENCE-PERSISTENCE-CONTRACT-CLEANUP — ``EvidencePersistencePort`` contract purity gates."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from intergrax.contracts.execution_evidence.persistence_port import (
    EvidencePersistencePort,
)
from intergrax.contracts.task_runtime_event_runs import TaskRuntimeEventRuns

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENCE_PORT_PATH = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "execution_evidence"
    / "persistence_port.py"
)

_CONTRACT_OWNED_PUBLIC_SYMBOLS: dict[str, str] = {
    "RuntimeEvent": "intergrax.contracts.runtime_event",
    "PositionedRuntimeEvent": "intergrax.contracts.positioned_runtime_event",
    "ExecutionEventPosition": "intergrax.contracts.execution_event_position",
    "AsOfBoundary": "intergrax.contracts.execution_event_position",
    "TaskRuntimeEventRuns": "intergrax.contracts.task_runtime_event_runs",
}


def _collect_runtime_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime"):
                hits.append(node.module)
    return hits


def test_evidence_persistence_port_module_has_no_runtime_imports() -> None:
    assert _collect_runtime_imports(_PERSISTENCE_PORT_PATH) == []


def test_evidence_persistence_port_public_type_graph_is_contract_owned() -> None:
    importlib.import_module("intergrax.contracts.execution_evidence.persistence_port")
    hints = inspect.get_annotations(EvidencePersistencePort, eval_str=True)
    for method in EvidencePersistencePort.__dict__.values():
        if not callable(method):
            continue
        hints.update(inspect.get_annotations(method, eval_str=True))
    for name, expected_module in _CONTRACT_OWNED_PUBLIC_SYMBOLS.items():
        for hint in hints.values():
            hint_name = getattr(hint, "__name__", None)
            if hint_name != name:
                continue
            assert hint.__module__ == expected_module, (
                f"{name} expected in {expected_module}, got {hint.__module__}"
            )


def test_list_positioned_for_task_grouped_by_run_returns_contract_task_runtime_event_runs() -> (
    None
):
    return_hint = (
        EvidencePersistencePort.list_positioned_for_task_grouped_by_run.__annotations__[
            "return"
        ]
    )
    assert return_hint in (TaskRuntimeEventRuns, "TaskRuntimeEventRuns")
    assert (
        TaskRuntimeEventRuns.__module__ == "intergrax.contracts.task_runtime_event_runs"
    )
    port_source = _PERSISTENCE_PORT_PATH.read_text(encoding="utf-8")
    assert (
        "from intergrax.contracts.task_runtime_event_runs import TaskRuntimeEventRuns"
        in port_source
    )
    assert "intergrax.runtime" not in port_source


class _CustomEvidencePersistence(EvidencePersistencePort):
    """Pluginability proof — custom provider returns contract-owned grouped runs."""

    def append(self, event, *, tenant_id: str):
        raise NotImplementedError

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through=None,
        after=None,
    ):
        raise NotImplementedError

    def list_for_task(self, task_id: str, *, tenant_id: str, limit: int = 1000):
        raise NotImplementedError

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return TaskRuntimeEventRuns(runs=())

    def get_by_event_id(self, *, tenant_id: str, event_id):
        raise NotImplementedError

    def list_positioned_through(self, boundary, *, tenant_id: str, limit: int = 1000):
        raise NotImplementedError


def test_custom_evidence_persistence_port_implementation_is_valid() -> None:
    port = _CustomEvidencePersistence()
    assert isinstance(port, EvidencePersistencePort)
    grouped = port.list_positioned_for_task_grouped_by_run("task", tenant_id="tenant")
    assert isinstance(grouped, TaskRuntimeEventRuns)
