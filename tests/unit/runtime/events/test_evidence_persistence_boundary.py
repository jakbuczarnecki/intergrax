# © Artur Czarnecki. All rights reserved.

"""Enterprise execution evidence persistence boundary (port + adapter)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.events.evidence_persistence_adapter import (
    RuntimeEventPersistenceEvidenceAdapter,
    as_evidence_persistence_port,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVENT_BUS_PATH = _REPO_ROOT / "intergrax/runtime/events/event_bus.py"
_EXECUTION_ROOT = _REPO_ROOT / "intergrax/runtime/execution"


def test_runtime_event_bus_uses_evidence_persistence_port_only() -> None:
    source = _EVENT_BUS_PATH.read_text(encoding="utf-8")
    assert "RuntimeEventPersistence" not in source
    assert "EvidencePersistencePort" in source


def test_execution_engine_has_no_runtime_event_persistence_dependency() -> None:
    violations: list[str] = []
    for path in _EXECUTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "intergrax.runtime.events.persistence_contract":
                continue
            for alias in node.names:
                if alias.name == "RuntimeEventPersistence":
                    violations.append(f"{rel}:{node.lineno}")
    assert violations == []


def test_adapter_is_evidence_persistence_port() -> None:
    inner = InMemoryRuntimeEventStore()
    adapter = RuntimeEventPersistenceEvidenceAdapter(inner)
    assert isinstance(adapter, EvidencePersistencePort)


def test_adapter_preserves_execution_reconstruction() -> None:
    inner = InMemoryRuntimeEventStore()
    adapter = RuntimeEventPersistenceEvidenceAdapter(inner)
    tenant_id = "tenant-boundary"
    task_id = mint_task_id()
    run_id = mint_run_id()
    event = sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id)
    inner.append(event, tenant_id=tenant_id)

    causal = InMemoryCausalEvidencePersistence()
    direct = ExecutionReconstructor(inner, causal).reconstruct_execution(
        tenant_id,
        task_id,
        run_id,
    )
    via_port = ExecutionReconstructor(adapter, causal).reconstruct_execution(
        tenant_id,
        task_id,
        run_id,
    )
    assert via_port == direct


def test_event_bus_wraps_legacy_runtime_event_persistence() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=False)
    port = bus.persistence
    assert port is not None
    assert isinstance(port, RuntimeEventPersistenceEvidenceAdapter)
    assert port.inner is store


def test_as_evidence_persistence_port_idempotent() -> None:
    inner = InMemoryRuntimeEventStore()
    first = as_evidence_persistence_port(inner)
    second = as_evidence_persistence_port(first)
    assert first is second
