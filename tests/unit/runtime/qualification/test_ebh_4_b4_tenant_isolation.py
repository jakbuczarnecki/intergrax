# © Artur Czarnecki. All rights reserved.

"""EBH-4-R1-R3-B4 adversarial tenant isolation (local evidence)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.memory.memory_vector_namespace import resolve_memory_index_collection
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest
from intergrax.runtime.governance.in_memory_metrics_store import InMemoryMetricsStore
from intergrax.runtime.interactions.actor_resolution import resolve_actor_from_envelope, resolve_actor_from_task
from intergrax.runtime.nexus.agents.runtime_request_bridge import runtime_request_to_agent_run
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.bootstrap import bootstrap_runtime_plugins
from intergrax.runtime.plugins.default_plugins import default_lab_plugins
from intergrax.runtime.task.task import Task
from testing_support.builder import build_runtime_request_for_tests, canonical_task_id_for_tests
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.contracts.execution_identity import mint_run_id

pytestmark = pytest.mark.unit


class _BridgeContract:
    id = "bridge-agent"
    risk_level = AgentRiskLevel.LOW


def test_b4_a_direct_task_empty_tenant_rejected() -> None:
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        Task(tenant_id="", user_id="u1", message="x")
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        Task(tenant_id="   ", user_id="u1", message="x")


def test_b4_b_task_a_maps_to_actor_a() -> None:
    task = Task(tenant_id="tenant-a", user_id="u1", message="go")
    actor = resolve_actor_from_task(task)
    assert actor.tenant_id == "tenant-a"


def test_b4_b_envelope_actor_preserves_tenant() -> None:
    envelope = TaskEnvelope(tenant_id="tenant-a", user_id="u1", message="go")
    actor = resolve_actor_from_envelope(envelope)
    assert actor.tenant_id == "tenant-a"


def test_b4_c_tenantless_task_completed_skips_trace_and_metrics() -> None:
    trace_store = MagicMock()
    metrics_store = InMemoryMetricsStore()
    bus = RuntimeEventBus(record_history=False)
    bootstrap_runtime_plugins(
        default_lab_plugins(trace_store=trace_store, metrics_store=metrics_store),
        event_bus=bus,
        hook_registry=HookRegistry(),
    )
    identity = runtime_event_test_identity(
        task_id=canonical_task_id_for_tests("metrics-tenantless"),
    )
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        tenant_id=None,
        **identity,
    )
    bus.record(event)
    trace_store.read_run.assert_not_called()
    assert len(metrics_store._data) == 0


def test_b4_c_task_completed_tenant_a_reads_trace_for_a_only() -> None:
    trace_store = MagicMock()
    trace_store.read_run.return_value = MagicMock()
    metrics_store = InMemoryMetricsStore()
    bus = RuntimeEventBus(record_history=False)
    bootstrap_runtime_plugins(
        default_lab_plugins(trace_store=trace_store, metrics_store=metrics_store),
        event_bus=bus,
        hook_registry=HookRegistry(),
    )
    identity = runtime_event_test_identity(
        task_id=canonical_task_id_for_tests("metrics-tenant-a"),
    )
    run_id = str(identity["run_id"])
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        tenant_id="tenant-a",
        agent_id="agent-1",
        **identity,
    )
    bus.record(event)
    trace_store.read_run.assert_called_once_with(run_id, "tenant-a")


def test_b4_d_runtime_request_metadata_tenant_b_rejected() -> None:
    req = build_runtime_request_for_tests(
        seed="meta-override",
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-b"},
    )
    with pytest.raises(ValueError, match="metadata tenant_id cannot override"):
        req.to_envelope()


def test_b4_bridge_metadata_tenant_b_rejected_without_canonical_identity() -> None:
    req = build_runtime_request_for_tests(
        seed="bridge-meta",
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-b"},
    )
    with pytest.raises(ValueError, match="metadata tenant_id cannot override"):
        runtime_request_to_agent_run(req, contract=_BridgeContract())


def test_b4_bridge_missing_tenant_rejected() -> None:
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        task_id=canonical_task_id_for_tests("bridge-missing-tenant"),
        run_id=mint_run_id(),
        tenant_id=None,
    )
    with pytest.raises(ValueError, match="tenant_id is required"):
        runtime_request_to_agent_run(req, contract=_BridgeContract())


def test_b4_memory_namespace_empty_tenant_rejected() -> None:
    with pytest.raises(ValueError, match="tenant_id or vector_index_namespace"):
        resolve_memory_index_collection(
            vector_index_namespace=None,
            tenant_id="",
            domain="ltm",
        )


def test_b4_memory_namespace_explicit_override_allowed() -> None:
    assert (
        resolve_memory_index_collection(
            vector_index_namespace="shared-ns",
            tenant_id="",
            domain="ltm",
        )
        == "shared-ns:ltm"
    )


def test_b4_delegation_actor_tenant_preserved() -> None:
    parent = ActorIdentity(
        kind=ActorKind.USER,
        actor_id="u1",
        tenant_id="tenant-a",
        permission_scopes=("read",),
    )
    assert parent.tenant_id == "tenant-a"


def test_b4_task_envelope_round_trip_tenant() -> None:
    task = Task(tenant_id="tenant-z", user_id="u1", message="hi")
    assert task.to_envelope().tenant_id == "tenant-z"
    restored = Task.from_envelope(task.to_envelope())
    assert restored.tenant_id == "tenant-z"


def test_b4_runtime_request_from_envelope_tenant_chain() -> None:
    envelope = TaskEnvelope(tenant_id="tenant-a", user_id="u1", message="go")
    task_id = canonical_task_id_for_tests("rr-chain")
    run_id = mint_run_id()
    req = RuntimeRequest.from_envelope(envelope, task_id=task_id, run_id=run_id)
    assert req.tenant_id == "tenant-a"
    assert req.to_envelope().tenant_id == "tenant-a"
