"""FC-1 Part B/C2: a production_mode NexusLoop never runs with persistence silently off.

NT-FC1-13..18, 23 and 25 of T-IGX-043-A (EQOA-A4-037, A4-024, A4-031; DP-24 hybrid): in LAB nothing
changes; in production_mode the openers never read the ambient environment, a missing store needs a
typed acknowledgement, and a context manager without engine or adapter is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.applications._shared.host_orchestration_backend_spec_builder import (
    build_host_orchestration_loop_init_spec_from_environment,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    ENV_RUNTIME_EVENTS_DB,
    ENV_TASK_MEMORY_DB,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.store import (
    RuntimeEventPersistenceNotConfiguredError,
    resolve_runtime_event_persistence,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution import environment_orchestration_materialization as materialization_module
from intergrax.runtime.execution.environment_orchestration_materialization import (
    materialize_host_orchestration_backend,
)
from intergrax.runtime.execution.host_orchestration_loop_init_spec import HostOrchestrationLoopInitSpec
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.human.persistence_errors import HumanDecisionPersistenceNotConfiguredError
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.nexus_loop import NexusLoop, ProductionContextCompositionError
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task_memory.store import (
    TaskMemoryPersistenceNotConfiguredError,
    resolve_task_memory_persistence,
)
from intergrax.runtime.task_memory.stores.memory_task_memory_store import InMemoryTaskMemoryStore

pytestmark = pytest.mark.unit


def _complete_context_manager() -> ContextManager:
    return ContextManager(context_engine=MagicMock(), llm_adapter=MagicMock())


def _strict_loop(**overrides: Any) -> NexusLoop:
    """STRICT loop whose every other composition requirement is satisfied."""
    kwargs: dict[str, Any] = {
        "retry_policy": RetryPolicy(max_retries=0),
        "max_run_retries": 0,
        "disable_execution_continuation": True,
        "context_manager": _complete_context_manager(),
        "runtime_event_store": InMemoryRuntimeEventStore(),
        "task_memory_store": InMemoryTaskMemoryStore(),
    }
    kwargs.update(overrides)
    return NexusLoop(AgentRegistry(), production_mode=True, **kwargs)


def test_strict_loop_refuses_missing_human_decision_store() -> None:
    """NT-FC1-13."""
    with pytest.raises(HumanDecisionPersistenceNotConfiguredError):
        _strict_loop()


def test_strict_loop_accepts_explicit_human_persistence_disable() -> None:
    """NT-FC1-14: the acknowledgement path is explicit and typed."""
    loop = _strict_loop(disable_human_decision_persistence=True)
    assert loop.production_mode is True


def test_strict_loop_refuses_in_memory_human_store() -> None:
    """NT-FC1-15."""
    with pytest.raises(HumanDecisionPersistenceNotConfiguredError):
        _strict_loop(human_decision_store=InMemoryHumanDecisionPersistence())


def test_strict_loop_ignores_ambient_task_memory_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NT-FC1-16."""
    ambient = tmp_path / "ambient_task_memory.db"
    monkeypatch.setenv(ENV_TASK_MEMORY_DB, str(ambient))
    with pytest.raises(TaskMemoryPersistenceNotConfiguredError):
        _strict_loop(task_memory_store=None, disable_human_decision_persistence=True)
    assert not ambient.exists()

    loop = _strict_loop(
        task_memory_store=None,
        disable_task_memory_persistence=True,
        disable_human_decision_persistence=True,
    )
    assert loop.production_mode is True
    assert resolve_task_memory_persistence(production_mode=True, disabled=True) is None
    assert not ambient.exists()


def test_strict_loop_ignores_ambient_runtime_events_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NT-FC1-17."""
    ambient = tmp_path / "ambient_runtime_events.db"
    monkeypatch.setenv(ENV_RUNTIME_EVENTS_DB, str(ambient))
    with pytest.raises(RuntimeEventPersistenceNotConfiguredError):
        _strict_loop(runtime_event_store=None, disable_human_decision_persistence=True)
    assert not ambient.exists()
    assert resolve_runtime_event_persistence(production_mode=True, disabled=True) is None

    persisting_bus = RuntimeEventBus(persistence=InMemoryRuntimeEventStore())
    loop = _strict_loop(
        runtime_event_store=None,
        event_bus=persisting_bus,
        disable_human_decision_persistence=True,
    )
    assert loop.event_bus is persisting_bus
    assert not ambient.exists()


def test_lab_loop_keeps_env_resolution_and_none_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """NT-FC1-18 (control): LAB behavior is unchanged."""
    monkeypatch.delenv(ENV_TASK_MEMORY_DB, raising=False)
    monkeypatch.delenv(ENV_RUNTIME_EVENTS_DB, raising=False)
    loop = NexusLoop(AgentRegistry(), production_mode=False)
    assert loop.production_mode is False
    assert resolve_task_memory_persistence() is None
    assert resolve_runtime_event_persistence() is None


def test_spec_forwards_persistence_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """NT-FC1-23: the four arguments travel Tier-3 builder -> spec -> materialization -> NexusLoop."""
    store = InMemoryHumanDecisionPersistence()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc1.nt23")
    spec = build_host_orchestration_loop_init_spec_from_environment(
        AgentRegistry(),
        env=env,
        human_decision_store=store,
        disable_human_decision_persistence=True,
        disable_task_memory_persistence=True,
        disable_runtime_event_persistence=True,
    )
    assert spec.human_decision_store is store
    assert spec.disable_human_decision_persistence is True
    assert spec.disable_task_memory_persistence is True
    assert spec.disable_runtime_event_persistence is True

    captured: dict[str, object] = {}

    class _RecordingLoop:
        def __init__(self, registry: object, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(materialization_module, "NexusLoop", _RecordingLoop)
    forwarded = HostOrchestrationLoopInitSpec(
        production_mode=False,
        human_decision_store=store,
        disable_human_decision_persistence=True,
        disable_task_memory_persistence=True,
        disable_runtime_event_persistence=True,
    )
    materialize_host_orchestration_backend(AgentRegistry(), forwarded)
    assert captured["human_decision_store"] is store
    assert captured["disable_human_decision_persistence"] is True
    assert captured["disable_task_memory_persistence"] is True
    assert captured["disable_runtime_event_persistence"] is True

    default_spec = HostOrchestrationLoopInitSpec(production_mode=False)
    assert default_spec.human_decision_store is None
    assert default_spec.disable_human_decision_persistence is False


def test_strict_loop_refuses_context_manager_without_engine_or_adapter() -> None:
    """NT-FC1-25 (Part C2): STRICT without a context engine or an LLM adapter is refused; LAB is not."""
    ack = {"disable_human_decision_persistence": True}
    with pytest.raises(ProductionContextCompositionError):
        _strict_loop(context_manager=ContextManager(), **ack)
    with pytest.raises(ProductionContextCompositionError):
        _strict_loop(context_manager=ContextManager(context_engine=MagicMock()), **ack)
    with pytest.raises(ProductionContextCompositionError):
        _strict_loop(context_manager=ContextManager(llm_adapter=MagicMock()), **ack)
    assert NexusLoop(AgentRegistry(), production_mode=False).production_mode is False


def test_strict_environment_through_tier3_builder_is_refused_without_engine_or_adapter() -> None:
    """NT-FC1-25 (harness path): the Tier-3 builder's STRICT spec cannot materialize without them."""
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc1.nt25")
    env.execution_mode = ExecutionMode.STRICT
    env.orchestration_profile.max_parallel_nodes = 1
    env.orchestration_profile.max_inflight_nodes = 1
    spec = build_host_orchestration_loop_init_spec_from_environment(
        AgentRegistry(),
        env=env,
        context_manager=ContextManager(),
        disable_human_decision_persistence=True,
        disable_task_memory_persistence=True,
        disable_runtime_event_persistence=True,
    )
    assert spec.production_mode is True
    with pytest.raises(ProductionContextCompositionError):
        materialize_host_orchestration_backend(AgentRegistry(), spec)
