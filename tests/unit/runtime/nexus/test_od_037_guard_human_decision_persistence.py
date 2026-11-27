# © Artur Czarnecki. All rights reserved.

"""OD-037-GUARD (EQOA-A4-037): lock NexusLoop production_mode human-decision persistence guard.

Locks the four existing guard and return-without-writing contracts on current HEAD without
modifying ``persist_human_decision`` (``OD-037-GUARD=A``, ``OD-037-PERSIST`` rejected).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.human.persistence_errors import HumanDecisionPersistenceNotConfiguredError
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration import human_response as human_response_module
from intergrax.runtime.nexus.orchestration.human_response import persist_human_decision
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task_memory.stores.memory_task_memory_store import InMemoryTaskMemoryStore
from testing_support.builder import build_task_for_tests

pytestmark = pytest.mark.unit


def _strict_loop(**overrides: Any) -> NexusLoop:
    kwargs: dict[str, Any] = {
        "retry_policy": RetryPolicy(max_retries=0),
        "max_run_retries": 0,
        "disable_execution_continuation": True,
        "context_manager": ContextManager(context_engine=MagicMock(), llm_adapter=MagicMock()),
        "runtime_event_store": InMemoryRuntimeEventStore(),
        "task_memory_store": InMemoryTaskMemoryStore(),
    }
    kwargs.update(overrides)
    return NexusLoop(AgentRegistry(), production_mode=True, **kwargs)


def _sample_task() -> Task:
    return build_task_for_tests(
        seed="od-037-guard-human-decision",
        tenant_id="tenant-a",
        user_id="user-a",
        agent_id="agent-a",
        message="approve deployment",
    )


def test_od_037_guard_1_production_mode_missing_store_and_ack_false_raises() -> None:
    """Case 1: production_mode=True, store=None, disable_human_decision_persistence=False raises."""
    with pytest.raises(
        HumanDecisionPersistenceNotConfiguredError,
        match="disable_human_decision_persistence",
    ):
        _strict_loop(
            human_decision_store=None,
            disable_human_decision_persistence=False,
        )


def test_od_037_guard_2_production_mode_in_memory_store_raises() -> None:
    """Case 2: production_mode=True with InMemoryHumanDecisionPersistence raises."""
    with pytest.raises(
        HumanDecisionPersistenceNotConfiguredError,
        match="in-memory",
    ):
        _strict_loop(
            human_decision_store=InMemoryHumanDecisionPersistence(),
            disable_human_decision_persistence=False,
        )


def test_od_037_guard_3_production_mode_ack_true_succeeds_and_returns_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 3: production_mode=True, disable_human_decision_persistence=True returns cleanly."""
    loop = _strict_loop(
        human_decision_store=None,
        disable_human_decision_persistence=True,
    )
    assert loop.production_mode is True
    assert loop._human_store is None

    def _forbid_record_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("persist_human_decision must not build or write a record when store is None")

    monkeypatch.setattr(human_response_module, "build_human_decision_record", _forbid_record_build)

    task = _sample_task()
    result = persist_human_decision(
        task,
        HumanResponseVerdict.APPROVE,
        human_store=loop._human_store,
    )
    assert result is None
    assert loop._persist_human_decision(task, HumanResponseVerdict.APPROVE) is None


def test_od_037_guard_4_lab_mode_missing_store_returns_cleanly_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case 4: production_mode=False (LAB), store=None returns cleanly without writing."""
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=False,
        human_decision_store=None,
    )
    assert loop.production_mode is False
    assert loop._human_store is None

    def _forbid_record_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("persist_human_decision must not build or write a record when store is None")

    monkeypatch.setattr(human_response_module, "build_human_decision_record", _forbid_record_build)

    task = _sample_task()
    result = persist_human_decision(
        task,
        HumanResponseVerdict.APPROVE,
        human_store=loop._human_store,
    )
    assert result is None
    assert loop._persist_human_decision(task, HumanResponseVerdict.APPROVE) is None
