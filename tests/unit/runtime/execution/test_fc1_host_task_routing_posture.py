"""FC-1 Part A: the host task execution routes through the loop's own router (single posture).

Negative tests NT-FC1-01..04 of T-IGX-043-A (EQOA-A4-044: the second router built without
``production_mode`` skipped the production-only lifecycle, owner and runbook checks).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.registry.agent_assembly_resolver import AgentAssemblyError
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from testing_support.builder import build_task_for_tests, canonical_governed_execution_scope
from testing_support.nexus_host_task_execution import (
    build_certified_internal_test_host_task_execution,
)

pytestmark = pytest.mark.unit


class _CountingAgent:
    """Agent that matches every task and counts how often it is run."""

    def __init__(self, contract: AgentContract) -> None:
        self._contract = contract
        self.runs = 0

    def get_contract(self) -> AgentContract:
        return self._contract

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.runs += 1
        return AgentRunResult(output="ran")

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, score=1.0, reason="stub")


def _contract(**updates: object) -> AgentContract:
    return AgentContract(
        id="demo",
        name="Demo",
        description="demo",
        capabilities=["demo.basic"],
        max_steps=1,
    ).model_copy(update=updates)


def _production_loop(registry: AgentRegistry) -> NexusLoop:
    """STRICT loop with every FC-1 acknowledgement spelled out (no persistence, no HITL tools)."""
    return NexusLoop(
        registry,
        production_mode=True,
        retry_policy=RetryPolicy(max_retries=0),
        max_run_retries=0,
        disable_execution_continuation=True,
        disable_human_decision_persistence=True,
        disable_task_memory_persistence=True,
        disable_runtime_event_persistence=True,
        context_manager=ContextManager(
            context_engine=MagicMock(),
            llm_adapter=MagicMock(),
        ),
    )


async def _execute_without_agent_id(execution: HostTaskExecution, seed: str) -> TaskResult | None:
    """Run a task with an empty ``agent_id``; a refusal is either an exception or a FAILED result."""
    task: Task = build_task_for_tests(seed=seed, message="route me")
    assert not task.agent_id
    with canonical_governed_execution_scope(seed):
        try:
            return await execution.execute(task)
        except Exception:
            return None


@pytest.mark.asyncio
async def test_host_task_rejects_non_production_state_agent_in_production_loop() -> None:
    """NT-FC1-01."""
    registry = AgentRegistry()
    agent = _CountingAgent(_contract(lifecycle_state=AgentLifecycleState.DEVELOPMENT))
    registry.register(agent)
    execution = build_certified_internal_test_host_task_execution(_production_loop(registry))
    result = await _execute_without_agent_id(execution, "fc1-nt01")
    assert result is None or result.state is not TaskState.COMPLETED
    assert agent.runs == 0


def test_host_task_rejects_production_eligible_agent_without_owner_or_runbook() -> None:
    """NT-FC1-02: an ownerless production-eligible agent never reaches the host task router.

    The registry assembly refuses it (owner, contact and runbook are required) before routing, so
    the production-only owner/runbook check of ``evaluate_agent_routing`` is the second line; the
    posture difference of the host path is exercised by NT-FC1-01 and pinned by NT-FC1-04.
    """
    registry = AgentRegistry()
    with pytest.raises(AgentAssemblyError, match="owner_team"):
        registry.register(_CountingAgent(_contract(production_eligible=True)))
    assert registry.list_routable_agent_ids(production_mode=True) == []


@pytest.mark.asyncio
async def test_host_task_routes_production_agent_with_owner_and_runbook() -> None:
    """NT-FC1-03 (control): complete production metadata routes."""
    registry = AgentRegistry()
    agent = _CountingAgent(
        _contract(
            production_eligible=True,
            owner_team="platform",
            owner_contact="owner@intergrax",
            runbook_ref="docs/project/architecture/intergrax_runtime_architecture.md",
        ),
    )
    registry.register(agent)
    execution = build_certified_internal_test_host_task_execution(_production_loop(registry))
    with canonical_governed_execution_scope("fc1-nt03"):
        selected = execution._agent_router.route(  # noqa: SLF001
            build_task_for_tests(seed="fc1-nt03", message="route me"),
        )
    assert selected is agent


def test_host_task_execution_uses_the_loops_router() -> None:
    """NT-FC1-04: one routing authority, same posture by construction."""
    for production_mode in (False, True):
        registry = AgentRegistry()
        loop = NexusLoop(registry, production_mode=False) if not production_mode else _production_loop(registry)
        execution = build_certified_internal_test_host_task_execution(loop)
        assert execution._agent_router is loop.agent_router  # noqa: SLF001
        assert loop.production_mode is production_mode
