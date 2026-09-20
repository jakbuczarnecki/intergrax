# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C-R5 — runtime AgentRegistryRead consumer contract closure."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.handoff.coordinator import HandoffCoordinator, HandoffValidationError
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.capability_routing import select_best_routable_agent
from intergrax.runtime.task.agent_capability_intake import task_envelope_from_task_context
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HANDOFF = _REPO_ROOT / "intergrax/runtime/nexus/handoff/coordinator.py"
_RETRY = _REPO_ROOT / "intergrax/runtime/nexus/retry/retry_engine.py"
_CAPABILITY_ROUTING = _REPO_ROOT / "intergrax/runtime/registry/capability_routing.py"
_NEXUS_ROOT = _REPO_ROOT / "intergrax/runtime/nexus"
_CONSUMER_MODULES = (
    "intergrax/runtime/nexus/agent_router.py",
    "intergrax/runtime/nexus/task_classifier.py",
    "intergrax/runtime/nexus/planning/task_planner.py",
    "intergrax/runtime/nexus/handoff/coordinator.py",
    "intergrax/runtime/nexus/retry/retry_engine.py",
    "intergrax/runtime/nexus/agents/agent_engine.py",
)


def _read_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _source_calls_can_handle(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return ".can_handle(" in source


class ExternalExecutionOnlyAgent:
    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="exec_only_handoff",
            name="Exec",
            description="execution-only structural agent",
            capabilities=["handoff.exec"],
        )


class ExternalRoutableHandoffAgent:
    def __init__(self, *, agent_id: str, capability: str, score: float) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._score = score

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="routable structural handoff agent",
            capabilities=[self._capability],
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        _ = task
        return CapabilityMatchResult(
            matched=True,
            agent_id=self._agent_id,
            matched_capabilities=[self._capability],
            score=self._score,
        )


class ExternalRetryAgent:
    def __init__(self, *, agent_id: str, capability: str) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._attempts = 0

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="retry structural agent",
            capabilities=[self._capability],
        )


def test_handoff_coordinator_uses_canonical_routing_primitive() -> None:
    source = _HANDOFF.read_text(encoding="utf-8")
    assert "select_best_routable_agent" in source
    assert "_best_capability_match" not in source
    assert ".can_handle(" not in source


def test_handoff_coordinator_has_no_duplicate_scoring_authority() -> None:
    for rel in _CONSUMER_MODULES:
        path = _REPO_ROOT / rel
        if path.name == "coordinator.py":
            continue
        if path.name == "agent_router.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "_best_capability_match" not in source, rel


def test_retry_engine_has_no_concrete_agent_import() -> None:
    imports = _read_imports(_RETRY)
    assert "intergrax.agents.agent_contract" not in imports
    source = _RETRY.read_text(encoding="utf-8")
    assert "from intergrax.agents.agent_contract import Agent" not in source
    assert "initial_agent: Agent" not in source
    assert "Callable[[Agent," not in source


def test_capability_routing_owns_can_handle_scoring_loop() -> None:
    routing_source = _CAPABILITY_ROUTING.read_text(encoding="utf-8")
    assert "def select_best_routable_agent" in routing_source
    assert ".can_handle(" in routing_source
    for path in sorted(_NEXUS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.endswith("agent_router.py"):
            continue
        if _source_calls_can_handle(path):
            assert rel.endswith("planning/task_planner.py"), rel


def test_handoff_non_routable_metadata_match_fail_closed() -> None:
    registry = AgentRegistry()
    registry.register(ExternalExecutionOnlyAgent())
    coordinator = HandoffCoordinator(registry)
    handoff = AgentHandoff(
        from_agent_id="exec_only_handoff",
        to_capability="handoff.exec",
        reason="delegate",
    )
    result = coordinator.validate(handoff, from_agent_id="exec_only_handoff")
    assert result.valid is False
    assert any("RoutableTier2Agent" in err for err in result.errors)


def test_handoff_structural_routable_canonical_higher_score_wins() -> None:
    registry = AgentRegistry()
    registry.register(
        ExternalRoutableHandoffAgent(
            agent_id="handoff_low",
            capability="handoff.route",
            score=1.0,
        )
    )
    registry.register(
        ExternalRoutableHandoffAgent(
            agent_id="handoff_high",
            capability="handoff.route",
            score=9.0,
        )
    )
    coordinator = HandoffCoordinator(registry)
    handoff = AgentHandoff(
        from_agent_id="handoff_low",
        to_capability="handoff.route",
        reason="delegate",
    )
    result = coordinator.validate(handoff, from_agent_id="handoff_low")
    assert result.valid is True
    assert result.resolved_agent_id == "handoff_high"


def test_handoff_no_capability_match_preserves_validation_semantics() -> None:
    registry = AgentRegistry()
    coordinator = HandoffCoordinator(registry)
    handoff = AgentHandoff(
        from_agent_id="missing",
        to_capability="handoff.missing",
        reason="delegate",
    )
    with pytest.raises(HandoffValidationError, match="no agent registered"):
        coordinator.resolve_target_agent_id(handoff)


@pytest.mark.asyncio
async def test_retry_engine_structural_tier2_agent_callbacks() -> None:
    registry = AgentRegistry()
    primary = ExternalRetryAgent(agent_id="retry_a", capability="retry.cap")
    alternate = ExternalRetryAgent(agent_id="retry_b", capability="retry.cap")
    registry.register(primary)
    registry.register(alternate)
    engine = RetryEngine(registry, policy=RetryPolicy(max_retries=1))
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="retry",
        context=TaskContext(capability="retry.cap"),
    )
    seen: list[str] = []

    async def execute_fn(agent: Tier2Agent):
        from intergrax.contracts.agent_execution_result import (
            AgentExecutionResult,
            AgentExecutionStatus,
        )

        seen.append(agent.get_contract().id)
        return AgentExecutionResult(
            agent_id=agent.get_contract().id,
            run_id="run_1",
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )

    def validate_fn(_execution, agent: Tier2Agent) -> ValidationResult:
        if agent.get_contract().id == "retry_a":
            return ValidationResult(valid=False, errors=["fail once"])
        return ValidationResult(valid=True)

    execution, records, validation = await engine.execute_with_retry(
        task,
        primary,
        execute_fn,
        validate_fn=validate_fn,
    )
    assert validation.valid is True
    assert execution.agent_id == "retry_b"
    assert seen == ["retry_a", "retry_b"]
    assert len(records) == 1
    assert records[0].alternate_agent_id == "retry_b"


def test_select_best_routable_agent_envelope_only_primitive() -> None:
    registry = AgentRegistry()
    registry.register(
        ExternalRoutableHandoffAgent(
            agent_id="env_agent",
            capability="env.cap",
            score=3.0,
        )
    )
    matches = registry.find_by_capability("env.cap")
    envelope = task_envelope_from_task_context(TaskContext(capability="env.cap"))
    route = select_best_routable_agent(
        capability="env.cap",
        envelope=envelope,
        candidates=matches,
    )
    assert route.selected is not None
    assert route.selected.get_contract().id == "env_agent"
