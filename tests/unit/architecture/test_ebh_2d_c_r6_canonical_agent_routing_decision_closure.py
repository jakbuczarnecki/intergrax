# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C-R6 — canonical agent routing decision closure."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.capability_routing import select_best_capability_match
from intergrax.runtime.task.agent_capability_intake import task_envelope_for_agent_capability_match
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_ROOT = _REPO_ROOT / "intergrax/runtime/registry"
_CAPABILITY_ROUTING = _REGISTRY_ROOT / "capability_routing.py"
_AGENT_REGISTRY = _REGISTRY_ROOT / "agent_registry.py"
_AGENT_ROUTER = _REPO_ROOT / "intergrax/runtime/nexus/agent_router.py"
_NEXUS_ROOT = _REPO_ROOT / "intergrax/runtime/nexus"
_CANONICAL_SCORING_OWNER = _CAPABILITY_ROUTING.relative_to(_REPO_ROOT).as_posix()
_PLANNER_EXEMPT = "intergrax/runtime/nexus/planning/task_planner.py"
_SCORING_SCAN_ROOTS = (_REGISTRY_ROOT, _NEXUS_ROOT)


def _source_calls_can_handle(path: Path) -> bool:
    return ".can_handle(" in path.read_text(encoding="utf-8")


def _production_can_handle_inventory() -> list[tuple[str, bool]]:
    rows: list[tuple[str, bool]] = []
    for root in _SCORING_SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if _source_calls_can_handle(path):
                allowed = rel == _CANONICAL_SCORING_OWNER or rel == _PLANNER_EXEMPT
                rows.append((rel, allowed))
    return rows


class _CallCountingRoutableAgent:
    def __init__(self, agent_id: str, capability: str, scores: list[float]) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._scores = list(scores)
        self.can_handle_calls = 0

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="call-count routable agent",
            capabilities=[self._capability],
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        self.can_handle_calls += 1
        score = self._scores.pop(0) if self._scores else 0.0
        return CapabilityMatchResult(
            matched=True,
            agent_id=self._agent_id,
            matched_capabilities=[self._capability],
            score=score,
        )


class _UnmatchedRoutableAgent:
    def __init__(self, agent_id: str, capability: str) -> None:
        self._agent_id = agent_id
        self._capability = capability

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="unmatched routable agent",
            capabilities=[self._capability],
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        _ = task
        return CapabilityMatchResult(matched=False, rationale="no match")


def test_agent_registry_has_no_can_handle_scoring_loop() -> None:
    assert ".can_handle(" not in _AGENT_REGISTRY.read_text(encoding="utf-8")


def test_agent_router_has_no_can_handle_scoring_loop() -> None:
    assert ".can_handle(" not in _AGENT_ROUTER.read_text(encoding="utf-8")


def test_capability_routing_is_canonical_scoring_owner() -> None:
    source = _CAPABILITY_ROUTING.read_text(encoding="utf-8")
    assert "def evaluate_routable_candidates" in source
    assert ".can_handle(" in source


def test_registry_modules_except_canonical_owner_forbid_can_handle() -> None:
    for path in sorted(_REGISTRY_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == _CANONICAL_SCORING_OWNER:
            continue
        assert ".can_handle(" not in path.read_text(encoding="utf-8"), rel


def test_nexus_except_planner_forbid_can_handle() -> None:
    for path in sorted(_NEXUS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == _PLANNER_EXEMPT:
            continue
        assert ".can_handle(" not in path.read_text(encoding="utf-8"), rel


def test_production_can_handle_inventory_classified() -> None:
    for rel, allowed in _production_can_handle_inventory():
        assert allowed, rel


def test_find_best_match_no_matched_returns_none() -> None:
    registry = AgentRegistry()
    registry.register(_UnmatchedRoutableAgent("u1", "cap.a"))
    envelope = TaskEnvelope(tenant_id="t", user_id="u", metadata={})
    assert registry.find_best_match(envelope) is None


def test_single_can_handle_per_candidate_on_capability_route() -> None:
    capability = "route.count"
    agents = [
        _CallCountingRoutableAgent("a1", capability, [1.0]),
        _CallCountingRoutableAgent("a2", capability, [5.0]),
        _CallCountingRoutableAgent("a3", capability, [3.0]),
    ]
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="route",
        context=TaskContext(capability=capability),
    )
    route = select_best_capability_match(registry, task, capability)
    assert route.selected is not None
    assert route.selected.get_contract().id == "a2"
    for agent in agents:
        assert agent.can_handle_calls == 1


def test_router_does_not_reinvoke_can_handle_for_telemetry() -> None:
    capability = "route.stateful"
    agent = _CallCountingRoutableAgent("stateful", capability, [9.0, 1.0])
    registry = AgentRegistry()
    registry.register(agent)
    router = AgentRouter(registry)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="route",
        context=TaskContext(capability=capability),
    )
    selected = router.route(task)
    assert selected.get_contract().id == "stateful"
    assert agent.can_handle_calls == 1


def test_capability_route_result_preserves_first_can_handle_score() -> None:
    capability = "route.stateful"
    agent = _CallCountingRoutableAgent("stateful", capability, [9.0, 1.0])
    registry = AgentRegistry()
    registry.register(agent)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="route",
        context=TaskContext(capability=capability),
    )
    route = select_best_capability_match(registry, task, capability)
    assert route.selected_match is not None
    assert route.selected_match.score == 9.0
    assert agent.can_handle_calls == 1


def test_agent_registry_find_best_match_delegates_without_local_scoring_loop() -> None:
    tree = ast.parse(_AGENT_REGISTRY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "find_best_match":
            body_source = ast.get_source_segment(
                _AGENT_REGISTRY.read_text(encoding="utf-8"),
                node,
            )
            assert body_source is not None
            assert "select_best_matched_routable_agent" in body_source
            assert ".can_handle(" not in body_source
            return
    raise AssertionError("find_best_match not found")
