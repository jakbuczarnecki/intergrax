# © Artur Czarnecki. All rights reserved.

"""EBH-2D-C-R4 — canonical runtime agent port alignment (registry, read, router)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.routable_tier2_agent import (
    AgentRoutingContractError,
    RoutableTier2Agent,
)
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.registry.agent_registry_read_view import freeze_agent_registry
from intergrax.runtime.registry.capability_routing import select_best_capability_match
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_READ_PORT = _REPO_ROOT / "intergrax/runtime/registry/agent_registry_read.py"
_REGISTRY = _REPO_ROOT / "intergrax/runtime/registry/agent_registry.py"
_ROUTER = _REPO_ROOT / "intergrax/runtime/nexus/agent_router.py"


def _read_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ExternalExecutionAgent:
    """Structural Tier-2 agent — explicit-id execution only."""

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="external_exec_only",
            name="Exec",
            description="R4 non-routable structural proof",
            capabilities=["external.exec"],
        )


class ExternalRoutableAgent:
    """Structural RoutableTier2Agent without framework ``Agent`` inheritance."""

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("proof-only")

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="external_routable",
            name="Routable",
            description="R4 routing structural proof",
            capabilities=["external.route"],
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        _ = task
        return CapabilityMatchResult(
            matched=True,
            agent_id="external_routable",
            matched_capabilities=["external.route"],
            score=10.0,
            rationale="structural proof",
        )


class ExternalExecFactory:
    def __call__(
        self,
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> Tier2Agent:
        _ = ctx, binding
        return ExternalExecutionAgent()


class ExternalRoutableFactory:
    def __call__(
        self,
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> Tier2Agent:
        _ = ctx, binding
        return ExternalRoutableAgent()


def test_agent_registry_read_has_no_concrete_agent_import() -> None:
    imports = _read_imports(_READ_PORT)
    assert "intergrax.agents.agent_contract" not in imports


def test_agent_registry_storage_uses_tier2_agent_not_concrete_agent() -> None:
    source = _REGISTRY.read_text(encoding="utf-8")
    assert "Dict[str, Agent]" not in source
    assert "Dict[str, Tier2Agent]" in source
    imports = _read_imports(_REGISTRY)
    assert "intergrax.agents.agent_contract" not in imports


def test_agent_router_has_no_concrete_agent_import() -> None:
    imports = _read_imports(_ROUTER)
    assert "intergrax.agents.agent_contract" not in imports


def test_external_structural_agent_registry_read_explicit_route() -> None:
    registry = AgentRegistry()
    agent = ExternalRoutableAgent()
    registry.register(agent)
    read_surface = freeze_agent_registry(registry)
    assert isinstance(read_surface, AgentRegistryRead)
    assert read_surface.get("external_routable") is agent
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="explicit",
        agent_id="external_routable",
        context=TaskContext(),
    )
    routed = AgentRouter(read_surface).route(task)
    assert type(routed) is ExternalRoutableAgent
    assert isinstance(routed, RoutableTier2Agent)


def test_external_structural_capability_routing() -> None:
    registry = AgentRegistry()
    registry.register(ExternalRoutableAgent())
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="capability",
        context=TaskContext(capability="external.route"),
    )
    route = select_best_capability_match(registry, task, "external.route")
    assert route.selected is not None
    assert route.selected.get_contract().id == "external_routable"
    selected = AgentRouter(registry).route(task)
    assert selected.get_contract().id == "external_routable"


def test_capability_routing_fail_closed_without_routable_contract() -> None:
    registry = AgentRegistry()
    registry.register(ExternalExecutionAgent())
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="capability",
        context=TaskContext(capability="external.exec"),
    )
    with pytest.raises(AgentRoutingContractError, match="RoutableTier2Agent"):
        select_best_capability_match(registry, task, "external.exec")


def test_factory_materialized_external_agents_use_canonical_read_port() -> None:
    manifest = ApplicationManifest.lab(
        app_id="r4_flow",
        name="R4",
        agents=[
            AgentBinding(contract_id="external_routable", factory=ExternalRoutableFactory()),
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_manifest_development_registry(manifest, ctx)
    assert registry.has("external_routable")
    read_agent = registry.get("external_routable")
    assert isinstance(read_agent, Tier2Agent)
    assert isinstance(read_agent, RoutableTier2Agent)
