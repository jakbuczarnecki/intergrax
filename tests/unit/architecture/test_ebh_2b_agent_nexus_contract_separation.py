# © Artur Czarnecki. All rights reserved.

"""EBH-2B — public Agent/UAEP contract separation from Nexus runtime types."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_type_hints

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep_protocol import UAEPAgent, is_uaep_agent
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, TerminalReason
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.validation import ValidationResult

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _module_imports_runtime(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.runtime"
        ):
            hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
    return hits


def test_agent_contract_import_purity() -> None:
    path = _REPO_ROOT / "intergrax/agents/agent_contract.py"
    assert _module_imports_runtime(path) == []


def test_uaep_protocol_import_purity() -> None:
    path = _REPO_ROOT / "intergrax/agents/uaep_protocol.py"
    assert _module_imports_runtime(path) == []


def test_agent_contract_run_uses_canonical_io() -> None:
    hints = get_type_hints(Agent.run)
    assert hints.get("request") is AgentRunRequest
    assert hints.get("return") is AgentRunResult


class _ThirdPartyPluginAgent(Agent):
    """Pluginability proof — imports only public contracts (no intergrax.runtime)."""

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="third_party.plugin",
            name="Third Party",
            description="plugin",
            version="0.0.1",
            capabilities=["demo.echo"],
            risk_level=AgentRiskLevel.LOW,
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        message = request.input if isinstance(request.input, str) else str(request.input)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output=f"echo:{message}",
            run_id=request.correlation_id or "run_plugin",
            trace_id=request.correlation_id or "run_plugin",
            terminal_reason=TerminalReason.GOAL_MET,
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, agent_id="third_party.plugin", score=1.0)


@pytest.mark.asyncio
async def test_third_party_plugin_agent_run() -> None:
    agent = _ThirdPartyPluginAgent()
    request = AgentRunRequest(
        input="hello",
        identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        correlation_id="corr-1",
    )
    result = await agent.run(request)
    assert result.output == "echo:hello"


class _MinimalUaepPlugin(UAEPAgent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="third_party.uaep",
            name="UAEP Plugin",
            description="plugin",
            version="0.0.1",
            capabilities=[],
            risk_level=AgentRiskLevel.LOW,
        )

    def get_steps(self, context: object | None = None) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="main", step_name="main", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = ctx
        return StepOutput(step_id=step.step_id, summary="ok")


def test_minimal_uaep_plugin_protocol() -> None:
    plugin = _MinimalUaepPlugin()
    assert is_uaep_agent(plugin)
    assert plugin.get_steps()[-1].step_id == "main"
