# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel

from intergrax.tools.providers.agent.contracts import (
    AgentGetContractInput,
    AgentGetContractOutput,
    AgentListAgentsInput,
    AgentListAgentsOutput,
    AgentSummaryOutput,
)
from intergrax.tools.registry.runtime_bindings import AgentRegistryBinding
from intergrax.tools.registry.wiring import ToolWiringContext

AGENT_LIST_AGENTS_TOOL_ID = "agent.list_agents"
AGENT_GET_CONTRACT_TOOL_ID = "agent.get_contract"


def _require_agent_registry(ctx: ToolWiringContext) -> AgentRegistryBinding:
    registry = ctx.agent_registry or ctx.extras.get("agent_registry")
    if registry is None:
        raise RuntimeError("agent_registry_not_configured")
    if not isinstance(registry, AgentRegistryBinding):
        raise RuntimeError("agent_registry_invalid_type")
    return registry


def _contract_to_dict(contract: object) -> dict:
    if isinstance(contract, BaseModel):
        return contract.model_dump()
    if hasattr(contract, "model_dump"):
        return contract.model_dump()
    if isinstance(contract, dict):
        return dict(contract)
    raise RuntimeError("agent_contract_not_serializable")


def agent_list_agents(ctx: ToolWiringContext, params: AgentListAgentsInput) -> AgentListAgentsOutput:
    registry = _require_agent_registry(ctx)
    summaries: list[AgentSummaryOutput] = []
    for agent_id in registry.list_agent_ids()[: params.limit]:
        contract = registry.get_agent_contract(agent_id)
        payload = _contract_to_dict(contract)
        summaries.append(
            AgentSummaryOutput(
                agent_id=str(payload.get("id") or agent_id),
                capabilities=[str(item) for item in payload.get("capabilities") or []],
                skill_ids=[str(item) for item in payload.get("skills") or payload.get("skill_ids") or []],
            )
        )
    return AgentListAgentsOutput(agents=summaries, total=len(summaries))


def agent_get_contract(ctx: ToolWiringContext, params: AgentGetContractInput) -> AgentGetContractOutput:
    registry = _require_agent_registry(ctx)
    agent_id = params.agent_id.strip()
    known = set(registry.list_agent_ids())
    if agent_id not in known:
        return AgentGetContractOutput(found=False, agent_id=agent_id)
    contract = registry.get_agent_contract(agent_id)
    return AgentGetContractOutput(
        found=True,
        agent_id=agent_id,
        contract=_contract_to_dict(contract),
    )
