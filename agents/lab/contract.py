# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.errors import AgentImportError
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.tier2_agent import Tier2Agent

_RESEARCH_MOCK_QUALNAME = "lab.mock_agents.ResearchMockAgent"
_DOCUMENT_MOCK_QUALNAME = "lab.mock_agents.DocumentMockAgent"
_VALIDATOR_MOCK_QUALNAME = "lab.mock_agents.ValidatorMockAgent"
_COMPOSER_MOCK_QUALNAME = "lab.mock_agents.ComposerMockAgent"


def _agent_qualname(agent_type: type[Tier2Agent]) -> str:
    return f"{agent_type.__module__}.{agent_type.__qualname__}"


def _mock_contract(
    *,
    agent_id: str,
    name: str,
    capability: str,
) -> AgentContract:
    return AgentContract(
        id=agent_id,
        name=name,
        description=f"Runtime validation mock ({agent_id}).",
        version="0.1.0",
        capabilities=[capability],
        skills=[],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.DEVELOPMENT,
        owner_team="platform",
        max_steps=5,
    )


def build_agent_contract(agent_type: type[Tier2Agent]) -> AgentContract:
    qualname = _agent_qualname(agent_type)
    if qualname == _RESEARCH_MOCK_QUALNAME:
        return _mock_contract(
            agent_id="research_mock",
            name="Research Mock Agent",
            capability="lab.research_mock",
        )
    if qualname == _DOCUMENT_MOCK_QUALNAME:
        return _mock_contract(
            agent_id="document_mock",
            name="Document Mock Agent",
            capability="lab.document_mock",
        )
    if qualname == _VALIDATOR_MOCK_QUALNAME:
        return _mock_contract(
            agent_id="validator_mock",
            name="Validator Mock Agent",
            capability="lab.validator_mock",
        )
    if qualname == _COMPOSER_MOCK_QUALNAME:
        return _mock_contract(
            agent_id="composer_mock",
            name="Composer Mock Agent",
            capability="lab.composer_mock",
        )
    raise AgentImportError(
        f"No declarative AgentContract for agent type {agent_type!r} in lab package"
    )
