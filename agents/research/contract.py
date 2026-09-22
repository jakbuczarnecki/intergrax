# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.errors import AgentImportError
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN

_RESEARCH_AGENT_QUALNAME = "research.research_agent.ResearchAgent"
_SUMMARY_AGENT_QUALNAME = "research.summary_agent.SummaryAgent"


def _agent_qualname(agent_type: type[Tier2Agent]) -> str:
    return f"{agent_type.__module__}.{agent_type.__qualname__}"


def _research_contract() -> AgentContract:
    return AgentContract(
        id="research",
        name="Research Agent",
        description="Prototype agent producing stub research findings.",
        version="0.1.0",
        capabilities=["research.web_search", "research.pipeline"],
        skills=[RESEARCH_LITERATURE_SCAN],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=10,
        validation_rules=["non_empty_summary"],
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version="acp.v1",
    )


def _summary_contract() -> AgentContract:
    return AgentContract(
        id="research-summary",
        name="Research Summary Agent",
        description="Summarizes research findings from prior graph nodes.",
        version="0.1.0",
        capabilities=["research.summarize"],
        skills=[],
        extra_tools=[],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.STAGING,
        owner_team="platform",
        max_steps=5,
        validation_rules=["non_empty_summary"],
        cognitive_pattern=CognitivePattern.REFLEX,
        pattern_version="acp.v1",
    )


def build_agent_contract(agent_type: type[Tier2Agent]) -> AgentContract:
    qualname = _agent_qualname(agent_type)
    if qualname == _RESEARCH_AGENT_QUALNAME:
        return _research_contract()
    if qualname == _SUMMARY_AGENT_QUALNAME:
        return _summary_contract()
    raise AgentImportError(
        f"No declarative AgentContract for agent type {agent_type!r} in research package"
    )
