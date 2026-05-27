# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult


class AgentRegistry:
    """
    Central registry for Tier-2 agents (canonical architecture §15).

    Nexus and AgentEngine use this for discovery and selection.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}
        self._contracts: Dict[str, AgentContract] = {}

    def register(self, agent: Agent, *, contract: Optional[AgentContract] = None) -> None:
        meta = contract or agent.get_contract()
        if meta.id in self._agents:
            raise ValueError(f"Agent already registered: {meta.id}")
        self._agents[meta.id] = agent
        self._contracts[meta.id] = meta

    def get(self, agent_id: str) -> Agent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"Agent not registered: {agent_id}") from exc

    def get_contract(self, agent_id: str) -> AgentContract:
        try:
            return self._contracts[agent_id]
        except KeyError as exc:
            raise KeyError(f"Agent not registered: {agent_id}") from exc

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def list_agent_ids(self) -> List[str]:
        return sorted(self._agents.keys())

    def list_contracts(self) -> List[AgentContract]:
        return [self._contracts[aid] for aid in self.list_agent_ids()]

    def find_by_capability(self, capability: str) -> List[Agent]:
        matched: List[Agent] = []
        for agent_id, contract in self._contracts.items():
            if capability in contract.capabilities:
                matched.append(self._agents[agent_id])
        return matched

    def find_best_match(self, task_context: object) -> Optional[Agent]:
        best: Optional[tuple[float, Agent]] = None
        for agent in self._agents.values():
            result = agent.can_handle(task_context)
            if not result.matched:
                continue
            if best is None or result.score > best[0]:
                best = (result.score, agent)
        return best[1] if best else None

    def as_dict(self) -> Dict[str, Agent]:
        """Snapshot for AgentEngine backward-compatible wiring."""
        return dict(self._agents)

    @classmethod
    def from_agents(
        cls,
        agents: Union[Dict[str, Agent], Iterable[Agent]],
    ) -> "AgentRegistry":
        registry = cls()
        if isinstance(agents, dict):
            for agent_id, agent in agents.items():
                contract = agent.get_contract()
                if contract.id != agent_id:
                    contract = contract.model_copy(update={"id": agent_id})
                registry.register(agent, contract=contract)
            return registry
        for agent in agents:
            registry.register(agent)
        return registry
