# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Dict, List, Optional

from intergrax.agents.agent_contract import Agent
from intergrax.agents.harness_reference_agent import assert_uaep_reference_agent
from intergrax.agents.uaep_protocol import UAEPAgent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_assembly_resolver import assert_agent_assembly_valid
from intergrax.runtime.registry.agent_routing_policy import evaluate_agent_routing
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.skills.integration.contract_resolution import resolve_contract_tools
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import ResolvedSkillPack, SkillResolver
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.tools.registry.runtime import ToolRegistry


def _bootstrap_default_skill_registry() -> SkillRegistry:
    """Load first-party skill catalog when Tier-3 did not pass an explicit registry."""
    register_default_skills()
    return build_registry_from_profile(SkillProfile(register_all_catalog_bundles=True))


class AgentRegistry:
    """
    Central registry for Tier-2 agents (canonical architecture §15).

    Nexus and AgentEngine use this for discovery and selection.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}
        self._contracts: Dict[str, AgentContract] = {}
        self._resolved_skill_packs: Dict[str, ResolvedSkillPack] = {}

    def register(
        self,
        agent: Agent,
        *,
        contract: Optional[AgentContract] = None,
        skill_registry: Optional[SkillRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        requires_uaep: bool = False,
    ) -> None:
        if requires_uaep:
            assert_uaep_reference_agent(agent)
        meta = contract or agent.get_contract()
        assert_agent_assembly_valid(meta)
        if meta.id in self._agents:
            raise ValueError(f"Agent already registered: {meta.id}")
        resolved_pack: ResolvedSkillPack | None = None
        if meta.skills or meta.extra_tools:
            if skill_registry is None:
                skill_registry = _bootstrap_default_skill_registry()
            resolver = SkillResolver(skill_registry, tool_registry)
            meta, resolved_pack = resolve_contract_tools(
                meta,
                skill_resolver=resolver,
                tool_registry=tool_registry,
            )
        self._agents[meta.id] = agent
        self._contracts[meta.id] = meta
        if resolved_pack is not None:
            self._resolved_skill_packs[meta.id] = resolved_pack

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

    def get_resolved_skill_pack(self, agent_id: str) -> ResolvedSkillPack | None:
        """Return the immutable skill composition snapshot bound at registration, if any."""
        return self._resolved_skill_packs.get(agent_id)

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def list_agent_ids(self) -> List[str]:
        return sorted(self._agents.keys())

    def list_contracts(self) -> List[AgentContract]:
        return [self._contracts[aid] for aid in self.list_agent_ids()]

    def is_routable(self, agent_id: str, *, production_mode: bool = False) -> bool:
        contract = self.get_contract(agent_id)
        return evaluate_agent_routing(contract, production_mode=production_mode).routable

    def list_routable_agent_ids(self, *, production_mode: bool = False) -> List[str]:
        return [
            agent_id
            for agent_id in self.list_agent_ids()
            if self.is_routable(agent_id, production_mode=production_mode)
        ]

    def find_by_capability(
        self,
        capability: str,
        *,
        production_mode: bool = False,
    ) -> List[Agent]:
        matched: List[Agent] = []
        for agent_id, contract in self._contracts.items():
            if capability not in contract.capabilities:
                continue
            if not self.is_routable(agent_id, production_mode=production_mode):
                continue
            matched.append(self._agents[agent_id])
        return matched

    def find_best_match(
        self,
        task_context: object,
        *,
        production_mode: bool = False,
    ) -> Optional[Agent]:
        best: Optional[tuple[float, Agent]] = None
        for agent_id, agent in self._agents.items():
            if not self.is_routable(agent_id, production_mode=production_mode):
                continue
            result = agent.can_handle(task_context)
            if not result.matched:
                continue
            if best is None or result.score > best[0]:
                best = (result.score, agent)
        return best[1] if best else None

