# © Artur Czarnecki. All rights reserved.

"""Post-materialization rebind for strict sanctioned external-work reflex agent (F03)."""

from __future__ import annotations

from typing import List, Optional

from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from governed_contractor_application.host.agent_builders import (
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.routable_tier2_agent import RoutableTier2Agent
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead

_EXTERNAL_CONTRACTOR_ADAPTER_AGENT_ID = "external_contractor_adapter"


class _StrictSanctionedExternalWorkAgentRegistryOverlay:
    """Delegates to revision-bound registry; overrides AC3 external contractor agent."""

    __slots__ = ("_delegate", "_strict_agent", "_strict_agent_id")

    def __init__(
        self,
        delegate: AgentRegistryRead,
        *,
        strict_agent: ExternalContractorAdapterAgent,
        strict_agent_id: str,
    ) -> None:
        self._delegate = delegate
        self._strict_agent = strict_agent
        self._strict_agent_id = strict_agent_id

    def _resolve(self, agent_id: str) -> Tier2Agent:
        if agent_id == self._strict_agent_id:
            return self._strict_agent
        return self._delegate.get(agent_id)

    def get(self, agent_id: str) -> Tier2Agent:
        return self._resolve(agent_id)

    def get_contract(self, agent_id: str) -> AgentContract:
        if agent_id == self._strict_agent_id:
            return self._strict_agent.get_contract()
        return self._delegate.get_contract(agent_id)

    def has(self, agent_id: str) -> bool:
        return self._delegate.has(agent_id)

    def list_agent_ids(self) -> List[str]:
        return self._delegate.list_agent_ids()

    def list_contracts(self) -> List[AgentContract]:
        return self._delegate.list_contracts()

    def is_routable(self, agent_id: str, *, production_mode: bool = False) -> bool:
        return self._delegate.is_routable(agent_id, production_mode=production_mode)

    def list_routable_agent_ids(self, *, production_mode: bool = False) -> List[str]:
        return self._delegate.list_routable_agent_ids(production_mode=production_mode)

    def find_by_capability(
        self,
        capability: str,
        *,
        production_mode: bool = False,
    ) -> List[Tier2Agent]:
        agents = self._delegate.find_by_capability(
            capability,
            production_mode=production_mode,
        )
        if capability != "external_contractor.adapt":
            return agents
        replaced = [
            self._strict_agent if agent.get_contract().id == self._strict_agent_id else agent
            for agent in agents
        ]
        return replaced

    def find_best_match(
        self,
        task_context: TaskEnvelope,
        *,
        production_mode: bool = False,
    ) -> Optional[RoutableTier2Agent]:
        matched = self._delegate.find_best_match(
            task_context,
            production_mode=production_mode,
        )
        if matched is None:
            return None
        if matched.get_contract().id == self._strict_agent_id:
            return self._strict_agent
        return matched


def rebind_strict_sanctioned_external_contractor_adapter_registry(
    registry: AgentRegistryRead,
    *,
    host_runtime: GovernedContractorHostRuntimeComposition,
    manifest: ApplicationManifest,
    build_context: ApplicationBuildContext,
) -> AgentRegistryRead:
    """Bind revision-bound AC3 traffic to host-sanctioned external-work operation port."""
    if not host_runtime.require_sanctioned_external_work_mutations:
        return registry
    if host_runtime.sanctioned_external_work_operation_port is None:
        raise ValueError(
            "strict sanctioned external work requires sanctioned_external_work_operation_port",
        )
    binding = next(
        (
            item
            for item in manifest.agents
            if item.contract_id == "external_contractor_adapter"
        ),
        manifest.agents[0],
    )
    builders = build_governed_contractor_agent_builders(host_runtime)
    strict_agent = builders[ExternalContractorAdapterAgent](build_context, binding)
    if not isinstance(strict_agent, ExternalContractorAdapterAgent):
        raise TypeError("strict external contractor adapter materialization failed")
    agent_id = strict_agent.get_contract().id
    return _StrictSanctionedExternalWorkAgentRegistryOverlay(
        registry,
        strict_agent=strict_agent,
        strict_agent_id=agent_id,
    )


__all__ = ["rebind_strict_sanctioned_external_contractor_adapter_registry"]
