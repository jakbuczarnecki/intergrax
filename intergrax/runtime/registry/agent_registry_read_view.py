# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable read-only view over a completed :class:`AgentRegistry`."""

from __future__ import annotations

from typing import List, Optional

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


class AgentRegistryReadView:
    """Runtime read surface without construction mutation authority."""

    __slots__ = ("_delegate",)

    def __init__(self, delegate: AgentRegistry) -> None:
        self._delegate = delegate

    def get(self, agent_id: str) -> Agent:
        return self._delegate.get(agent_id)

    def get_contract(self, agent_id: str) -> AgentContract:
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
    ) -> List[Agent]:
        return self._delegate.find_by_capability(
            capability,
            production_mode=production_mode,
        )

    def find_best_match(
        self,
        task_context: object,
        *,
        production_mode: bool = False,
    ) -> Optional[Agent]:
        return self._delegate.find_best_match(
            task_context,
            production_mode=production_mode,
        )


def freeze_agent_registry(registry: AgentRegistry) -> AgentRegistryRead:
    """Return a read-only runtime surface for one completed registry projection."""
    return AgentRegistryReadView(registry)


__all__ = ["AgentRegistryReadView", "freeze_agent_registry"]
