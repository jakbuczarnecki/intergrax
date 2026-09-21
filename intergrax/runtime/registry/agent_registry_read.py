# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only runtime surface for production registry projection consumers."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.routable_tier2_agent import RoutableTier2Agent
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.tier2_agent import Tier2Agent


@runtime_checkable
class AgentRegistryRead(Protocol):
    """Runtime read surface for agents materialized from an active RuntimeRevision."""

    def get(self, agent_id: str) -> Tier2Agent:
        """Return the registered agent for ``agent_id``."""

    def get_contract(self, agent_id: str) -> AgentContract:
        """Return the registered contract for ``agent_id``."""

    def has(self, agent_id: str) -> bool:
        """Return whether ``agent_id`` is registered."""

    def list_agent_ids(self) -> List[str]:
        """Return sorted registered agent ids."""

    def list_contracts(self) -> List[AgentContract]:
        """Return registered contracts in sorted agent-id order."""

    def is_routable(self, agent_id: str, *, production_mode: bool = False) -> bool:
        """Return whether ``agent_id`` is routable under the current policy."""

    def list_routable_agent_ids(self, *, production_mode: bool = False) -> List[str]:
        """Return sorted routable agent ids."""

    def find_by_capability(
        self,
        capability: str,
        *,
        production_mode: bool = False,
    ) -> List[Tier2Agent]:
        """Return routable agents exposing ``capability``."""

    def find_best_match(
        self,
        task_context: TaskEnvelope,
        *,
        production_mode: bool = False,
    ) -> Optional[RoutableTier2Agent]:
        """Return the highest-scoring routable agent for ``task_context``."""


__all__ = ["AgentRegistryRead"]
