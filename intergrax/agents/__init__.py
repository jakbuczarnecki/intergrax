# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Framework agent bridge (Tier-1 → Tier-2)."""

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.uaep import UAEPExecutor, supports_uaep

__all__ = ["Agent", "AgentEngine", "UAEPExecutor", "supports_uaep"]
