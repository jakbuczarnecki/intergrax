# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Framework agent bridge (Tier-1 → Tier-2)."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep_protocol import UAEPAgent, supports_uaep

__all__ = ["Agent", "UAEPAgent", "supports_uaep"]
