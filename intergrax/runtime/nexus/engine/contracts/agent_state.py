# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC


class AgentState(ABC):
    """
    Marker base class for agent-specific state.

    Runtime MUST treat this as opaque container.
    Agents provide concrete implementations (e.g. LegalAgentState).
    """
    pass