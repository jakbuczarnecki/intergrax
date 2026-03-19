# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from abc import ABC, abstractmethod

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


class Agent(ABC):
    """
    Tier-2 Agent contract.

    Agent is responsible for:
    - building RuntimeContext (including RuntimeConfig)
    - configuring pipeline via config.pipeline

    Agent is NOT responsible for:
    - RuntimeState
    - execution
    - lifecycle management
    """

    @abstractmethod
    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        """
        Build fully configured RuntimeContext for this agent.
        Must include:
        - RuntimeConfig
        - config.pipeline
        - all required dependencies
        """
        ...