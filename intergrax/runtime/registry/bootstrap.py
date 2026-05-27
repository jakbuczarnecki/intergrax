# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import warnings

from intergrax.runtime.registry.agent_registry import AgentRegistry

warnings.warn(
    "intergrax.supervisor is deprecated; use Nexus RuntimeEngine, NexusLoop, and AgentRegistry instead.",
    DeprecationWarning,
    stacklevel=2,
)


def build_harness_registry(*, include_echo: bool = True) -> AgentRegistry:
    """
    Build a minimal registry for experimentation (§41).

    Registers EchoAgent by default for harness smoke tests.
    """
    registry = AgentRegistry()
    if include_echo:
        from echo.echo_agent import EchoAgent

        registry.register(EchoAgent())
    return registry
