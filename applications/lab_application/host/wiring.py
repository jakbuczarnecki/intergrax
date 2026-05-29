# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.settings import LabApplicationSettings


def build_lab_registry(*, settings: LabApplicationSettings | None = None) -> AgentRegistry:
    """
    Compose the default lab agent registry (Echo + optional mocks + optional Research).

    Applications select agents; agents remain reusable across applications (Tier-3 rule).
    """
    settings = settings or LabApplicationSettings.from_env()
    registry = AgentRegistry()

    if settings.include_echo:
        from echo.echo_agent import EchoAgent

        registry.register(EchoAgent())

    if settings.include_mock_agents:
        from lab.mock_agents import (
            ComposerMockAgent,
            DocumentMockAgent,
            ResearchMockAgent,
            ValidatorMockAgent,
        )

        registry.register(ResearchMockAgent())
        registry.register(DocumentMockAgent())
        registry.register(ValidatorMockAgent())
        registry.register(ComposerMockAgent())

    if settings.include_signoff_probe:
        from signoff_probe.signoff_probe_agent import SignoffProbeAgent

        registry.register(SignoffProbeAgent())

    if settings.include_research:
        from research.research_agent import ResearchAgent
        from research.summary_agent import SummaryAgent

        registry.register(ResearchAgent())
        registry.register(SummaryAgent())

    return registry
