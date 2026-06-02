# © Artur Czarnecki. All rights reserved.

"""Declarative lab agent roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from echo.echo_agent import EchoAgent
from lab.mock_agents import (
    ComposerMockAgent,
    DocumentMockAgent,
    ResearchMockAgent,
    ValidatorMockAgent,
)
from problem_radar.problem_radar_agent import ProblemRadarAgent
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from signoff_probe.signoff_probe_agent import SignoffProbeAgent
from lab_application.host.settings import LabApplicationSettings


def build_lab_manifest(settings: LabApplicationSettings) -> ApplicationManifest:
    """Build manifest from lab env toggles (mirrors legacy ``build_lab_registry`` flags)."""
    agents: list[AgentBinding] = []

    if settings.include_echo:
        agents.append(
            AgentBinding.mount(
                EchoAgent,
                capabilities=["echo.basic"],
                requires_uaep=True,
            )
        )

    if settings.include_mock_agents:
        agents.extend(
            [
                AgentBinding.mount(ResearchMockAgent, requires_uaep=True),
                AgentBinding.mount(DocumentMockAgent, requires_uaep=True),
                AgentBinding.mount(ValidatorMockAgent, requires_uaep=True),
                AgentBinding.mount(ComposerMockAgent, requires_uaep=True),
            ]
        )

    if settings.include_signoff_probe:
        agents.append(AgentBinding.mount(SignoffProbeAgent, requires_uaep=True))

    if settings.include_research:
        agents.extend(
            [
                AgentBinding.mount(ResearchAgent),
                AgentBinding.mount(SummaryAgent),
            ]
        )

    if settings.include_problem_radar:
        agents.append(
            AgentBinding.mount(
                ProblemRadarAgent,
                capabilities=["problem_radar.scan"],
                requires_uaep=True,
            )
        )

    return ApplicationManifest.lab(
        app_id="lab",
        name="Intergrax Lab Application",
        route_prefix=settings.route_prefix,
        env_prefix="LAB_",
        default_port=8090,
        agents=agents,
        description="Universal Agent OS experimentation environment",
    )


def build_lab_manifest_default() -> ApplicationManifest:
    """Manifest with default env flags (documentation / conformance)."""
    return build_lab_manifest(LabApplicationSettings())
