# © Artur Czarnecki. All rights reserved.

"""Declarative lab agent roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
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
                contract_id="echo",
                capabilities=["echo.basic"],
                requires_uaep=True,
            )
        )

    if settings.include_mock_agents:
        agents.extend(
            [
                AgentBinding.mount(
                    ResearchMockAgent,
                    contract_id="research_mock",
                    requires_uaep=True,
                ),
                AgentBinding.mount(
                    DocumentMockAgent,
                    contract_id="document_mock",
                    requires_uaep=True,
                ),
                AgentBinding.mount(
                    ValidatorMockAgent,
                    contract_id="validator_mock",
                    requires_uaep=True,
                ),
                AgentBinding.mount(
                    ComposerMockAgent,
                    contract_id="composer_mock",
                    requires_uaep=True,
                ),
            ]
        )

    if settings.include_signoff_probe:
        agents.append(
            AgentBinding.mount(
                SignoffProbeAgent,
                contract_id="signoff_probe",
                requires_uaep=True,
            )
        )

    if settings.include_research:
        agents.extend(
            [
                AgentBinding.mount(
                    ResearchAgent,
                    contract_id="research",
                    requires_uaep=True,
                ),
                AgentBinding.mount(
                    SummaryAgent,
                    contract_id="research-summary",
                    requires_uaep=True,
                ),
            ]
        )

    if settings.include_problem_radar:
        agents.append(
            AgentBinding.mount(
                ProblemRadarAgent,
                contract_id="problem_radar",
                capabilities=["problem_radar.scan"],
                requires_uaep=True,
            )
        )

    environment = build_lab_environment_profile(settings)
    return ApplicationManifest.lab(
        app_id="lab",
        name="Intergrax Lab Application",
        route_prefix=settings.route_prefix,
        env_prefix="LAB_",
        default_port=8090,
        environment=environment,
        agents=agents,
        description="Universal Agent OS experimentation environment",
    )


def build_lab_manifest_default() -> ApplicationManifest:
    """Manifest with default env flags (documentation / conformance)."""
    return build_lab_manifest(LabApplicationSettings())
