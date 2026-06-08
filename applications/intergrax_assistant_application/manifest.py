# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for intergrax_assistant_application."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax_assistant.intergrax_assistant_agent import IntergraxAssistantAgent
from intergrax_assistant_application.host.environment_profile import (
    build_intergrax_assistant_environment_profile,
)
from intergrax_assistant_application.host.settings import IntergraxAssistantApplicationSettings


def build_intergrax_assistant_manifest(
    settings: IntergraxAssistantApplicationSettings | None = None,
) -> ApplicationManifest:
    """Build manifest from env toggles — hub agent always mounted; specialists optional."""
    settings = settings or IntergraxAssistantApplicationSettings.from_env()
    environment = build_intergrax_assistant_environment_profile(settings)
    agents: list[AgentBinding] = [
        AgentBinding.mount(
            IntergraxAssistantAgent,
            capabilities=["platform.assist"],
            requires_uaep=True,
        ),
    ]

    if settings.include_echo:
        from echo.echo_agent import EchoAgent

        agents.append(
            AgentBinding.mount(EchoAgent, capabilities=["echo.basic"], requires_uaep=True)
        )

    if settings.include_legal:
        from legal.legal_agent import LegalAgent

        agents.append(
            AgentBinding.mount(LegalAgent, capabilities=["legal.review"], requires_uaep=True)
        )

    if settings.include_research:
        from research.research_agent import ResearchAgent
        from research.summary_agent import SummaryAgent

        agents.extend(
            [
                AgentBinding.mount(
                    ResearchAgent,
                    capabilities=["research.web_search", "research.pipeline"],
                    requires_uaep=True,
                ),
                AgentBinding.mount(
                    SummaryAgent,
                    capabilities=["research.summarize"],
                    requires_uaep=True,
                ),
            ]
        )

    return ApplicationManifest.lab(
        app_id="intergrax_assistant",
        name="Intergrax Assistant Lab Application",
        route_prefix=settings.route_prefix,
        env_prefix="INTERGRAX_ASSISTANT_",
        default_port=settings.backend_port,
        integration_profile=IntegrationProfile.lab_stack(),
        environment=environment,
        agents=agents,
        description=(
            "Harness-native conversational environment — hub agent, swappable LLM, "
            "optional platform specialist delegation"
        ),
    )


def build_intergrax_assistant_manifest_default() -> ApplicationManifest:
    """Manifest with default env flags (documentation / conformance)."""
    return build_intergrax_assistant_manifest(IntergraxAssistantApplicationSettings())
