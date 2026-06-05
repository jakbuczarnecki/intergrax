# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for poc_template_application."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from echo.echo_agent import EchoAgent


def build_poc_template_manifest() -> ApplicationManifest:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="poc_template.scaffold")
    return ApplicationManifest.lab(
        app_id="poc_template",
        name="Poc Template Lab Application",
        route_prefix="/v1/poc_template",
        env_prefix="POC_TEMPLATE_",
        integration_profile=IntegrationProfile.lab_stack(),
        environment=environment,
        agents=[
            AgentBinding.mount(EchoAgent, capabilities=["echo.basic"]),
        ],
        description="Scaffolded Tier-3 lab environment (Phase AA-POC.2)",
    )


APPLICATION_MANIFEST = build_poc_template_manifest()
