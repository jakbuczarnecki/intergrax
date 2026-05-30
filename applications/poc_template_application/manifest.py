# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for poc_template_application."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from echo.echo_agent import EchoAgent


def build_poc_template_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="poc_template",
        name="Poc Template Lab Application",
        route_prefix="/v1/poc_template",
        env_prefix="POC_TEMPLATE_",
        agents=[
        AgentBinding.mount(EchoAgent, capabilities=['echo.basic']),
        ],
        description="Scaffolded Tier-3 lab environment (Phase N.3)",
    )


APPLICATION_MANIFEST = build_poc_template_manifest()
