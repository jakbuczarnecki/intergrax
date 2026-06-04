# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative legal product roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from legal.legal_agent import LegalAgent
from legal_application.host.agent_factories import build_legal_agent_from_context


def _legal_environment() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.product_defaults(
        profile_id="legal.product",
        skill_bundles=["legal"],
    ).model_copy(update={"integration_profile": IntegrationProfile.legal_product()})


LEGAL_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="legal",
    name="Intergrax Legal API",
    route_prefix="/v1/legal",
    env_prefix="LEGAL_",
    default_port=8000,
    integration_profile=IntegrationProfile.legal_product(),
    environment=_legal_environment(),
    agents=[
        AgentBinding.mount(
            LegalAgent,
            factory=build_legal_agent_from_context,
            capabilities=["legal.review"],
            default=True,
        ),
    ],
    description="Legal review host composing Tier-2 LegalAgent",
)
