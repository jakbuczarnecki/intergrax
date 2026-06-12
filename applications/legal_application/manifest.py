# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative legal product roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications._shared.agent_certification_wiring import apply_roster_agent_governance
from intergrax.applications._shared.budget_wiring import product_agent_budget_slice
from intergrax.applications._shared.ownership_wiring import standard_product_operational_ownership
from intergrax.applications.contracts.environment_profile import (
    AdaptiveProfile,
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from legal.legal_agent import LegalAgent
from legal_application.host.agent_factories import build_legal_agent_from_context


_LEGAL_AGENTS = [
    AgentBinding.mount(
        LegalAgent,
        factory=build_legal_agent_from_context,
        capabilities=["legal.review"],
        default=True,
        budget_slice=product_agent_budget_slice(),
    ),
]


def _legal_environment() -> ApplicationEnvironmentProfile:
    base = (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="legal.product",
            skill_bundles=["legal"],
        )
        .model_copy(
            update={
                "integration_profile": IntegrationProfile.legal_product(),
                "adaptive_profile": AdaptiveProfile(enabled=False, mode="observe"),
            }
        )
        .with_harness_memory()
        .with_reference_host_platform_defaults()
    )
    return apply_roster_agent_governance(base, agents=_LEGAL_AGENTS, app_id="legal")


LEGAL_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="legal",
    name="Intergrax Legal API",
    route_prefix="/v1/legal",
    env_prefix="LEGAL_",
    default_port=8000,
    integration_profile=IntegrationProfile.legal_product(),
    environment=_legal_environment(),
    agents=list(_LEGAL_AGENTS),
    description="Legal review host composing Tier-2 LegalAgent",
    ownership=standard_product_operational_ownership("legal"),
)
