# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for governed_contractor_application (product profile)."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from external_contractor_adapter.external_contractor_adapter_agent import ExternalContractorAdapterAgent
from governed_contractor_application.host.agent_factories import build_governed_contractor_external_contractor_adapter_from_context


def _resolve_integration_profile() -> IntegrationProfile:
    import json
    import os

    raw = os.environ.get("INTERGRAX_INTEGRATION_PROFILE_JSON", "").strip()
    if raw:
        return IntegrationProfile.model_validate_json(raw)
    return IntegrationProfile.legal_product()


def build_governed_contractor_manifest() -> ApplicationManifest:
    return ApplicationManifest.product(
        app_id="governed_contractor",
        name="Governed Contractor API",
        route_prefix="/v1/governed_contractor",
        env_prefix="GOVERNED_CONTRACTOR_",
        default_capability="external_contractor.adapt",
        integration_profile=_resolve_integration_profile(),
        agents=[
            AgentBinding.mount(
                ExternalContractorAdapterAgent,
                factory=build_governed_contractor_external_contractor_adapter_from_context,
                capabilities=["external_contractor.adapt"],
                default=True,
            ),
        ],
        description=(
            "Governed External Contractor (GEC) Tier-3 proof host — "
            "policy, HITL, trace and receipt surfaces around an external contractor agent"
        ),
    )


APPLICATION_MANIFEST = build_governed_contractor_manifest()
