# © Artur Czarnecki. All rights reserved.

"""Build dispute_sim ApplicationEnvironmentProfile with ORCH-CONFIG wiring."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.catalog_manifests import OTEL
from dispute_sim_application.host.graph_spec import DEFAULT_DISPUTE_SIM_GRAPH
from dispute_sim_application.host.settings import DisputeSimBackendSettings


def build_dispute_sim_environment_profile(
    settings: DisputeSimBackendSettings,
) -> ApplicationEnvironmentProfile:
    _ = settings
    profile = ApplicationEnvironmentProfile.product_defaults(
        skill_bundles=["harness", "legal"],
        profile_id="dispute_sim.product",
    )
    profile.observability_profile.otel_enabled = True
    profile.observability_profile.debug_surface_override = True
    otel_backend = IntegrationBinding.from_manifest(OTEL)
    profile.integration_profile = profile.integration_profile.model_copy(
        update={
            "observability_backend": otel_backend,
            "options": {**profile.integration_profile.options, OTEL.slug: {}},
        },
    )
    profile.context_profile = profile.context_profile.model_copy(
        update={"enable_rag": True, "enable_websearch": True},
    )
    profile.graph_spec = DEFAULT_DISPUTE_SIM_GRAPH
    profile.orchestration_profile = OrchestrationProfile(
        classifier_kind="rules",
        merge_strategy="structured_json",
        intent_routes=[
            IntentRoute(
                capability="dispute.pipeline",
                keywords=[
                    "podwykonaw",
                    "pismo",
                    "odpowied",
                    "odpisa",
                    "spór",
                    "spor",
                    "zapłat",
                    "zaplata",
                    "korespondenc",
                    "subcontractor",
                    "payment",
                    "defect",
                ],
            ),
            IntentRoute(
                capability="dispute.intake",
                keywords=["załącz", "zalacz", "index", "materiał", "material", "ingest"],
            ),
        ],
    )
    return profile.with_harness_memory()
