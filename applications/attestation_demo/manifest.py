# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for attestation_demo."""

from __future__ import annotations

from boundary_demo.boundary_demo_agent import BoundaryDemoAgent
from intergrax.applications.contracts.environment_profile import (
    AdaptiveProfile,
    ApplicationEnvironmentProfile,
    ContextProfile,
    ExecutionBoundaryExportProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.skills.registry.profile import SkillProfile


from intergrax.tools.providers.records.service import (
    RECORDS_COUNT_TOOL_ID,
    RECORDS_DELETE_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
)

_RECORDS_TOOL_IDS = [
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_DELETE_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_COUNT_TOOL_ID,
]


def build_attestation_demo_environment() -> ApplicationEnvironmentProfile:
    environment = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="attestation_demo.lab",
        harness_tools=False,
    )
    return environment.model_copy(
        update={
            "integration_profile": IntegrationProfile.lab(),
            "tool_profile": ToolProfile(
                enabled=list(_RECORDS_TOOL_IDS),
                enabled_bundles=["records"],
            ),
            "skill_profile": SkillProfile(enabled=["data.records_admin"]),
            "context_profile": ContextProfile(enable_rag=False, enable_websearch=False),
            "sandbox": None,
            "adaptive_profile": AdaptiveProfile(enabled=False, mode="observe"),
            "execution_boundary_export_profile": ExecutionBoundaryExportProfile(
                enabled=True,
                capture_mode="side_effects_only",
                include_canonical_io=True,
                step_level_enabled=True,
                host_signing_enabled=True,
                host_signing_public_key_id="attestation-demo-host-1",
            ),
        }
    )


def build_attestation_demo_manifest() -> ApplicationManifest:
    environment = build_attestation_demo_environment()
    return ApplicationManifest.lab(
        app_id="attestation_demo",
        name="Attestation Demo Lab Application",
        route_prefix="/v1/attestation_demo",
        env_prefix="ATTESTATION_DEMO_",
        integration_profile=IntegrationProfile.lab(),
        environment=environment,
        agents=[
            AgentBinding.mount(
                BoundaryDemoAgent,
                capabilities=["attestation.demo"],
            ),
        ],
        description="Partner PoC host — execution boundary events with optional EBE-9 host signing",
    )


APPLICATION_MANIFEST = build_attestation_demo_manifest()
