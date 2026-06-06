# © Artur Czarnecki. All rights reserved.

"""Static harness reference manifests for capability-graph seeding (FAUDIT-TIER.1).

Tier-0 catalog uses contract-id bindings only — no imports from ``applications/``.
"""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest


def _binding(contract_id: str) -> AgentBinding:
    return AgentBinding.reference(contract_id)


def _lab_reference_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="lab",
        name="Intergrax Lab Application (reference)",
        route_prefix="/v1/lab",
        env_prefix="LAB_",
        default_port=8090,
        agents=[_binding("echo")],
        description="Harness reference lab roster for capability graph edges",
    )


def _legal_reference_manifest() -> ApplicationManifest:
    return ApplicationManifest.product(
        app_id="legal",
        name="Intergrax Legal API (reference)",
        route_prefix="/v1/legal",
        env_prefix="LEGAL_",
        default_port=8000,
        agents=[_binding("legal")],
        description="Harness reference legal roster for capability graph edges",
    )


def _research_reference_manifest() -> ApplicationManifest:
    return ApplicationManifest.product(
        app_id="research",
        name="Intergrax Research API (reference)",
        route_prefix="/v1/research",
        env_prefix="RESEARCH_",
        default_port=8010,
        agents=[_binding("research"), _binding("research-summary")],
        description="Harness reference research roster for capability graph edges",
    )


def _poc_template_reference_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="poc_template",
        name="Poc Template Lab Application (reference)",
        route_prefix="/v1/poc_template",
        env_prefix="POC_TEMPLATE_",
        agents=[_binding("echo")],
        description="Harness reference poc_template roster for capability graph edges",
    )


def build_harness_reference_manifests() -> tuple[ApplicationManifest, ...]:
    """Reference Tier-3 manifests for harness capability graph edges."""
    return (
        _lab_reference_manifest(),
        _legal_reference_manifest(),
        _research_reference_manifest(),
        _poc_template_reference_manifest(),
    )
