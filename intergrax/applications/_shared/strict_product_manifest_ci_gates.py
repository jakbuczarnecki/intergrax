# © Artur Czarnecki. All rights reserved.

"""CI-only STRICT product manifest gates (lab compatibility authority allowed).

These helpers validate manifest-local product rules before revision-bound lifecycle
authority exists in the repository checkout. They MUST NOT be used on production
runtime serving or activation paths.
"""

from __future__ import annotations

from intergrax.applications._shared.capability_graph_deploy_gate import (
    validate_strict_capability_graph_deploy,
)
from intergrax.applications._shared.agent_certification_wiring import (
    validate_strict_roster_agent_certification,
)
from intergrax.applications._shared.roster_agent_contract_authority import (
    ManifestAgentContractAuthority,
    materialize_manifest_contract_authority_lab_compat,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest


def _gate_wiring_environment(env):
    from intergrax.applications.contracts.application_host import ApplicationProfile
    from intergrax.integrations.registry.profile import IntegrationProfile

    return env.model_copy(
        update={
            "integration_profile": IntegrationProfile.lab(),
            "application_profile": ApplicationProfile.LAB,
        }
    )


def manifest_ci_contract_authority(
    manifest: ApplicationManifest,
) -> ManifestAgentContractAuthority:
    """CI-only manifest contract snapshots (not production serving)."""
    return materialize_manifest_contract_authority_lab_compat(manifest)


def check_strict_product_capability_graph(
    product_id: str,
    manifest: ApplicationManifest,
) -> list[str]:
    """Return deploy-gate violations for one STRICT product manifest (CI only)."""
    from intergrax.applications._shared.environment_wiring import wire_application_environment

    env = manifest.resolved_environment()
    if env.execution_mode is not ExecutionMode.STRICT:
        return []
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return []

    gate_env = _gate_wiring_environment(env)
    try:
        wiring = wire_application_environment(manifest, gate_env, conformance_check=False)
    except Exception as exc:  # noqa: BLE001 — gate surfaces wiring failures
        return [f"{product_id}: wire_application_environment failed: {exc}"]

    view = wiring.capability_graph
    snapshot = wiring.registry_snapshot
    if view is None:
        return [f"{product_id}: capability_graph not materialized"]
    if snapshot is None:
        return [f"{product_id}: registry_snapshot not materialized"]

    authority = materialize_manifest_contract_authority_lab_compat(manifest)
    result = validate_strict_capability_graph_deploy(
        view,
        snapshot,
        manifest,
        env,
        contract_authority=authority,
    )
    return [f"{product_id}: {error}" for error in result.errors]


def check_strict_product_agent_certification(
    product_id: str,
    manifest: ApplicationManifest,
) -> list[str]:
    """Return certification-gate violations for one STRICT product manifest (CI only)."""
    env = manifest.resolved_environment()
    authority = materialize_manifest_contract_authority_lab_compat(manifest)
    prefix = f"{product_id}:"
    return [
        f"{prefix}{item}"
        for item in validate_strict_roster_agent_certification(
            manifest,
            env,
            contract_authority=authority,
        )
    ]


__all__ = [
    "check_strict_product_agent_certification",
    "check_strict_product_capability_graph",
    "manifest_ci_contract_authority",
]
