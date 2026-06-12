# © Artur Czarnecki. All rights reserved.

"""STRICT product roster agent certification gate (APP-EVOL-4 · §49.4)."""

from __future__ import annotations

from intergrax.applications._shared.capability_graph_deploy_gate import (
    STRICT_DEPLOY_BLOCKED_AGENT_LIFECYCLES,
)
from intergrax.applications.contracts.agent_governance import (
    AgentCertificationRecord,
    AgentGovernanceProfile,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.runtime.registry.semver_compat import is_compatible_runtime
from intergrax.utils.time_provider import SystemTimeProvider

_CERTIFICATION_REQUIRED_STATES: frozenset[AgentLifecycleState] = frozenset(
    {
        AgentLifecycleState.STAGING,
        AgentLifecycleState.PRODUCTION,
    },
)


def materialize_roster_certifications_for_agents(
    agents: list[AgentBinding],
    *,
    app_id: str,
    certified_by: str = "platform-reference-host",
) -> AgentGovernanceProfile:
    """Build staging certification records for a reference product roster."""
    records: list[AgentCertificationRecord] = []
    certified_at = SystemTimeProvider.utc_now().isoformat()
    for binding in agents:
        if not binding.enabled:
            continue
        contract = binding.resolved_agent_type()().get_contract()
        records.append(
            AgentCertificationRecord(
                agent_id=contract.id,
                agent_version=contract.version,
                certified_at=certified_at,
                certified_by=certified_by,
                evidence_refs=[f"{app_id}:reference-host-staging-roster"],
            ),
        )
    return AgentGovernanceProfile(certifications=records)


def apply_roster_agent_governance(
    env: ApplicationEnvironmentProfile,
    *,
    agents: list[AgentBinding],
    app_id: str,
) -> ApplicationEnvironmentProfile:
    """Attach roster certification records to an environment profile."""
    return env.model_copy(
        update={
            "agent_governance_profile": materialize_roster_certifications_for_agents(
                agents,
                app_id=app_id,
            ),
        },
    )


def validate_certification_record(
    record: AgentCertificationRecord,
    contract: AgentContract,
) -> list[str]:
    """Validate one certification record against a resolved agent contract."""
    violations: list[str] = []
    if record.agent_id != contract.id:
        violations.append(
            f"certification agent_id {record.agent_id!r} does not match contract {contract.id!r}",
        )
    compat = is_compatible_runtime(record.agent_version, contract.version)
    if not compat.compatible:
        violations.append(
            f"certification version {record.agent_version!r} incompatible with contract "
            f"{contract.version!r}: {compat.reason}",
        )
    return violations


def validate_strict_roster_agent_certification(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
) -> list[str]:
    """Validate STRICT product roster lifecycle and certification coverage."""
    if env.execution_mode is not ExecutionMode.STRICT:
        return []
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return []

    violations: list[str] = []
    governance = env.agent_governance_profile
    allowed_states = frozenset(governance.approval_policy.allowed_states_for_strict)
    certifications = {record.agent_id: record for record in governance.certifications}

    for binding in manifest.enabled_agents():
        contract = binding.resolved_agent_type()().get_contract()
        contract_id = contract.id
        lifecycle = contract.lifecycle_state

        if lifecycle in STRICT_DEPLOY_BLOCKED_AGENT_LIFECYCLES:
            violations.append(
                f"roster agent {contract_id} lifecycle {lifecycle.value} blocked on STRICT product host",
            )
            continue

        if lifecycle not in allowed_states:
            violations.append(
                f"roster agent {contract_id} lifecycle {lifecycle.value} not in "
                f"allowed_states_for_strict",
            )
            continue

        if lifecycle not in _CERTIFICATION_REQUIRED_STATES:
            continue

        record = certifications.get(contract_id)
        if record is None:
            violations.append(
                f"roster agent {contract_id} requires AgentCertificationRecord on STRICT product host",
            )
            continue
        violations.extend(validate_certification_record(record, contract))

    return violations


def check_strict_product_agent_certification(
    product_id: str,
    manifest: ApplicationManifest,
) -> list[str]:
    """Return certification-gate violations for one STRICT product manifest."""
    env = manifest.resolved_environment()
    prefix = f"{product_id}:"
    return [f"{prefix}{item}" for item in validate_strict_roster_agent_certification(manifest, env)]
