# © Artur Czarnecki. All rights reserved.

"""Cross-host agent reuse certification (AUDIT-IDEAL-18.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_assembly_resolver import validate_agent_assembly


@dataclass(frozen=True, slots=True)
class CrossHostCertificationResult:
    agent_id: str
    host_profile_id: str
    passed: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossHostCertificationReport:
    results: tuple[CrossHostCertificationResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)


def certify_agent_across_hosts(
    contract: AgentContract,
    *,
    environments: tuple[ApplicationEnvironmentProfile, ...],
) -> CrossHostCertificationReport:
    """Ensure the same agent contract is assembly-valid on each reference host profile."""
    results: list[CrossHostCertificationResult] = []
    for env in environments:
        assembly = validate_agent_assembly(contract)
        errors = list(assembly.errors)
        if contract.production_eligible and env.application_profile.value == "product":
            if not (contract.modality_profile_id or "").strip():
                errors.append("production_eligible agents require modality_profile_id on product hosts")
        results.append(
            CrossHostCertificationResult(
                agent_id=contract.id,
                host_profile_id=env.profile_id,
                passed=not errors,
                errors=tuple(errors),
            )
        )
    return CrossHostCertificationReport(results=tuple(results))
