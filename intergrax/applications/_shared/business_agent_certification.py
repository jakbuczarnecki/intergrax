# © Artur Czarnecki. All rights reserved.

"""Business agent certification and deploy readiness (AUDIT-IDEAL-28.4 / K.1 / K.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from intergrax.applications._shared.cross_host_agent_certification import certify_agent_across_hosts
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_contract_meta import AgentContract

BUSINESS_AGENT_IDS: tuple[str, ...] = ("problem_radar", "vendor_discovery")


@dataclass(frozen=True, slots=True)
class BusinessAgentCertificationResult:
    agent_id: str
    passed: bool
    deploy_ready: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BusinessAgentDeployReport:
    results: tuple[BusinessAgentCertificationResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)

    @property
    def deploy_ready(self) -> bool:
        return all(item.deploy_ready for item in self.results)


def _deploy_ready(contract: AgentContract) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if not contract.capabilities:
        errors.append("business agents require at least one capability")
    if "structured_output" not in contract.validation_rules:
        errors.append("business agents require structured_output validation rule")
    if not (contract.owner_team or "").strip():
        errors.append("business agents require owner_team")
    return not errors, tuple(errors)


def certify_business_agent(
    contract: AgentContract,
    *,
    environments: tuple[ApplicationEnvironmentProfile, ...],
) -> BusinessAgentCertificationResult:
    """Certify a K-band business agent for cross-host reuse and deploy."""
    cross_host = certify_agent_across_hosts(contract, environments=environments)
    deploy_ok, deploy_errors = _deploy_ready(contract)
    errors: list[str] = []
    if not cross_host.passed:
        for item in cross_host.results:
            if not item.passed:
                errors.extend(item.errors)
    errors.extend(deploy_errors)
    passed = cross_host.passed and not deploy_errors
    return BusinessAgentCertificationResult(
        agent_id=contract.id,
        passed=passed,
        deploy_ready=passed and deploy_ok,
        errors=tuple(errors),
    )


def certify_business_agents_for_deploy(
    factories: tuple[Callable[[], object], ...],
    *,
    environments: tuple[ApplicationEnvironmentProfile, ...],
) -> BusinessAgentDeployReport:
    """Run K.1/K.2 certification across reference host profiles."""
    results: list[BusinessAgentCertificationResult] = []
    for factory in factories:
        contract = factory().get_contract()
        results.append(certify_business_agent(contract, environments=environments))
    return BusinessAgentDeployReport(results=tuple(results))
