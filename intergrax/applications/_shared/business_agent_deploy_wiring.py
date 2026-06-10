# © Artur Czarnecki. All rights reserved.

"""Business agent deploy wiring for product hosts (AUDIT-IDEAL-28.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from intergrax.applications._shared.business_agent_certification import (
    BUSINESS_AGENT_IDS,
    BusinessAgentDeployReport,
    certify_business_agents_for_deploy,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class BusinessAgentDeployWiring:
    enabled: bool
    report: BusinessAgentDeployReport | None
    agent_ids: tuple[str, ...]


def resolve_business_agent_deploy_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    agent_factories: tuple[Callable[[], object], ...],
    reference_environments: tuple[ApplicationEnvironmentProfile, ...] | None = None,
) -> BusinessAgentDeployWiring:
    """Validate K.1/K.2 deploy readiness when business agent deployment is enabled."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return BusinessAgentDeployWiring(enabled=False, report=None, agent_ids=())
    if not env.host_deployment_profile.business_agents_deploy_enabled:
        return BusinessAgentDeployWiring(enabled=False, report=None, agent_ids=())

    environments = reference_environments or (
        ApplicationEnvironmentProfile.lab_defaults(),
        ApplicationEnvironmentProfile.product_defaults(),
    )
    report = certify_business_agents_for_deploy(agent_factories, environments=environments)
    if not report.deploy_ready:
        return BusinessAgentDeployWiring(enabled=False, report=report, agent_ids=BUSINESS_AGENT_IDS)

    return BusinessAgentDeployWiring(
        enabled=True,
        report=report,
        agent_ids=BUSINESS_AGENT_IDS,
    )
