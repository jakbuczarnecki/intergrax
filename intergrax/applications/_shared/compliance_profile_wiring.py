# © Artur Czarnecki. All rights reserved.

"""Compliance profile template wiring for regulated hosts (AUDIT-IDEAL-5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.policy.compliance_profiles import (
    ComplianceDomainClass,
    ComplianceProfileTemplate,
    compliance_domain_fragments,
    resolve_compliance_template,
)


@dataclass(frozen=True, slots=True)
class ComplianceProfileWiring:
    enabled: bool
    template: ComplianceProfileTemplate | None
    domain_fragments: dict[str, Any]


def resolve_compliance_profile_wiring(
    env: ApplicationEnvironmentProfile,
) -> ComplianceProfileWiring:
    """Apply regulated-domain compliance templates on product hosts."""
    compliance = env.compliance_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ComplianceProfileWiring(enabled=False, template=None, domain_fragments={})
    if not compliance.enabled:
        return ComplianceProfileWiring(enabled=False, template=None, domain_fragments={})

    template = resolve_compliance_template(compliance.domain_class)
    return ComplianceProfileWiring(
        enabled=True,
        template=template,
        domain_fragments=compliance_domain_fragments(template),
    )
