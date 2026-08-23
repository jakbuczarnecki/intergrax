# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application platform plugin bootstrap/admission evidence (APP-ADOPTION-1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_MEMORY_STORES
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle

PLATFORM_PLUGIN_DOMAIN_MEMORY = "memory"
PLATFORM_PLUGIN_DOMAIN_CONTEXT = "context"
PLATFORM_PLUGIN_DOMAIN_POLICY = "policy"
PLATFORM_PLUGIN_DOMAIN_SECURITY = "security"


@dataclass(frozen=True, slots=True)
class ApplicationPlatformPluginEvidence:
    """
    Immutable bootstrap snapshot of per-domain plugin load/admission evidence.

    Not a global installed-plugin inventory and not production qualification.
    """

    _domain_reports: Mapping[str, DomainPluginLoadReport]

    @classmethod
    def from_domain_reports(
        cls,
        reports: Mapping[str, DomainPluginLoadReport],
    ) -> ApplicationPlatformPluginEvidence:
        return cls(_domain_reports=MappingProxyType(dict(reports)))

    @property
    def domain_reports(self) -> Mapping[str, DomainPluginLoadReport]:
        return self._domain_reports

    def report_for(self, domain: str) -> DomainPluginLoadReport | None:
        """Return domain evidence when that domain participated in bootstrap."""
        return self._domain_reports.get(domain)

    def memory_report(self) -> DomainPluginLoadReport:
        """Memory domain always participates in Tier-3 environment wiring."""
        return self._domain_reports[PLATFORM_PLUGIN_DOMAIN_MEMORY]


def build_application_platform_plugin_evidence(
    *,
    memory_report: DomainPluginLoadReport,
    context_report: DomainPluginLoadReport,
    security_report: DomainPluginLoadReport,
    policy_bundle: RuntimePolicyBundle,
) -> ApplicationPlatformPluginEvidence:
    """Compose application evidence from the same domain wiring invocations."""
    reports: dict[str, DomainPluginLoadReport] = {
        PLATFORM_PLUGIN_DOMAIN_MEMORY: memory_report,
        PLATFORM_PLUGIN_DOMAIN_CONTEXT: context_report,
        PLATFORM_PLUGIN_DOMAIN_SECURITY: security_report,
    }
    declarative_runtime = policy_bundle.declarative_policy_runtime
    if declarative_runtime is not None:
        reports[PLATFORM_PLUGIN_DOMAIN_POLICY] = declarative_runtime.load_report
    return ApplicationPlatformPluginEvidence.from_domain_reports(reports)


def empty_memory_platform_plugin_evidence() -> ApplicationPlatformPluginEvidence:
    """Deterministic baseline when only Memory domain participates."""
    return ApplicationPlatformPluginEvidence.from_domain_reports(
        {
            PLATFORM_PLUGIN_DOMAIN_MEMORY: DomainPluginLoadReport.empty(EP_MEMORY_STORES),
        },
    )
