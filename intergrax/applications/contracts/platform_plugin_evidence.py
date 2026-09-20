# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application platform plugin bootstrap/admission evidence DTO (APP-ADOPTION-1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable


@runtime_checkable
class DomainPluginLoadReportView(Protocol):
    """Immutable per-domain plugin load/admission evidence (structural contract)."""

    group: str
    registered_count: int

    def to_audit_dict(self) -> dict[str, object]: ...


PLATFORM_PLUGIN_DOMAIN_MEMORY = "memory"
PLATFORM_PLUGIN_DOMAIN_CONTEXT = "context"
PLATFORM_PLUGIN_DOMAIN_POLICY = "policy"
PLATFORM_PLUGIN_DOMAIN_SECURITY = "security"
PLATFORM_PLUGIN_DOMAIN_TOOLS = "tools"
PLATFORM_PLUGIN_DOMAIN_SKILLS = "skills"
PLATFORM_PLUGIN_DOMAIN_INTEGRATIONS = "integrations"
PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS = "rag_chunkers"
PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS = "rag_retrievers"
PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS = "rag_rerankers"


@dataclass(frozen=True, slots=True)
class ApplicationPlatformPluginEvidence:
    """
    Immutable bootstrap snapshot of per-domain plugin load/admission evidence.

    Not a global installed-plugin inventory and not production qualification.
    """

    _domain_reports: Mapping[str, DomainPluginLoadReportView]

    @classmethod
    def from_domain_reports(
        cls,
        reports: Mapping[str, DomainPluginLoadReportView],
    ) -> ApplicationPlatformPluginEvidence:
        return cls(_domain_reports=MappingProxyType(dict(reports)))

    @property
    def domain_reports(self) -> Mapping[str, DomainPluginLoadReportView]:
        return self._domain_reports

    def report_for(self, domain: str) -> DomainPluginLoadReportView | None:
        """Return domain evidence when that domain participated in bootstrap."""
        return self._domain_reports.get(domain)

    def memory_report(self) -> DomainPluginLoadReportView:
        """Memory domain always participates in Tier-3 environment wiring."""
        return self._domain_reports[PLATFORM_PLUGIN_DOMAIN_MEMORY]


__all__ = [
    "PLATFORM_PLUGIN_DOMAIN_CONTEXT",
    "PLATFORM_PLUGIN_DOMAIN_INTEGRATIONS",
    "PLATFORM_PLUGIN_DOMAIN_MEMORY",
    "PLATFORM_PLUGIN_DOMAIN_POLICY",
    "PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS",
    "PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS",
    "PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS",
    "PLATFORM_PLUGIN_DOMAIN_SECURITY",
    "PLATFORM_PLUGIN_DOMAIN_SKILLS",
    "PLATFORM_PLUGIN_DOMAIN_TOOLS",
    "ApplicationPlatformPluginEvidence",
    "DomainPluginLoadReportView",
]
