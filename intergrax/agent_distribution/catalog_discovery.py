# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog-backed agent discovery for production composition (AC-4 Phase 9)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryRequest,
    AgentDiscoveryResult,
    AgentDiscoveryStrategyId,
    build_agent_discovery_result,
    project_package_contract_capabilities,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentProjectMetadataProvider,
)
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    AgentPackageCandidate,
)
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
)

CATALOG_PROVIDER_DISCOVERY_STRATEGY_ID: Final = AgentDiscoveryStrategyId(
    value="catalog.providers.v1",
)
EMPTY_PRODUCTION_DISCOVERY_STRATEGY_ID: Final = AgentDiscoveryStrategyId(
    value="production.empty.v1",
)


class EmptyProductionDiscoveryStrategy:
    """Inert discovery for AC-3-only process compositions."""

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return EMPTY_PRODUCTION_DISCOVERY_STRATEGY_ID

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        return build_agent_discovery_result(
            strategy_id=self.strategy_id,
            request=request,
            candidates=(),
        )


class CatalogSourceProviderDiscoveryStrategy:
    """Discover candidates from registered catalog providers with metadata projection."""

    def __init__(
        self,
        *,
        catalog_registry: CatalogSourceProviderRegistry,
        metadata_provider: AgentProjectMetadataProvider,
        package_metadata_refs: Mapping[str, str],
        strategy_id: AgentDiscoveryStrategyId = CATALOG_PROVIDER_DISCOVERY_STRATEGY_ID,
    ) -> None:
        self._catalog_registry = catalog_registry
        self._metadata_provider = metadata_provider
        self._package_metadata_refs = dict(package_metadata_refs)
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return self._strategy_id

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        allowed = request.scope.allowed_catalog_source_ids
        allowed_set = frozenset(allowed) if allowed else None
        candidates: list[AgentDiscoveryCandidate] = []
        for source_id in self._catalog_registry.registered_source_ids:
            if allowed_set is not None and source_id not in allowed_set:
                continue
            provider = self._catalog_registry.require(source_id)
            for entry in provider.list_entries():
                metadata_ref = self._package_metadata_refs.get(entry.package_id_line)
                if metadata_ref is None:
                    continue
                metadata = self._metadata_provider.get_metadata(metadata_ref)
                if metadata is None:
                    continue
                contract = (
                    metadata.declared_contracts[0]
                    if metadata.declared_contracts
                    else None
                )
                if contract is None:
                    continue
                resolution = provider.resolve_package(
                    entry,
                    version_selector=metadata.package_version or "1.0.0",
                )
                package = resolution.package_candidate
                identity = AgentDiscoveryCandidateIdentity(
                    source=entry.catalog_source,
                    package=AgentPackageCandidate(
                        distribution_package_id=package.distribution_package_id,
                        package_version=package.package_version,
                        package_digest=package.package_digest,
                    ),
                )
                candidates.append(
                    AgentDiscoveryCandidate(
                        identity=identity,
                        capabilities=project_package_contract_capabilities(contract),
                        catalog_entry_id=entry.catalog_entry_id,
                        artifact_locator=resolution.artifact_locator,
                    ),
                )
        return build_agent_discovery_result(
            strategy_id=self._strategy_id,
            request=request,
            candidates=tuple(candidates),
        )


__all__ = [
    "CATALOG_PROVIDER_DISCOVERY_STRATEGY_ID",
    "CatalogSourceProviderDiscoveryStrategy",
    "EMPTY_PRODUCTION_DISCOVERY_STRATEGY_ID",
    "EmptyProductionDiscoveryStrategy",
]
