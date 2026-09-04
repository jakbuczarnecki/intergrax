# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated CatalogSourceProvider — neutral multi-source catalog aggregation."""

from __future__ import annotations

from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogEntryFilters,
    CatalogPackageResolution,
    CatalogSourceProvider,
    ProviderHealth,
    ProviderHealthStatus,
)


def _catalog_entry_sort_key(entry: AgentCatalogEntry) -> tuple[str, ...]:
    return (
        entry.catalog_source.catalog_source_id,
        entry.catalog_source.provider_kind.value,
        entry.catalog_entry_id,
        entry.package_id_line,
    )


class FederatedCatalogSourceProvider:
    """Aggregates multiple CatalogSourceProvider instances with deterministic ordering."""

    def __init__(
        self,
        providers: tuple[CatalogSourceProvider, ...],
        *,
        federated_source_id: str = "federated",
    ) -> None:
        if not providers:
            raise ValueError("federated catalog requires at least one provider")
        seen: set[str] = set()
        for provider in providers:
            source_id = provider.catalog_source_id
            if source_id in seen:
                raise ValueError(
                    f"duplicate catalog_source_id in federation: {source_id!r}",
                )
            seen.add(source_id)
        self._providers = providers
        self._federated_source_id = federated_source_id

    @property
    def catalog_source_id(self) -> str:
        return self._federated_source_id

    @property
    def child_providers(self) -> tuple[CatalogSourceProvider, ...]:
        return self._providers

    def list_entries(
        self,
        filters: CatalogEntryFilters | None = None,
    ) -> list[AgentCatalogEntry]:
        merged: list[AgentCatalogEntry] = []
        seen_keys: set[tuple[str, str]] = set()
        for provider in self._providers:
            for entry in provider.list_entries(filters):
                key = (
                    entry.catalog_source.catalog_source_id,
                    entry.catalog_entry_id,
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(entry)
        merged.sort(key=_catalog_entry_sort_key)
        return merged

    def resolve_package(
        self,
        entry: AgentCatalogEntry,
        *,
        version_selector: str,
    ) -> CatalogPackageResolution:
        for provider in self._providers:
            if provider.catalog_source_id != entry.catalog_source.catalog_source_id:
                continue
            return provider.resolve_package(entry, version_selector=version_selector)
        raise LookupError(
            "catalog entry source does not match any federated child provider",
        )

    def health(self) -> ProviderHealth | None:
        statuses: list[ProviderHealthStatus] = []
        details: list[str] = []
        for provider in self._providers:
            probe = provider.health()
            if probe is None:
                continue
            statuses.append(probe.status)
            if probe.detail:
                details.append(f"{provider.catalog_source_id}:{probe.detail}")
        if not statuses:
            return None
        if any(status is ProviderHealthStatus.UNAVAILABLE for status in statuses):
            aggregate = ProviderHealthStatus.UNAVAILABLE
        elif any(status is ProviderHealthStatus.DEGRADED for status in statuses):
            aggregate = ProviderHealthStatus.DEGRADED
        else:
            aggregate = ProviderHealthStatus.HEALTHY
        detail = "; ".join(details) if details else None
        return ProviderHealth(status=aggregate, detail=detail)
