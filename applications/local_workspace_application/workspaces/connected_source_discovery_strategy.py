# © Artur Czarnecki. All rights reserved.

"""Provider-neutral strategy contract for connected-source discovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceCandidateV1,
    RemoteResourceTypeV1,
)


@dataclass(frozen=True, slots=True)
class ConnectedSourceRevalidationLimits:
    max_pages: int = 8
    max_total_candidates: int = 800
    max_duration_seconds: float = 5.0
    page_size: int = 100


@dataclass(frozen=True, slots=True)
class RemoteResourceStrategyPage:
    """Provider-neutral candidates plus an opaque-to-the-application cursor value."""

    items: tuple[RemoteResourceCandidateV1, ...]
    provider_cursor: str | None


class RemoteResourceDiscoveryStrategy(Protocol):
    resource_type: RemoteResourceTypeV1

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        ...

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        limits: ConnectedSourceRevalidationLimits,
    ) -> str:
        ...


class RemoteResourceDiscoveryStrategyRegistry:
    """Immutable resource-type registry for provider-owned discovery strategies."""

    def __init__(
        self,
        strategies: Iterable[RemoteResourceDiscoveryStrategy] = (),
    ) -> None:
        entries: dict[RemoteResourceTypeV1, RemoteResourceDiscoveryStrategy] = {}
        for strategy in strategies:
            resource_type = strategy.resource_type
            if resource_type in entries:
                raise ValueError("duplicate_remote_resource_discovery_strategy")
            entries[resource_type] = strategy
        self._strategies = MappingProxyType(entries)

    def resolve(
        self,
        resource_type: RemoteResourceTypeV1,
    ) -> RemoteResourceDiscoveryStrategy:
        strategy = self._strategies.get(resource_type)
        if strategy is None:
            raise ConnectedSourceDiscoveryError("resource_type_unsupported")
        return strategy
