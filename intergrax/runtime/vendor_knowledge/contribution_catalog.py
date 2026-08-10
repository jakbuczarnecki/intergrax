"""Discovery and composition boundary for Vendor Knowledge contributions."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Final

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    VendorKnowledgeLiveRegistrationRegistry,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeSourcePluginRegistry,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_registry import (
    TenantConnectionIntegrationFactoryRegistry,
)

VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP: Final = (
    "intergrax.vendor_knowledge.providers"
)
VendorKnowledgeContributionFactory = Callable[[], VendorKnowledgeProviderContribution]
VendorKnowledgeProviderKey = tuple[str, IntegrationCategory]


class VendorKnowledgePluginLoadError(ValueError):
    """Raised when an opted-in external contribution cannot be loaded."""


class VendorKnowledgePluginConflict(ValueError):
    """Raised when external or built-in contributions collide."""


def _entry_points(group: str) -> tuple[object, ...]:
    try:
        discovered = importlib.metadata.entry_points()
    except Exception:
        raise VendorKnowledgePluginLoadError("entry_point_discovery_failed") from None
    if hasattr(discovered, "select"):
        selected = discovered.select(group=group)
    else:
        selected = discovered.get(group, ())
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                str(getattr(item, "name", "")),
                str(getattr(item, "value", "")),
            ),
        )
    )


def discover_vendor_knowledge_contributions() -> tuple[VendorKnowledgeProviderContribution, ...]:
    """Load opted-in entry-point contributions without arbitrary module scanning."""
    loaded: list[VendorKnowledgeProviderContribution] = []
    seen_names: set[str] = set()
    for entry_point in _entry_points(VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP):
        name = getattr(entry_point, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise VendorKnowledgePluginLoadError("entry_point_name_invalid")
        if name in seen_names:
            raise VendorKnowledgePluginConflict("duplicate_entry_point_name")
        seen_names.add(name)
        load = getattr(entry_point, "load", None)
        if not callable(load):
            raise VendorKnowledgePluginLoadError("entry_point_loader_invalid")
        try:
            target = load()
            contribution = (
                target
                if isinstance(target, VendorKnowledgeProviderContribution)
                else target()
                if callable(target)
                else None
            )
        except Exception:
            raise VendorKnowledgePluginLoadError("external_contribution_load_failed") from None
        if not isinstance(contribution, VendorKnowledgeProviderContribution):
            raise VendorKnowledgePluginLoadError("external_contribution_invalid")
        loaded.append(contribution)
    return tuple(loaded)


class VendorKnowledgeContributionCatalog:
    """Instance-local contribution catalog with publication-snapshot semantics."""

    def __init__(
        self,
        contributions: Iterable[VendorKnowledgeProviderContribution] = (),
    ) -> None:
        self._contributions: dict[
            VendorKnowledgeProviderKey,
            VendorKnowledgeProviderContribution,
        ] = {}
        for contribution in contributions:
            self.register(contribution)

    def register(self, contribution: VendorKnowledgeProviderContribution) -> None:
        if not isinstance(contribution, VendorKnowledgeProviderContribution):
            raise TypeError("vendor_knowledge_contribution_invalid")
        key = contribution.provider_key
        existing = self._contributions.get(key)
        if existing is not None:
            if existing == contribution:
                return
            raise VendorKnowledgePluginConflict("conflicting_provider_contribution")
        self._validate_components(contribution)
        self._contributions[key] = contribution

    def list_contributions(self) -> tuple[VendorKnowledgeProviderContribution, ...]:
        """Return an immutable deterministic publication snapshot."""
        return tuple(
            self._contributions[key]
            for key in sorted(
                self._contributions,
                key=lambda item: (item[0], item[1].value),
            )
        )

    def _registration_snapshot(
        self,
    ) -> tuple[VendorKnowledgeProviderContribution, ...]:
        """Return deterministic built-in publication order for parity-sensitive consumers."""
        return tuple(self._contributions.values())

    def with_connection_factory_overrides(
        self,
        overrides: Mapping[
            VendorKnowledgeProviderKey,
            VendorKnowledgeConnectionFactoryContribution,
        ],
    ) -> VendorKnowledgeContributionCatalog:
        known = set(self._contributions)
        unknown = set(overrides) - known
        if unknown:
            raise VendorKnowledgePluginConflict("connection_factory_provider_unknown")
        updated = []
        for contribution in self.list_contributions():
            override = overrides.get(contribution.provider_key)
            if override is None:
                updated.append(contribution)
                continue
            if override.key != contribution.provider_key:
                raise VendorKnowledgePluginConflict(
                    "connection_factory_identity_mismatch"
                )
            updated.append(
                replace(contribution, connection_factories=(override,))
            )
        return VendorKnowledgeContributionCatalog(updated)

    def _validate_components(
        self,
        contribution: VendorKnowledgeProviderContribution,
    ) -> None:
        for existing in self._contributions.values():
            if any(
                left.identity.key == right.identity.key
                for left in contribution.source_plugins
                for right in existing.source_plugins
            ):
                raise VendorKnowledgePluginConflict("duplicate_source_plugin")
            if any(
                (left.provider_id, left.integration_kind, left.source_kind)
                == (right.provider_id, right.integration_kind, right.source_kind)
                for left in contribution.adapters
                for right in existing.adapters
            ):
                raise VendorKnowledgePluginConflict("duplicate_adapter")
            if any(
                left.runtime_ref == right.runtime_ref
                for left in contribution.indexed_materializers
                for right in existing.indexed_materializers
            ):
                raise VendorKnowledgePluginConflict("duplicate_materializer_runtime_ref")
            if any(
                (
                    left.descriptor.provider_id,
                    left.descriptor.integration_kind,
                    left.descriptor.source_kind,
                    left.descriptor.capability_id,
                )
                == (
                    right.descriptor.provider_id,
                    right.descriptor.integration_kind,
                    right.descriptor.source_kind,
                    right.descriptor.capability_id,
                )
                for left in contribution.live_contributions
                for right in existing.live_contributions
            ):
                raise VendorKnowledgePluginConflict("duplicate_live_capability")


def _default_builtin_builders() -> tuple[VendorKnowledgeContributionFactory, ...]:
    from intergrax.runtime.vendor_knowledge.confluence_contribution import (
        build_confluence_vendor_knowledge_contribution,
    )
    from intergrax.runtime.vendor_knowledge.databricks_contribution import (
        build_databricks_vendor_knowledge_contribution,
    )
    from intergrax.runtime.vendor_knowledge.google_workspace_contribution import (
        build_google_workspace_vendor_knowledge_contribution,
    )
    from intergrax.runtime.vendor_knowledge.jira_contribution import (
        build_jira_vendor_knowledge_contribution,
    )
    from intergrax.runtime.vendor_knowledge.msgraph_contribution import (
        build_msgraph_vendor_knowledge_contribution,
    )
    from intergrax.runtime.vendor_knowledge.slack_contribution import (
        build_slack_vendor_knowledge_contribution,
    )

    return (
        build_msgraph_vendor_knowledge_contribution,
        build_slack_vendor_knowledge_contribution,
        build_google_workspace_vendor_knowledge_contribution,
        build_jira_vendor_knowledge_contribution,
        build_confluence_vendor_knowledge_contribution,
        build_databricks_vendor_knowledge_contribution,
    )


def build_default_vendor_knowledge_contribution_catalog(
    *,
    discover_entry_points: bool = False,
    built_in_builders: Iterable[VendorKnowledgeContributionFactory] | None = None,
) -> VendorKnowledgeContributionCatalog:
    """Build built-ins plus optional explicitly enabled external contributions."""
    builders = (
        tuple(built_in_builders)
        if built_in_builders is not None
        else _default_builtin_builders()
    )
    contributions: list[VendorKnowledgeProviderContribution] = []
    for builder in builders:
        try:
            contribution = builder()
        except Exception:
            raise VendorKnowledgePluginLoadError("builtin_contribution_load_failed") from None
        if not isinstance(contribution, VendorKnowledgeProviderContribution):
            raise VendorKnowledgePluginLoadError("builtin_contribution_invalid")
        contributions.append(contribution)
    if discover_entry_points:
        contributions.extend(discover_vendor_knowledge_contributions())
    return VendorKnowledgeContributionCatalog(contributions)


def build_vendor_knowledge_adapter_registry(
    catalog: VendorKnowledgeContributionCatalog,
) -> KnowledgeAdapterRegistry:
    registry = KnowledgeAdapterRegistry()
    for contribution in catalog._registration_snapshot():
        for adapter in contribution.adapters:
            registry.register(adapter)
    return registry


def build_vendor_knowledge_source_plugin_registry(
    catalog: VendorKnowledgeContributionCatalog,
) -> VendorKnowledgeSourcePluginRegistry:
    registry = VendorKnowledgeSourcePluginRegistry()
    for contribution in catalog.list_contributions():
        for plugin in contribution.source_plugins:
            registry.register(plugin)
    return registry


def build_vendor_knowledge_connection_factory_registry(
    catalog: VendorKnowledgeContributionCatalog,
) -> TenantConnectionIntegrationFactoryRegistry:
    factories = tuple(
        (
            contribution.provider_id,
            contribution.integration_category,
            factory.factory,
        )
        for contribution in catalog.list_contributions()
        for factory in contribution.connection_factories
    )
    return TenantConnectionIntegrationFactoryRegistry(factories)


def build_vendor_knowledge_live_registration_registry(
    catalog: VendorKnowledgeContributionCatalog,
) -> VendorKnowledgeLiveRegistrationRegistry:
    plugin_registry = VendorKnowledgeSourcePluginRegistry()
    for contribution in catalog.list_contributions():
        for plugin in contribution.source_plugins:
            if plugin.supports(VendorKnowledgeMode.LIVE):
                plugin_registry.register(plugin)
    registry = VendorKnowledgeLiveRegistrationRegistry(
        plugin_registry=plugin_registry,
    )
    bundles = tuple(
        bundle
        for contribution in catalog.list_contributions()
        for bundle in contribution.live_contributions
    )
    registry.register(bundles)
    return registry


__all__ = [
    "VENDOR_KNOWLEDGE_PROVIDER_ENTRY_POINT_GROUP",
    "VendorKnowledgeContributionCatalog",
    "VendorKnowledgePluginConflict",
    "VendorKnowledgePluginLoadError",
    "build_default_vendor_knowledge_contribution_catalog",
    "build_vendor_knowledge_adapter_registry",
    "build_vendor_knowledge_connection_factory_registry",
    "build_vendor_knowledge_live_registration_registry",
    "build_vendor_knowledge_source_plugin_registry",
    "discover_vendor_knowledge_contributions",
]
