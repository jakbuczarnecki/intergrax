"""Immutable provider contribution ABI for Vendor Knowledge extensions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeAdapter
from intergrax.runtime.vendor_knowledge.live.registration import (
    LiveRegistrationBundleV1,
    publish_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
)

VENDOR_KNOWLEDGE_PROVIDER_CONTRIBUTION_CONTRACT_VERSION: Final = (
    "vendor-knowledge.provider-contribution.v1"
)
APPLICATION_OWNED_EXTENSION_SURFACE: Final = "APPLICATION_OWNED_EXTENSION_SURFACE"
_PROVIDER_EXTENSION_SOURCE_KIND = "__provider_extension__"
_MAX_REFERENCE_LENGTH = 256

VendorKnowledgeDiscoveryFactory = Callable[..., object]
VendorKnowledgeIndexedMaterializerFactory = Callable[[], object]


class VendorKnowledgeContributionError(ValueError):
    """Raised when a provider contribution is invalid or internally inconsistent."""


def _provider_identity(
    provider_id: object,
    integration_category: object,
) -> tuple[str, IntegrationCategory]:
    try:
        identity = VendorKnowledgeSourceIdentity(
            provider_id=provider_id,
            integration_category=integration_category,
            source_kind=_PROVIDER_EXTENSION_SOURCE_KIND,
        )
    except (TypeError, ValueError):
        raise VendorKnowledgeContributionError("provider_identity_invalid") from None
    return identity.provider_id, identity.integration_category


def _reference(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise VendorKnowledgeContributionError(f"{field_name}_invalid")
    cleaned = value.strip()
    if not cleaned or cleaned != value:
        raise VendorKnowledgeContributionError(f"{field_name}_invalid")
    if len(cleaned) > _MAX_REFERENCE_LENGTH:
        raise VendorKnowledgeContributionError(f"{field_name}_too_long")
    return cleaned


def _component_identity(
    component: object,
    *,
    component_name: str,
    provider_id: str,
    integration_category: IntegrationCategory,
) -> VendorKnowledgeSourceIdentity:
    try:
        identity = VendorKnowledgeSourceIdentity(
            provider_id=getattr(component, "provider_id"),
            integration_category=getattr(component, "integration_kind"),
            source_kind=getattr(component, "source_kind"),
        )
    except (AttributeError, TypeError, ValueError):
        raise VendorKnowledgeContributionError(
            f"{component_name}_identity_invalid"
        ) from None
    if (
        identity.provider_id != provider_id
        or identity.integration_category is not integration_category
    ):
        raise VendorKnowledgeContributionError(
            f"{component_name}_identity_mismatch"
        )
    return identity


@dataclass(frozen=True, slots=True)
class VendorKnowledgeConnectionFactoryContribution:
    """Provider/category factory hook; it never contains tenant state or secrets."""

    provider_id: str
    integration_category: IntegrationCategory
    factory: TenantConnectionIntegrationFactory

    def __post_init__(self) -> None:
        provider_id, integration_category = _provider_identity(
            self.provider_id,
            self.integration_category,
        )
        if not isinstance(self.factory, TenantConnectionIntegrationFactory):
            raise VendorKnowledgeContributionError("connection_factory_invalid")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "integration_category", integration_category)

    @property
    def key(self) -> tuple[str, IntegrationCategory]:
        return (self.provider_id, self.integration_category)


@dataclass(frozen=True, slots=True)
class VendorKnowledgeDiscoveryContribution:
    """Hook for the application-owned connected-resource discovery surface.

    ``APPLICATION_OWNED_EXTENSION_SURFACE`` is intentional: the current
    discovery strategy protocol depends on LKW host resources and is not
    imported into the canonical runtime ABI.
    """

    identity: VendorKnowledgeSourceIdentity
    factory: VendorKnowledgeDiscoveryFactory

    def __post_init__(self) -> None:
        if not isinstance(self.identity, VendorKnowledgeSourceIdentity):
            raise VendorKnowledgeContributionError("discovery_identity_invalid")
        if not callable(self.factory):
            raise VendorKnowledgeContributionError("discovery_factory_invalid")


@dataclass(frozen=True, slots=True)
class VendorKnowledgeIndexedMaterializerContribution:
    """Typed hook for the application-owned indexed materializer surface."""

    identity: VendorKnowledgeSourceIdentity
    runtime_ref: str
    factory: VendorKnowledgeIndexedMaterializerFactory

    def __post_init__(self) -> None:
        if not isinstance(self.identity, VendorKnowledgeSourceIdentity):
            raise VendorKnowledgeContributionError("materializer_identity_invalid")
        runtime_ref = _reference(self.runtime_ref, field_name="materializer_runtime_ref")
        if not callable(self.factory):
            raise VendorKnowledgeContributionError("materializer_factory_invalid")
        object.__setattr__(self, "runtime_ref", runtime_ref)


@dataclass(frozen=True, slots=True)
class VendorKnowledgeProviderContribution:
    """One immutable provider/category contribution bundle.

    The bundle contains declarative source descriptors and approved construction
    hooks only. It is not a registry, service locator, dependency-injection
    container, tenant connection, workspace configuration, or secrets store.
    """

    provider_id: str
    integration_category: IntegrationCategory
    adapters: tuple[VendorKnowledgeAdapter, ...] = ()
    source_plugins: tuple[VendorKnowledgeSourcePlugin, ...] = ()
    connection_factories: tuple[VendorKnowledgeConnectionFactoryContribution, ...] = ()
    discovery_contributions: tuple[VendorKnowledgeDiscoveryContribution, ...] = ()
    indexed_materializers: tuple[
        VendorKnowledgeIndexedMaterializerContribution, ...
    ] = ()
    live_contributions: tuple[LiveRegistrationBundleV1, ...] = ()
    contract_version: str = VENDOR_KNOWLEDGE_PROVIDER_CONTRIBUTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        provider_id, integration_category = _provider_identity(
            self.provider_id,
            self.integration_category,
        )
        contract_version = _reference(
            self.contract_version,
            field_name="contribution_contract_version",
        )
        adapters = tuple(self.adapters)
        source_plugins = tuple(self.source_plugins)
        connection_factories = tuple(self.connection_factories)
        discovery_contributions = tuple(self.discovery_contributions)
        indexed_materializers = tuple(self.indexed_materializers)
        live_contributions = tuple(self.live_contributions)

        plugin_by_key: dict[
            tuple[str, IntegrationCategory, str],
            VendorKnowledgeSourcePlugin,
        ] = {}
        for plugin in source_plugins:
            if not isinstance(plugin, VendorKnowledgeSourcePlugin):
                raise VendorKnowledgeContributionError("source_plugin_invalid")
            identity = plugin.identity
            if (
                identity.provider_id != provider_id
                or identity.integration_category is not integration_category
            ):
                raise VendorKnowledgeContributionError("source_plugin_identity_mismatch")
            if identity.key in plugin_by_key:
                raise VendorKnowledgeContributionError("duplicate_source_identity")
            plugin_by_key[identity.key] = plugin

        adapter_identities: list[VendorKnowledgeSourceIdentity] = []
        for adapter in adapters:
            if not isinstance(adapter, VendorKnowledgeAdapter):
                raise VendorKnowledgeContributionError("adapter_invalid")
            identity = _component_identity(
                adapter,
                component_name="adapter",
                provider_id=provider_id,
                integration_category=integration_category,
            )
            if identity.key in {item.key for item in adapter_identities}:
                raise VendorKnowledgeContributionError("duplicate_adapter_identity")
            if identity.key not in plugin_by_key:
                raise VendorKnowledgeContributionError("adapter_source_plugin_missing")
            adapter_identities.append(identity)

        factory_keys: set[tuple[str, IntegrationCategory]] = set()
        for contribution in connection_factories:
            if not isinstance(
                contribution,
                VendorKnowledgeConnectionFactoryContribution,
            ):
                raise VendorKnowledgeContributionError("connection_factory_invalid")
            if contribution.key != (provider_id, integration_category):
                raise VendorKnowledgeContributionError(
                    "connection_factory_identity_mismatch"
                )
            if contribution.key in factory_keys:
                raise VendorKnowledgeContributionError(
                    "duplicate_connection_factory_identity"
                )
            factory_keys.add(contribution.key)

        discovery_keys: set[tuple[str, IntegrationCategory, str]] = set()
        for contribution in discovery_contributions:
            if not isinstance(contribution, VendorKnowledgeDiscoveryContribution):
                raise VendorKnowledgeContributionError("discovery_contribution_invalid")
            identity = contribution.identity
            if (
                identity.provider_id != provider_id
                or identity.integration_category is not integration_category
            ):
                raise VendorKnowledgeContributionError(
                    "discovery_identity_mismatch"
                )
            if identity.key not in plugin_by_key:
                raise VendorKnowledgeContributionError(
                    "discovery_source_plugin_missing"
                )
            if identity.key in discovery_keys:
                raise VendorKnowledgeContributionError(
                    "duplicate_discovery_identity"
                )
            discovery_keys.add(identity.key)

        materializer_keys: set[tuple[str, IntegrationCategory, str]] = set()
        materializer_runtime_refs: set[str] = set()
        for contribution in indexed_materializers:
            if not isinstance(
                contribution,
                VendorKnowledgeIndexedMaterializerContribution,
            ):
                raise VendorKnowledgeContributionError(
                    "materializer_contribution_invalid"
                )
            identity = contribution.identity
            if (
                identity.provider_id != provider_id
                or identity.integration_category is not integration_category
            ):
                raise VendorKnowledgeContributionError(
                    "materializer_identity_mismatch"
                )
            plugin = plugin_by_key.get(identity.key)
            if plugin is None:
                raise VendorKnowledgeContributionError(
                    "materializer_source_plugin_missing"
                )
            indexed = plugin.capability(VendorKnowledgeMode.INDEXED)
            if indexed is None:
                raise VendorKnowledgeContributionError(
                    "materializer_mode_not_declared"
                )
            if indexed.runtime_ref != contribution.runtime_ref:
                raise VendorKnowledgeContributionError(
                    "materializer_runtime_ref_mismatch"
                )
            if identity.key in materializer_keys:
                raise VendorKnowledgeContributionError(
                    "duplicate_materializer_identity"
                )
            if contribution.runtime_ref in materializer_runtime_refs:
                raise VendorKnowledgeContributionError(
                    "duplicate_materializer_runtime_ref"
                )
            materializer_keys.add(identity.key)
            materializer_runtime_refs.add(contribution.runtime_ref)

        for bundle in live_contributions:
            if not isinstance(bundle, LiveRegistrationBundleV1):
                raise VendorKnowledgeContributionError("live_contribution_invalid")

        try:
            publish_live_registration_bundles(live_contributions)
        except ValueError as exc:
            code = str(exc)
            if code != "duplicate_live_capability_identity":
                code = "live_contribution_invalid"
            raise VendorKnowledgeContributionError(code) from None
        except Exception:
            raise VendorKnowledgeContributionError("live_contribution_invalid") from None

        live_by_capability: dict[
            tuple[str, IntegrationCategory, str, str],
            LiveRegistrationBundleV1,
        ] = {}
        for bundle in live_contributions:
            descriptor = bundle.descriptor
            identity = _component_identity(
                descriptor,
                component_name="live",
                provider_id=provider_id,
                integration_category=integration_category,
            )
            if identity.key not in plugin_by_key:
                raise VendorKnowledgeContributionError(
                    "live_source_plugin_missing"
                )
            key = (
                identity.provider_id,
                identity.integration_category,
                identity.source_kind,
                descriptor.capability_id,
            )
            live_by_capability[key] = bundle

        for plugin in source_plugins:
            live = plugin.capability(VendorKnowledgeMode.LIVE)
            if live is None:
                continue
            for capability_id in live.capability_refs:
                bundle = live_by_capability.get(
                    (
                        plugin.identity.provider_id,
                        plugin.identity.integration_category,
                        plugin.identity.source_kind,
                        capability_id,
                    )
                )
                if bundle is None:
                    raise VendorKnowledgeContributionError(
                        "live_capability_registration_missing"
                    )
                if bundle.descriptor.source_kind != plugin.identity.source_kind:
                    raise VendorKnowledgeContributionError(
                        "live_capability_source_mismatch"
                    )

        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "integration_category", integration_category)
        object.__setattr__(
            self,
            "adapters",
            tuple(
                sorted(
                    adapters,
                    key=lambda item: (
                        item.provider_id,
                        item.integration_kind.value,
                        item.source_kind,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "source_plugins",
            tuple(
                sorted(
                    source_plugins,
                    key=lambda item: (
                        item.identity.provider_id,
                        item.identity.integration_category.value,
                        item.identity.source_kind,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "connection_factories",
            tuple(
                sorted(
                    connection_factories,
                    key=lambda item: (item.provider_id, item.integration_category.value),
                )
            ),
        )
        object.__setattr__(
            self,
            "discovery_contributions",
            tuple(
                sorted(
                    discovery_contributions,
                    key=lambda item: (
                        item.identity.provider_id,
                        item.identity.integration_category.value,
                        item.identity.source_kind,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "indexed_materializers",
            tuple(
                sorted(
                    indexed_materializers,
                    key=lambda item: (
                        item.identity.provider_id,
                        item.identity.integration_category.value,
                        item.identity.source_kind,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "live_contributions",
            tuple(
                sorted(
                    live_contributions,
                    key=lambda item: (
                        item.descriptor.provider_id,
                        item.descriptor.integration_kind.value,
                        item.descriptor.source_kind,
                        item.descriptor.capability_id,
                        item.descriptor.contract_version,
                    ),
                )
            ),
        )
        object.__setattr__(self, "contract_version", contract_version)

    @property
    def source_identities(self) -> tuple[VendorKnowledgeSourceIdentity, ...]:
        """Return source identities in deterministic canonical order."""

        return tuple(plugin.identity for plugin in self.source_plugins)

    @property
    def provider_key(self) -> tuple[str, IntegrationCategory]:
        return (self.provider_id, self.integration_category)


__all__ = [
    "APPLICATION_OWNED_EXTENSION_SURFACE",
    "VendorKnowledgeConnectionFactoryContribution",
    "VendorKnowledgeContributionError",
    "VendorKnowledgeDiscoveryContribution",
    "VendorKnowledgeDiscoveryFactory",
    "VendorKnowledgeIndexedMaterializerContribution",
    "VendorKnowledgeIndexedMaterializerFactory",
    "VendorKnowledgeProviderContribution",
    "VENDOR_KNOWLEDGE_PROVIDER_CONTRIBUTION_CONTRACT_VERSION",
]
