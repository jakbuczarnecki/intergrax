from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import BaseModel

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.contracts import (
    LiveCapabilityExecutionResultV1,
    LiveCapabilityHandlerV1,
)
from intergrax.runtime.vendor_knowledge.live.identity import (
    CapabilityIdentityV1,
    LIVE_CONTRACT_VERSION,
    exact_capability_key,
    validate_capability_identity,
)
from intergrax.runtime.vendor_knowledge.live.schemas import (
    SchemaRegistrationV1,
    SchemaRegistryV1,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
    VendorKnowledgeSourcePluginRegistry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)
from intergrax.utils import attribute_access

# Backwards-compatible name; the runtime protocol is the sole authority.
LiveCapabilityHandlerProtocolV1 = LiveCapabilityHandlerV1


@dataclass(frozen=True, slots=True)
class LiveRegistrationBundleV1:
    descriptor: LiveCapabilityDescriptorV1
    handler: LiveCapabilityHandlerV1
    request_schema: SchemaRegistrationV1
    result_schema: SchemaRegistrationV1


@dataclass(frozen=True, slots=True)
class PublishedLiveRegistrationV1:
    descriptors: MappingProxyType
    handlers: MappingProxyType
    schemas: SchemaRegistryV1

    def resolve_descriptor(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        capability_id: str,
        contract_version: str,
    ) -> LiveCapabilityDescriptorV1:
        identity = validate_capability_identity(
            capability_id=capability_id,
            provider_id=provider_id,
            integration_kind=integration_kind,
            source_kind=self._source_kind_from_descriptor(
                provider_id=provider_id,
                integration_kind=integration_kind,
                capability_id=capability_id,
                contract_version=contract_version,
            ),
            contract_version=contract_version,
        )
        descriptor = self.descriptors.get(exact_capability_key(identity))
        if descriptor is None:
            raise LookupError("live_capability_unavailable")
        return descriptor

    def resolve_handler(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        capability_id: str,
        contract_version: str,
    ) -> LiveCapabilityHandlerV1:
        key = (provider_id, integration_kind, capability_id, contract_version)
        handler = self.handlers.get(key)
        if handler is None:
            raise LookupError("live_capability_unavailable")
        return handler

    def _source_kind_from_descriptor(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        capability_id: str,
        contract_version: str,
    ) -> str:
        for key, descriptor in self.descriptors.items():
            if key == (
                provider_id,
                integration_kind,
                capability_id,
                contract_version,
            ):
                return descriptor.source_kind
        raise LookupError("live_capability_unavailable")


def _handler_identity(handler: LiveCapabilityHandlerV1) -> CapabilityIdentityV1:
    try:
        return validate_capability_identity(
            capability_id=handler.capability_id,
            provider_id=handler.provider_id,
            integration_kind=handler.integration_kind,
            source_kind=handler.source_kind,
            contract_version=handler.contract_version,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid_live_handler_identity") from exc


def _validate_bundle(
    bundle: LiveRegistrationBundleV1,
) -> tuple[CapabilityIdentityV1, type[BaseModel], type[BaseModel]]:
    if not isinstance(bundle, LiveRegistrationBundleV1):
        raise ValueError("invalid_live_registration_bundle")
    if not isinstance(bundle.descriptor, LiveCapabilityDescriptorV1) or not isinstance(
        bundle.request_schema, SchemaRegistrationV1
    ) or not isinstance(bundle.result_schema, SchemaRegistrationV1):
        raise ValueError("live_registration_bundle_incomplete")
    try:
        execute = bundle.handler.execute
    except AttributeError as exc:
        raise ValueError("live_handler_execution_missing") from exc
    if not callable(execute):
        raise ValueError("live_handler_execution_not_callable")
    descriptor = bundle.descriptor
    identity = validate_capability_identity(
        capability_id=descriptor.capability_id,
        provider_id=descriptor.provider_id,
        integration_kind=descriptor.integration_kind,
        source_kind=descriptor.source_kind,
        contract_version=descriptor.contract_version,
    )
    if descriptor.effect is not CapabilityEffectV1.READ or not descriptor.read_only:
        raise ValueError("live_descriptor_must_be_read_only")
    handler_identity = _handler_identity(bundle.handler)
    if handler_identity != identity:
        raise ValueError("live_descriptor_handler_identity_mismatch")
    if bundle.request_schema.schema_ref != descriptor.request_schema_ref:
        raise ValueError("live_request_schema_reference_mismatch")
    if bundle.result_schema.schema_ref != descriptor.result_schema_ref:
        raise ValueError("live_result_schema_reference_mismatch")
    if bundle.request_schema.role.value != "request":
        raise ValueError("live_request_schema_role_mismatch")
    if bundle.result_schema.role.value != "result":
        raise ValueError("live_result_schema_role_mismatch")
    if bundle.request_schema.contract_version != descriptor.contract_version:
        raise ValueError("live_request_schema_version_mismatch")
    if bundle.result_schema.contract_version != descriptor.contract_version:
        raise ValueError("live_result_schema_version_mismatch")
    if not {
        "call_id",
        "items",
        "item_count",
        "byte_count",
        "normalized_outcome",
    }.issubset(bundle.result_schema.model.model_fields):
        raise ValueError("live_result_schema_incompatible")
    try:
        if not issubclass(bundle.result_schema.model, LiveCapabilityExecutionResultV1):
            raise ValueError("live_result_schema_incompatible")
    except TypeError as exc:
        raise ValueError("live_result_schema_incompatible") from exc
    try:
        expected_model = bundle.handler.expected_request_model
        if expected_model is not bundle.request_schema.model:
            raise ValueError("live_handler_request_model_mismatch")
        if bundle.handler.request_schema_ref != descriptor.request_schema_ref:
            raise ValueError("live_handler_request_schema_mismatch")
        if bundle.handler.result_schema_ref != descriptor.result_schema_ref:
            raise ValueError("live_handler_result_schema_mismatch")
    except AttributeError as exc:
        raise ValueError("live_handler_schema_declaration_missing") from exc
    return identity, bundle.request_schema.model, bundle.result_schema.model


def publish_live_registration_bundles(
    bundles: Iterable[LiveRegistrationBundleV1],
    *,
    additional_descriptors: Iterable[LiveCapabilityDescriptorV1] = (),
    additional_handlers: Iterable[LiveCapabilityHandlerV1] = (),
) -> PublishedLiveRegistrationV1:
    """Validate everything first, then publish one immutable snapshot."""
    if tuple(additional_descriptors) or tuple(additional_handlers):
        raise ValueError("live_additional_registration_inputs_not_supported")

    descriptors: dict[
        tuple[str, IntegrationCategory, str, str], LiveCapabilityDescriptorV1
    ] = {}
    handlers: dict[
        tuple[str, IntegrationCategory, str, str], LiveCapabilityHandlerV1
    ] = {}
    schema_entries: list[SchemaRegistrationV1] = []
    bundle_list = tuple(bundles)

    for bundle in bundle_list:
        identity, _request_model, _result_model = _validate_bundle(bundle)
        key = exact_capability_key(identity)
        if key in descriptors or key in handlers:
            raise ValueError("duplicate_live_capability_identity")
        descriptors[key] = bundle.descriptor
        handlers[key] = bundle.handler
        schema_entries.extend((bundle.request_schema, bundle.result_schema))

    if set(descriptors) != set(handlers):
        if set(descriptors) - set(handlers):
            raise ValueError("descriptor_without_handler")
        raise ValueError("handler_without_descriptor")

    schema_registry = SchemaRegistryV1(tuple(schema_entries))
    return PublishedLiveRegistrationV1(
        descriptors=MappingProxyType(dict(descriptors)),
        handlers=MappingProxyType(dict(handlers)),
        schemas=schema_registry,
    )


class VendorKnowledgeLiveRegistrationRegistry:
    """Provider-neutral source registration and publication boundary."""

    def __init__(
        self,
        *,
        plugin_registry: VendorKnowledgeSourcePluginRegistry | None = None,
    ) -> None:
        self._plugins = plugin_registry or VendorKnowledgeSourcePluginRegistry()
        self._bundles: dict[
            tuple[str, IntegrationCategory, str, str],
            LiveRegistrationBundleV1,
        ] = {}

    def register(self, bundles: Iterable[LiveRegistrationBundleV1]) -> None:
        """Register bundles atomically; identical registrations are idempotent."""
        bundle_list = tuple(bundles)
        validated: list[
            tuple[tuple[str, IntegrationCategory, str, str], LiveRegistrationBundleV1]
        ] = []
        for bundle in bundle_list:
            identity, _request_model, _result_model = _validate_bundle(bundle)
            validated.append((exact_capability_key(identity), bundle))

        pending: dict[
            tuple[str, IntegrationCategory, str, str],
            LiveRegistrationBundleV1,
        ] = {}
        for key, bundle in validated:
            existing = self._bundles.get(key) or pending.get(key)
            if existing is None:
                for candidate in (*self._bundles.values(), *pending.values()):
                    if (
                        candidate.descriptor.capability_id
                        == bundle.descriptor.capability_id
                        and candidate.descriptor.contract_version
                        == bundle.descriptor.contract_version
                        and not self._same_registration(candidate, bundle)
                    ):
                        raise ValueError("conflicting_live_capability_registration")
                pending[key] = bundle
                continue
            if not self._same_registration(existing, bundle):
                raise ValueError("conflicting_live_capability_registration")

        self._bundles.update(pending)
        for plugin in self._plugins.list_plugins():
            self._validate_plugin_live_capabilities(plugin)

    def register_plugin(self, plugin: VendorKnowledgeSourcePlugin) -> None:
        """Register a VK-2 plugin only when all declared LIVE refs resolve."""
        if not isinstance(plugin, VendorKnowledgeSourcePlugin):
            raise TypeError("source_plugin_invalid")
        self._validate_plugin_live_capabilities(plugin)
        self._plugins.register(plugin)

    def list_registrations(self) -> tuple[LiveRegistrationBundleV1, ...]:
        """Return registrations in deterministic identity order."""
        return tuple(
            self._bundles[key]
            for key in sorted(
                self._bundles,
                key=lambda item: (item[0], item[1].value, item[2], item[3]),
            )
        )

    def resolve_for_source(
        self,
        identity: VendorKnowledgeSourceIdentity,
    ) -> tuple[LiveRegistrationBundleV1, ...]:
        """Resolve exactly the LIVE registrations declared by one VK-2 source."""
        if not isinstance(identity, VendorKnowledgeSourceIdentity):
            raise TypeError("source_identity_required")
        plugin = self._plugins.require(identity)
        self._validate_plugin_live_capabilities(plugin)
        live = plugin.capability(VendorKnowledgeMode.LIVE)
        assert live is not None
        return tuple(
            self._bundles[
                self._key_for_identity(
                    identity,
                    capability_id=capability_id,
                )
            ]
            for capability_id in live.capability_refs
        )

    def publish(self) -> PublishedLiveRegistrationV1:
        """Publish one immutable snapshot for the canonical Live executor."""
        return publish_live_registration_bundles(self.list_registrations())

    def publish_for_source(
        self,
        identity: VendorKnowledgeSourceIdentity,
    ) -> PublishedLiveRegistrationV1:
        """Publish only the source subset declared by a VK-2 plugin."""
        return publish_live_registration_bundles(self.resolve_for_source(identity))

    def publish_to_tenant_catalog(self, catalog: object) -> tuple[LiveCapabilityDescriptorV1, ...]:
        """Publish provider-neutral descriptors into a tenant catalog boundary."""
        register = attribute_access.optional(catalog, "register", None)
        if not callable(register):
            raise TypeError("tenant_live_capability_catalog_not_registrable")
        descriptors = tuple(bundle.descriptor for bundle in self.list_registrations())
        for descriptor in descriptors:
            register(descriptor)
        return descriptors

    @staticmethod
    def _key_for_identity(
        identity: VendorKnowledgeSourceIdentity,
        *,
        capability_id: str,
    ) -> tuple[str, IntegrationCategory, str, str]:
        return (
            identity.provider_id,
            identity.integration_category,
            capability_id,
            LIVE_CONTRACT_VERSION,
        )

    def _validate_plugin_live_capabilities(
        self,
        plugin: VendorKnowledgeSourcePlugin,
    ) -> None:
        live = plugin.capability(VendorKnowledgeMode.LIVE)
        if live is None:
            raise ValueError("live_capability_not_declared")
        for capability_id in live.capability_refs:
            key = self._key_for_identity(plugin.identity, capability_id=capability_id)
            bundle = self._bundles.get(key)
            if bundle is None:
                raise LookupError("live_capability_registration_missing")
            descriptor = bundle.descriptor
            if (
                descriptor.provider_id != plugin.identity.provider_id
                or descriptor.integration_kind != plugin.identity.integration_category
                or descriptor.source_kind != plugin.identity.source_kind
                or descriptor.capability_id != capability_id
            ):
                raise ValueError("live_capability_source_mismatch")

    @staticmethod
    def _same_registration(
        left: LiveRegistrationBundleV1,
        right: LiveRegistrationBundleV1,
    ) -> bool:
        handler_fields = (
            "provider_id",
            "integration_kind",
            "source_kind",
            "capability_id",
            "contract_version",
            "request_schema_ref",
            "result_schema_ref",
            "expected_request_model",
        )
        return (
            left.descriptor == right.descriptor
            and left.request_schema == right.request_schema
            and left.result_schema == right.result_schema
            and type(left.handler) is type(right.handler)
            and all(
                attribute_access.optional(left.handler, field, object())
                == attribute_access.optional(right.handler, field, object())
                for field in handler_fields
            )
        )
