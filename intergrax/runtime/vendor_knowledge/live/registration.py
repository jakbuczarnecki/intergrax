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
    exact_capability_key,
    validate_capability_identity,
)
from intergrax.runtime.vendor_knowledge.live.schemas import (
    SchemaRegistrationV1,
    SchemaRegistryV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)

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
