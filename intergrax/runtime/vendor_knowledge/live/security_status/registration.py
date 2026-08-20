# © Artur Czarnecki. All rights reserved.

"""Security Status live capability registration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
    SECURITY_STATUS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.live import LiveCapabilityExecutionResultV1
from intergrax.runtime.vendor_knowledge.live.registration import LiveRegistrationBundleV1
from intergrax.runtime.vendor_knowledge.live.schemas import SchemaRegistrationV1, SchemaRoleV1
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)

from .security import (
    SECURITY_STATUS_READ_CAPABILITY_ID,
    SECURITY_STATUS_READ_REQUEST_SCHEMA_REF,
    SECURITY_STATUS_READ_RESULT_SCHEMA_REF,
    SecurityStatusReadLiveHandlerV1,
    SecurityStatusReadLiveRequestV1,
    build_security_status_read_descriptor,
)


def build_security_status_live_registration_bundles() -> tuple[LiveRegistrationBundleV1, ...]:
    return (
        LiveRegistrationBundleV1(
            descriptor=build_security_status_read_descriptor(),
            handler=SecurityStatusReadLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=SECURITY_STATUS_READ_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=SecurityStatusReadLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=SECURITY_STATUS_READ_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
    )


def build_security_status_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    live_capability_refs = tuple(
        bundle.descriptor.capability_id
        for bundle in build_security_status_live_registration_bundles()
    )
    identity = VendorKnowledgeSourceIdentity(
        provider_id=SECURITY_STATUS_PROVIDER_ID,
        integration_category=IntegrationCategory.SECURITY_SCANNER,
        source_kind=SECURITY_STATUS_SOURCE_KIND,
    )
    return VendorKnowledgeSourcePlugin(
        identity=identity,
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.LIVE,
                contract_version="vendor-knowledge.live.v1",
                operations=("read",),
                runtime_ref="live-registration:security_status:security",
                capability_refs=live_capability_refs,
                constraints={"read_only": True, "bounded": True},
            ),
        ),
    )
