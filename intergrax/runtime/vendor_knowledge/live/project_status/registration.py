# © Artur Czarnecki. All rights reserved.

"""Project Status live capability registration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
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

from .project import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
    PROJECT_STATUS_READ_REQUEST_SCHEMA_REF,
    PROJECT_STATUS_READ_RESULT_SCHEMA_REF,
    ProjectStatusReadLiveHandlerV1,
    ProjectStatusReadLiveRequestV1,
    build_project_status_read_descriptor,
)


def build_project_status_live_registration_bundles() -> tuple[LiveRegistrationBundleV1, ...]:
    return (
        LiveRegistrationBundleV1(
            descriptor=build_project_status_read_descriptor(),
            handler=ProjectStatusReadLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=PROJECT_STATUS_READ_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=ProjectStatusReadLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=PROJECT_STATUS_READ_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
    )


def build_project_status_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    live_capability_refs = tuple(
        bundle.descriptor.capability_id
        for bundle in build_project_status_live_registration_bundles()
    )
    identity = VendorKnowledgeSourceIdentity(
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_category=IntegrationCategory.ISSUE_TRACKER,
        source_kind=PROJECT_STATUS_SOURCE_KIND,
    )
    return VendorKnowledgeSourcePlugin(
        identity=identity,
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.LIVE,
                contract_version="vendor-knowledge.live.v1",
                operations=("read",),
                runtime_ref="live-registration:project_status:project",
                capability_refs=live_capability_refs,
                constraints={"read_only": True, "bounded": True},
            ),
        ),
    )
