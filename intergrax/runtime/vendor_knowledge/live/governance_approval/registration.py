# © Artur Czarnecki. All rights reserved.

"""Governance Approval live capability registration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
    GOVERNANCE_APPROVAL_SOURCE_KIND,
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

from .approval import (
    GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
    GOVERNANCE_APPROVAL_READ_REQUEST_SCHEMA_REF,
    GOVERNANCE_APPROVAL_READ_RESULT_SCHEMA_REF,
    GovernanceApprovalReadLiveHandlerV1,
    GovernanceApprovalReadLiveRequestV1,
    build_governance_approval_read_descriptor,
)


def build_governance_approval_live_registration_bundles() -> tuple[
    LiveRegistrationBundleV1, ...
]:
    return (
        LiveRegistrationBundleV1(
            descriptor=build_governance_approval_read_descriptor(),
            handler=GovernanceApprovalReadLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=GOVERNANCE_APPROVAL_READ_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=GovernanceApprovalReadLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=GOVERNANCE_APPROVAL_READ_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
    )


def build_governance_approval_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    live_capability_refs = tuple(
        bundle.descriptor.capability_id
        for bundle in build_governance_approval_live_registration_bundles()
    )
    identity = VendorKnowledgeSourceIdentity(
        provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
        integration_category=IntegrationCategory.WORKFLOW_ORCHESTRATOR,
        source_kind=GOVERNANCE_APPROVAL_SOURCE_KIND,
    )
    return VendorKnowledgeSourcePlugin(
        identity=identity,
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.LIVE,
                contract_version="vendor-knowledge.live.v1",
                operations=("read",),
                runtime_ref="live-registration:governance_approval:approval",
                capability_refs=live_capability_refs,
                constraints={"read_only": True, "bounded": True},
            ),
        ),
    )
