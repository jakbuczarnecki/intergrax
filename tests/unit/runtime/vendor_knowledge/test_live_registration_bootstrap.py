from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from local_workspace_application.workspaces.hybrid_ask_execution import (
    LiveCapabilityExecutorV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    ExecutableLiveCallV1,
    KnowledgeQueryAudienceV1,
    ResolvedLiveResourceScopeV1,
)
from pydantic import BaseModel, ConfigDict

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.live.contracts import (
    EffectiveLiveCallBudgetV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
    ValidatedLiveCapabilityCallV1,
    content_sha256,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    LiveRegistrationBundleV1,
    VendorKnowledgeLiveRegistrationRegistry,
)
from intergrax.runtime.vendor_knowledge.live.schemas import (
    SchemaRegistrationV1,
    SchemaRoleV1,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)

_PROVIDER = "fixture_provider"
_SOURCE = "fixture_source"
_CAPABILITY = f"vendor.{_PROVIDER}.{_SOURCE}.list"
_REQUEST_SCHEMA = (
    f"schema://vendor-knowledge/live/{_PROVIDER}/{_SOURCE}/list/request/v1"
)
_RESULT_SCHEMA = (
    f"schema://vendor-knowledge/live/{_PROVIDER}/{_SOURCE}/list/result/v1"
)
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FixtureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    query: str


class _FixtureHandler:
    provider_id = _PROVIDER
    integration_kind = IntegrationCategory.ISSUE_TRACKER
    source_kind = _SOURCE
    capability_id = _CAPABILITY
    contract_version = "1"
    request_schema_ref = _REQUEST_SCHEMA
    result_schema_ref = _RESULT_SCHEMA
    expected_request_model = _FixtureRequest

    async def execute(
        self,
        *,
        integration: object,
        call: ValidatedLiveCapabilityCallV1,
        context,
    ) -> LiveCapabilityExecutionResultV1:
        del integration
        content = "fixture evidence"
        item = LiveCapabilityResultItemV1(
            remote_item_id="fixture-item",
            safe_display_name="Fixture evidence",
            content=content,
            content_hash=content_sha256(content),
            retrieved_at=context.started_at,
        )
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
            items=(item,),
            item_count=1,
            byte_count=len(content.encode("utf-8")),
            started_at=context.started_at,
            completed_at=context.started_at,
        )


def _fixture_bundle() -> LiveRegistrationBundleV1:
    descriptor = LiveCapabilityDescriptorV1(
        capability_id=_CAPABILITY,
        provider_id=_PROVIDER,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=_SOURCE,
        contract_version="1",
        effect=CapabilityEffectV1.READ,
        read_only=True,
        resource_scope_required=False,
        request_schema_ref=_REQUEST_SCHEMA,
        result_schema_ref=_RESULT_SCHEMA,
        max_result_items=2,
        max_result_bytes=1024,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=2,
        max_provider_page_size=2,
        max_content_bytes_per_item=1024,
    )
    return LiveRegistrationBundleV1(
        descriptor=descriptor,
        handler=_FixtureHandler(),
        request_schema=SchemaRegistrationV1(
            schema_ref=_REQUEST_SCHEMA,
            role=SchemaRoleV1.REQUEST,
            model=_FixtureRequest,
            contract_version="1",
        ),
        result_schema=SchemaRegistrationV1(
            schema_ref=_RESULT_SCHEMA,
            role=SchemaRoleV1.RESULT,
            model=LiveCapabilityExecutionResultV1,
            contract_version="1",
        ),
    )


def _fixture_plugin() -> VendorKnowledgeSourcePlugin:
    return VendorKnowledgeSourcePlugin(
        identity=VendorKnowledgeSourceIdentity(
            provider_id=_PROVIDER,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            source_kind=_SOURCE,
        ),
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.LIVE,
                contract_version="vendor-knowledge.live.v1",
                operations=("list",),
                runtime_ref="live-registration:fixture_provider:fixture_source",
                capability_refs=(_CAPABILITY,),
                constraints={"read_only": True},
            ),
        ),
    )


class _RecordingCatalog:
    def __init__(self) -> None:
        self.descriptors: list[LiveCapabilityDescriptorV1] = []

    def register(self, descriptor: LiveCapabilityDescriptorV1) -> None:
        self.descriptors.append(descriptor)


class _FixtureIntegrationResolver:
    def resolve(self, **_: object) -> object:
        return object()


@pytest.mark.unit
def test_registry_is_idempotent_and_resolves_vk2_source_identity() -> None:
    registry = VendorKnowledgeLiveRegistrationRegistry()
    registry.register((_fixture_bundle(),))
    registry.register((_fixture_bundle(),))
    registry.register_plugin(_fixture_plugin())

    resolved = registry.resolve_for_source(_fixture_plugin().identity)
    assert tuple(item.descriptor.capability_id for item in resolved) == (_CAPABILITY,)
    assert registry.publish_for_source(_fixture_plugin().identity).handlers


@pytest.mark.unit
def test_registry_fails_closed_for_conflict_and_missing_plugin_registration() -> None:
    bundle = _fixture_bundle()
    registry = VendorKnowledgeLiveRegistrationRegistry()
    registry.register((bundle,))

    conflicting_descriptor = bundle.descriptor.model_copy(update={"max_result_items": 1})
    conflicting = LiveRegistrationBundleV1(
        descriptor=conflicting_descriptor,
        handler=bundle.handler,
        request_schema=bundle.request_schema,
        result_schema=bundle.result_schema,
    )
    with pytest.raises(ValueError, match="conflicting_live_capability_registration"):
        registry.register((conflicting,))

    with pytest.raises(LookupError, match="live_capability_registration_missing"):
        VendorKnowledgeLiveRegistrationRegistry().register_plugin(_fixture_plugin())


@pytest.mark.unit
def test_default_bootstrap_is_deterministic_and_publishes_tenant_descriptors() -> None:
    first = build_vendor_knowledge_live_registration_registry()
    second = build_vendor_knowledge_live_registration_registry()

    first_keys = tuple(item.descriptor.capability_id for item in first.list_registrations())
    second_keys = tuple(item.descriptor.capability_id for item in second.list_registrations())
    assert first_keys == second_keys
    assert len(first_keys) == 8

    catalog = _RecordingCatalog()
    published = first.publish_to_tenant_catalog(catalog)
    assert tuple(item.capability_id for item in published) == first_keys
    assert tuple(item.capability_id for item in catalog.descriptors) == first_keys


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fixture_provider_executes_through_existing_live_executor() -> None:
    registry = VendorKnowledgeLiveRegistrationRegistry()
    registry.register((_fixture_bundle(),))
    registry.register_plugin(_fixture_plugin())
    published = registry.publish()
    budget = EffectiveLiveCallBudgetV1(
        max_live_calls=1,
        max_total_duration_ms=30_000,
        max_result_items=2,
        max_result_bytes=1024,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=2,
        max_provider_page_size=2,
        max_content_bytes_per_item=1024,
    )
    call = ExecutableLiveCallV1(
        call_id="fixture-call",
        capability_id=_CAPABILITY,
        contract_version="1",
        connection_ref="fixture-connection",
        live_access_binding_id="fixture-binding",
        validated_request=_FixtureRequest(query="hello"),
        effective_budget=budget,
        provider_id=_PROVIDER,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=_SOURCE,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(),
    )

    result = await LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=_FixtureIntegrationResolver(),
        clock=lambda: _NOW,
    ).execute(
        run_id="fixture-run",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        call=call,
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.source_kind == _SOURCE
    assert result.receipt is None
    assert result.items[0].content == "fixture evidence"


@pytest.mark.unit
def test_lkw_route_uses_neutral_bootstrap_without_provider_live_imports() -> None:
    route = Path("applications/local_workspace_application/serving/workspace_routes.py")
    source = route.read_text(encoding="utf-8")

    assert "build_slack_live_registration_bundles" not in source
    assert "build_msgraph_live_registration_bundles" not in source
    assert "build_vendor_knowledge_live_registration_registry" in source
