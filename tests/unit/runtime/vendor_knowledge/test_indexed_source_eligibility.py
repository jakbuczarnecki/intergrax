# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.vendor_knowledge import (
    IndexedSourceDescriptorV1,
    IndexedSourceEligibilityRequestV1,
    IndexedSourceEligibilityResolverV1,
    IndexedSourceEligibilityStatusV1,
    IndexedSourceMaterializationRegistry,
    RemoteResourceAvailabilityV1,
    RemoteResourceDescriptorV1,
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
    canonical_indexed_source_ref,
)
from intergrax.runtime.vendor_knowledge.sync_task import (
    VendorKnowledgeSyncDispatcher,
    VendorKnowledgeSyncHandlerRegistry,
    register_vendor_knowledge_sync_handler,
    unregister_vendor_knowledge_sync_handler,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def _connection(
    *,
    tenant_id: str = "tenant-a",
    connection_ref: str = "connection-a",
    provider_id: str = "provider-a",
    integration_kind: IntegrationCategory = IntegrationCategory.ISSUE_TRACKER,
    status: TenantConnectionAdministrativeStatus = TenantConnectionAdministrativeStatus.ACTIVE,
) -> SafeTenantConnectionV1:
    return SafeTenantConnectionV1(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        provider_id=provider_id,
        integration_kind=integration_kind,
        safe_display_name="Synthetic connection",
        administrative_status=status,
        configuration_version=1,
        connected_principal_ref=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _resource(
    connection: SafeTenantConnectionV1,
    *,
    remote_resource_id: str = "resource-a",
    source_kind: str = "issues",
    availability: RemoteResourceAvailabilityV1 = RemoteResourceAvailabilityV1.AVAILABLE,
    snapshot_version: str = "snapshot-1",
) -> RemoteResourceDescriptorV1:
    return RemoteResourceDescriptorV1(
        connection_ref=connection.connection_ref,
        remote_resource_id=remote_resource_id,
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind,
        source_kind=source_kind,
        resource_type="project",
        safe_display_label="Synthetic project",
        safe_description="Safe synthetic resource",
        availability=availability,
        supported_capability_ids=("live.read",),
        discovered_at=_NOW,
        snapshot_version=snapshot_version,
    )


class _Discovery:
    def __init__(self, resources: tuple[RemoteResourceDescriptorV1, ...]) -> None:
        self.resources = resources

    async def list_remote_resources(self, **_: object):
        from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
            RemoteResourceDiscoveryPageV1,
        )

        snapshot = self.resources[0].snapshot_version if self.resources else "snapshot-1"
        return RemoteResourceDiscoveryPageV1(
            resources=self.resources,
            snapshot_version=snapshot,
        )


class _Connections:
    def __init__(self, *connections: SafeTenantConnectionV1) -> None:
        self.connections = {
            (connection.tenant_id, connection.connection_ref): connection
            for connection in connections
        }

    def get_connection(self, *, tenant_id: str, connection_ref: str):
        return self.connections.get((tenant_id, connection_ref))


class _Provider:
    def __init__(
        self,
        connection: SafeTenantConnectionV1,
        *,
        source_kind: str,
        handler_ref: str = "synthetic.sync.v1",
        handler_available: bool = True,
        contract_version: str = "synthetic.materialization.v1",
        invalid_descriptor: bool = False,
    ) -> None:
        self._connection = connection
        self._source_kind = source_kind
        self._handler_ref = handler_ref
        self._handler_available = handler_available
        self._contract_version = contract_version
        self._invalid_descriptor = invalid_descriptor

    @property
    def provider_id(self) -> str:
        return self._connection.provider_id

    @property
    def integration_kind(self) -> IntegrationCategory:
        return self._connection.integration_kind

    @property
    def source_kind(self) -> str:
        return self._source_kind

    @property
    def materialization_contract_version(self) -> str:
        return self._contract_version

    def qualify(self, *, connection, resource) -> IndexedSourceDescriptorV1:
        canonical_ref = canonical_indexed_source_ref(
            tenant_id=connection.tenant_id,
            connection_ref=connection.connection_ref,
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            remote_resource_id=resource.remote_resource_id,
            source_kind=resource.source_kind,
        )
        if self._invalid_descriptor:
            canonical_ref = "vksrc:" + "0" * 64
        return IndexedSourceDescriptorV1(
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            connection_ref=connection.connection_ref,
            remote_resource_id=resource.remote_resource_id,
            source_kind=resource.source_kind,
            canonical_source_ref=canonical_ref,
            safe_display_label=resource.safe_display_label,
            safe_description=resource.safe_description,
            resource_type=resource.resource_type,
            discovery_snapshot_version=resource.snapshot_version,
            materialization_contract_version=self._contract_version,
        )

    def sync_handler_ref(self) -> str:
        return self._handler_ref

    def sync_handler_available(self) -> bool:
        return self._handler_available


class _NonCallableHandlerProvider(_Provider):
    handler_for_registry = object()

    def sync_handler_ref(self) -> str:
        return "not-callable"


def _request(
    *,
    tenant_id: str = "tenant-a",
    connection_ref: str = "connection-a",
    remote_resource_id: str = "resource-a",
    source_kind: str = "issues",
    snapshot_version: str = "snapshot-1",
) -> IndexedSourceEligibilityRequestV1:
    return IndexedSourceEligibilityRequestV1(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        remote_resource_id=remote_resource_id,
        source_kind=source_kind,
        discovery_snapshot_version=snapshot_version,
    )


def _resolver(
    *,
    connections: tuple[SafeTenantConnectionV1, ...],
    discovery: _Discovery,
    providers: tuple[_Provider, ...] = (),
    sync_handler_registry: VendorKnowledgeSyncHandlerRegistry | None = None,
    now: datetime = _NOW,
) -> IndexedSourceEligibilityResolverV1:
    registry = IndexedSourceMaterializationRegistry()
    for provider in providers:
        registry.register(provider)
    handler_registry = sync_handler_registry or VendorKnowledgeSyncHandlerRegistry()
    if sync_handler_registry is None:
        def shared_handler() -> None:
            return None

        for provider in providers:
            handler_registry.register(
                provider_id=provider.provider_id,
                integration_kind=provider.integration_kind,
                source_kind=provider.source_kind,
                handler_ref=provider.sync_handler_ref(),
                handler=shared_handler,
                registration_version="registration-1",
            )
    return IndexedSourceEligibilityResolverV1(
        connection_port=_Connections(*connections),
        discovery_service_factory=lambda _tenant_id: discovery,
        materialization_registry=registry,
        sync_handler_availability_port=handler_registry,
        clock=lambda: now,
    )


@pytest.mark.asyncio
async def test_two_neutral_plugins_produce_same_canonical_plan_shape() -> None:
    first_connection = _connection()
    second_connection = _connection(
        connection_ref="connection-b",
        provider_id="provider-b",
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
    )
    first_provider = _Provider(first_connection, source_kind="issues")
    second_provider = _Provider(second_connection, source_kind="documents")
    shared_handler_registry = VendorKnowledgeSyncHandlerRegistry()

    def shared_handler() -> None:
        return None

    shared_handler_registry.register(
        provider_id=first_provider.provider_id,
        integration_kind=first_provider.integration_kind,
        source_kind=first_provider.source_kind,
        handler_ref=first_provider.sync_handler_ref(),
        handler=shared_handler,
    )
    shared_handler_registry.register(
        provider_id=second_provider.provider_id,
        integration_kind=second_provider.integration_kind,
        source_kind=second_provider.source_kind,
        handler_ref=second_provider.sync_handler_ref(),
        handler=shared_handler,
    )
    first = _resolver(
        connections=(first_connection, second_connection),
        discovery=_Discovery((_resource(first_connection),)),
        providers=(first_provider,),
        sync_handler_registry=shared_handler_registry,
    )
    second = _resolver(
        connections=(first_connection, second_connection),
        discovery=_Discovery(
            (
                _resource(
                    second_connection,
                    remote_resource_id="resource-b",
                    source_kind="documents",
                ),
            )
        ),
        providers=(second_provider,),
        sync_handler_registry=shared_handler_registry,
    )

    first_proof = await first.resolve(_request())
    second_proof = await second.resolve(
        _request(
            connection_ref="connection-b",
            remote_resource_id="resource-b",
            source_kind="documents",
        )
    )

    assert first_proof.status is IndexedSourceEligibilityStatusV1.ELIGIBLE
    assert second_proof.status is IndexedSourceEligibilityStatusV1.ELIGIBLE
    assert first_proof.binding_plan is not None
    assert second_proof.binding_plan is not None
    assert first_proof.binding_plan.source_descriptor.provider_id == "provider-a"
    assert second_proof.binding_plan.source_descriptor.provider_id == "provider-b"
    assert "credential_ref" not in first_proof.model_dump()
    assert "credential_ref" not in second_proof.model_dump()


@pytest.mark.asyncio
async def test_qualification_is_deterministic_and_proof_expires() -> None:
    connection = _connection()
    resolver = _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues"),),
    )

    first = await resolver.resolve(_request())
    second = await resolver.resolve(_request())

    assert first.proof_revision == second.proof_revision
    assert first.binding_plan == second.binding_plan
    assert first.is_current(_NOW + timedelta(seconds=299))
    assert not first.is_current(_NOW + timedelta(seconds=300))
    changed_snapshot = await _resolver(
            connections=(connection,),
            discovery=_Discovery((_resource(connection, snapshot_version="snapshot-2"),)),
            providers=(_Provider(connection, source_kind="issues"),),
        ).resolve(_request(snapshot_version="snapshot-2"))
    assert first.proof_revision != changed_snapshot.proof_revision

    def handler() -> None:
        return None

    version_one_registry = VendorKnowledgeSyncHandlerRegistry()
    version_one_registry.register(
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
        handler=handler,
        registration_version="registration-1",
    )
    version_two_registry = VendorKnowledgeSyncHandlerRegistry()
    version_two_registry.register(
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
        handler=handler,
        registration_version="registration-2",
    )
    version_one = await _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues"),),
        sync_handler_registry=version_one_registry,
    ).resolve(_request())
    version_two = await _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues"),),
        sync_handler_registry=version_two_registry,
    ).resolve(_request())
    assert version_one.proof_revision != version_two.proof_revision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection", "resource", "provider", "req_payload", "status", "reason"),
    (
        (
            _connection(status=TenantConnectionAdministrativeStatus.DISABLED),
            None,
            None,
            _request(),
            IndexedSourceEligibilityStatusV1.CONNECTION_INACTIVE,
            "indexed_source_eligibility_connection_not_active",
        ),
        (
            _connection(),
            _resource(
                _connection(),
                availability=RemoteResourceAvailabilityV1.PERMISSION_DENIED,
            ),
            _Provider(_connection(), source_kind="issues"),
            _request(),
            IndexedSourceEligibilityStatusV1.RESOURCE_UNAVAILABLE,
            "indexed_source_eligibility_resource_unavailable",
        ),
        (
            _connection(),
            _resource(_connection()),
            None,
            _request(),
            IndexedSourceEligibilityStatusV1.NOT_SUPPORTED,
            "indexed_source_eligibility_materialization_not_supported",
        ),
    ),
)
async def test_negative_qualification_matrix(
    connection: SafeTenantConnectionV1,
    resource: RemoteResourceDescriptorV1 | None,
    provider: _Provider | None,
    req_payload: IndexedSourceEligibilityRequestV1,
    status: IndexedSourceEligibilityStatusV1,
    reason: str,
) -> None:
    resources = () if resource is None else (resource,)
    resolver = _resolver(
        connections=(connection,),
        discovery=_Discovery(resources),
        providers=() if provider is None else (provider,),
    )

    proof = await resolver.resolve(req_payload)

    assert proof.status is status
    assert proof.eligible is False
    assert proof.binding_plan is None
    assert proof.safe_reason_code == reason


@pytest.mark.asyncio
async def test_stale_snapshot_and_invalid_descriptor_are_rejected() -> None:
    connection = _connection()
    stale = _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection, snapshot_version="snapshot-2"),)),
        providers=(_Provider(connection, source_kind="issues"),),
    )
    invalid = _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues", invalid_descriptor=True),),
    )

    stale_proof = await stale.resolve(_request())
    invalid_proof = await invalid.resolve(_request())

    assert stale_proof.status is IndexedSourceEligibilityStatusV1.SNAPSHOT_STALE
    assert stale_proof.safe_reason_code == "indexed_source_eligibility_snapshot_stale"
    assert invalid_proof.status is IndexedSourceEligibilityStatusV1.NOT_SUPPORTED
    assert invalid_proof.safe_reason_code == "indexed_source_eligibility_invalid_provider_response"


@pytest.mark.asyncio
async def test_cross_tenant_missing_resource_and_non_callable_handler_fail_closed() -> None:
    connection = _connection()
    cross_tenant = _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues"),),
    )
    missing_resource = _resolver(
        connections=(connection,),
        discovery=_Discovery(()),
        providers=(_Provider(connection, source_kind="issues"),),
    )
    non_callable_registry = VendorKnowledgeSyncHandlerRegistry()
    with pytest.raises(ValueError, match="handler_not_executable"):
        non_callable_registry.register(
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            source_kind="issues",
            handler_ref="not-callable",
            handler=object(),
        )
    non_callable_handler = _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_NonCallableHandlerProvider(connection, source_kind="issues"),),
        sync_handler_registry=non_callable_registry,
    )

    cross_tenant_proof = await cross_tenant.resolve(_request(tenant_id="tenant-b"))
    missing_resource_proof = await missing_resource.resolve(
        _request(remote_resource_id="resource-missing")
    )
    non_callable_proof = await non_callable_handler.resolve(_request())

    assert cross_tenant_proof.status is IndexedSourceEligibilityStatusV1.CONNECTION_INACTIVE
    assert cross_tenant_proof.safe_reason_code == (
        "indexed_source_eligibility_connection_not_found"
    )
    assert missing_resource_proof.status is IndexedSourceEligibilityStatusV1.RESOURCE_UNAVAILABLE
    assert non_callable_proof.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE


@pytest.mark.asyncio
async def test_declared_but_unregistered_handler_is_unavailable() -> None:
    connection = _connection()
    provider = _Provider(connection, source_kind="issues", handler_available=True)
    proof = await _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(provider,),
        sync_handler_registry=VendorKnowledgeSyncHandlerRegistry(),
    ).resolve(_request())

    assert proof.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE
    assert proof.safe_reason_code == "indexed_source_eligibility_handler_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "registered_provider",
        "registered_integration",
        "registered_source",
        "registered_handler_ref",
    ),
    (
        ("provider-b", IntegrationCategory.ISSUE_TRACKER, "issues", "synthetic.sync.v1"),
        ("provider-a", IntegrationCategory.COLLABORATION_SUITE, "issues", "synthetic.sync.v1"),
        ("provider-a", IntegrationCategory.ISSUE_TRACKER, "documents", "synthetic.sync.v1"),
        ("provider-a", IntegrationCategory.ISSUE_TRACKER, "issues", "other.sync.v1"),
    ),
)
async def test_handler_registration_dimensions_must_match(
    registered_provider: str,
    registered_integration: IntegrationCategory,
    registered_source: str,
    registered_handler_ref: str,
) -> None:
    connection = _connection()
    handler_registry = VendorKnowledgeSyncHandlerRegistry()
    handler_registry.register(
        provider_id=registered_provider,
        integration_kind=registered_integration,
        source_kind=registered_source,
        handler_ref=registered_handler_ref,
        handler=lambda: None,
    )

    proof = await _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues"),),
        sync_handler_registry=handler_registry,
    ).resolve(_request())

    assert proof.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE
    assert proof.safe_reason_code == "indexed_source_eligibility_handler_unavailable"


@pytest.mark.asyncio
async def test_handler_removal_invalidates_next_resolution() -> None:
    connection = _connection()
    handler_registry = VendorKnowledgeSyncHandlerRegistry()
    handler_registry.register(
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
        handler=lambda: None,
    )
    resolver = _resolver(
        connections=(connection,),
        discovery=_Discovery((_resource(connection),)),
        providers=(_Provider(connection, source_kind="issues"),),
        sync_handler_registry=handler_registry,
    )

    eligible = await resolver.resolve(_request())
    assert eligible.status is IndexedSourceEligibilityStatusV1.ELIGIBLE
    assert handler_registry.unregister(
        provider_id=connection.provider_id,
        integration_kind=connection.integration_kind,
        source_kind="issues",
        handler_ref="synthetic.sync.v1",
    )

    unavailable = await resolver.resolve(_request())
    assert unavailable.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE


@pytest.mark.asyncio
async def test_canonical_registration_is_the_eligibility_source_of_truth() -> None:
    connection = _connection()
    provider = _Provider(connection, source_kind="issues")
    materialization_registry = IndexedSourceMaterializationRegistry()
    materialization_registry.register(provider)
    task_registry = TaskExecutionRegistry()
    handler_registry = VendorKnowledgeSyncHandlerRegistry(task_registry)
    resolver = IndexedSourceEligibilityResolverV1(
        connection_port=_Connections(connection),
        discovery_service_factory=lambda _tenant_id: _Discovery((_resource(connection),)),
        materialization_registry=materialization_registry,
        sync_handler_availability_port=handler_registry,
        clock=lambda: _NOW,
    )

    before = await resolver.resolve(_request())
    assert before.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE

    register_vendor_knowledge_sync_handler(
        task_registry,
        lambda _tenant_id, _owner_id: object(),  # handler is not invoked here
        VendorKnowledgeSyncDispatcher(DocumentStoreTaskQueue(InMemoryDocumentStore())),
        handler_registry=handler_registry,
        provider_id=provider.provider_id,
        integration_kind=provider.integration_kind,
        source_kind=provider.source_kind,
        handler_ref=provider.sync_handler_ref(),
        registration_version="registration-1",
        retry_delays_seconds=(),
        sleeper=lambda _: None,
    )
    registered_handler = task_registry.get_handler("vendor_knowledge.sync.v1")
    after = await resolver.resolve(_request())
    assert after.status is IndexedSourceEligibilityStatusV1.ELIGIBLE
    assert after.binding_plan is not None
    assert handler_registry.resolve_registration(
        provider_id=provider.provider_id,
        integration_kind=provider.integration_kind,
        source_kind=provider.source_kind,
        handler_ref=provider.sync_handler_ref(),
    ).handler is registered_handler
    assert handler_registry.disable(
        provider_id=provider.provider_id,
        integration_kind=provider.integration_kind,
        source_kind=provider.source_kind,
        handler_ref=provider.sync_handler_ref(),
    )
    after_disable = await resolver.resolve(_request())
    assert after_disable.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE

    assert unregister_vendor_knowledge_sync_handler(
        handler_registry,
        provider_id=provider.provider_id,
        integration_kind=provider.integration_kind,
        source_kind=provider.source_kind,
        handler_ref=provider.sync_handler_ref(),
    )
    with pytest.raises(ValueError, match="not registered"):
        task_registry.get_handler("vendor_knowledge.sync.v1")
    after_removal = await resolver.resolve(_request())
    assert after_removal.status is IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE


def test_registry_is_complete_deterministic_and_removable() -> None:
    connection = _connection()
    registry = IndexedSourceMaterializationRegistry()
    provider = _Provider(connection, source_kind="issues")

    registry.register(provider)
    assert registry.registered_keys() == (
        ("provider-a", IntegrationCategory.ISSUE_TRACKER, "issues"),
    )
    with pytest.raises(ValueError, match="already_registered"):
        registry.register(provider)
    assert registry.unregister(
        provider_id="provider-a",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
    )
    assert registry.registered_keys() == ()


def test_partial_registration_is_rejected() -> None:
    class _DescriptorOnly:
        provider_id = "provider-a"
        integration_kind = IntegrationCategory.ISSUE_TRACKER
        source_kind = "issues"
        materialization_contract_version = "synthetic.materialization.v1"

        def qualify(self, *, connection, resource):
            return None

    with pytest.raises(ValueError, match="handler_registration_incomplete"):
        IndexedSourceMaterializationRegistry().register(_DescriptorOnly())  # type: ignore[arg-type]

    class _HandlerOnly:
        provider_id = "provider-a"
        integration_kind = IntegrationCategory.ISSUE_TRACKER
        source_kind = "issues"
        materialization_contract_version = "synthetic.materialization.v1"
        sync_handler_ref = "synthetic.sync.v1"
        sync_handler_available = staticmethod(lambda: True)

    with pytest.raises(ValueError, match="qualifier_required"):
        IndexedSourceMaterializationRegistry().register(_HandlerOnly())  # type: ignore[arg-type]
