# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3 configured relational execution unit proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.configured_relational_store_execution import (
    RelationalQueryRequest,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningStore,
)
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.execution_bound_configured_relational_store_port import (
    ExecutionBoundConfiguredRelationalStorePort,
)
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationMaterializationPort,
    ExecutionBoundIntegrationResolution,
)
from intergrax.integrations.invocation_bound_configured_relational_wiring_resolver import (
    InvocationBoundConfiguredRelationalStoreWiringResolver,
)
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.tools.invocation_wiring import ToolInvocationContext, ToolRegistrationWiringView


@dataclass
class _RecordingPinningStore(ExecutionIntegrationConfigurationPinningStore):
    pins: list[tuple[object, object]] = field(default_factory=list)

    def pin(self, *, subject, provenance) -> None:
        self.pins.append((subject, provenance))


class _FakeRelationalClient:
    def __init__(self, token: object) -> None:
        self.token = token
        self.fetch_calls = 0

    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        return None

    def fetch_all(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> Sequence[Mapping[str, Any]]:
        self.fetch_calls += 1
        return ({"v": 1},)

    def close(self) -> None:
        return None


class _FakeRelationalIntegration(RelationalStoreIntegrationContract):
    def __init__(self, token: object) -> None:
        super().__init__(
            **RelationalStoreIntegrationContract.for_provider(
                provider_id="sqlite",
                display_name="Fake Sqlite",
            ).model_dump(),
        )
        self._client = _FakeRelationalClient(token)

    def connect(self) -> None:
        self._client.connect()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._client.execute(sql, params)

    def fetch_all(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> Sequence[Mapping[str, Any]]:
        return self._client.fetch_all(sql, params)

    def close(self) -> None:
        self._client.close()


class _CountingMaterialization(ExecutionBoundIntegrationMaterializationPort):
    def __init__(self, instance: RelationalStoreIntegrationContract) -> None:
        self.count = 0
        self.instance = instance
        self.last_materialized: RelationalStoreIntegrationContract | None = None

    def resolve_catalog(
        self,
        category: IntegrationCategory,
        *,
        slug: str,
        profile=None,
    ) -> RelationalStoreIntegrationContract:
        self.count += 1
        self.last_materialized = self.instance
        return self.instance

    def resolve_from_profile(self, profile, category: IntegrationCategory):
        self.count += 1
        self.last_materialized = self.instance
        return self.instance


def _adoption(tenant: str = "t1") -> ExecutionIntegrationConfigurationAdoption:
    binding = ConfiguredCapabilityBinding(
        tenant_id=tenant,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        configuration_type="test",
        configuration_version="v1",
        configuration_fingerprint="fp",
        realization_evidence_refs=(),
    )
    return ExecutionIntegrationConfigurationAdoption(
        configured_binding=binding,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        resource_scope="default",
    )


def test_lazy_port_materializes_once_and_reuses_same_provider() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    pinning = _RecordingPinningStore()
    resolution = ExecutionBoundIntegrationResolution(
        pinning_store=pinning,
        materialization=materialization,
    )
    execution_id = mint_execution_id()
    port = ExecutionBoundConfiguredRelationalStorePort(
        tenant_id="t1",
        execution_id=execution_id,
        adoption=_adoption(),
        resolution=resolution,
        catalog_slug="sqlite",
    )
    resolver = InvocationBoundConfiguredRelationalStoreWiringResolver(
        configured_relational_store_execution=port,
    )
    wiring1 = resolver.resolve(
        tool_id="database.query",
        invocation_context=ToolInvocationContext(run_id="r", step_id="s", tool_id="database.query"),
        registration_wiring=ToolRegistrationWiringView(),
    )
    wiring2 = resolver.resolve(
        tool_id="database.query",
        invocation_context=ToolInvocationContext(run_id="r", step_id="s", tool_id="database.query"),
        registration_wiring=ToolRegistrationWiringView(),
    )
    assert (
        wiring1.configured_relational_store_execution
        is wiring2.configured_relational_store_execution
    )
    assert materialization.count == 0
    result = port.query(RelationalQueryRequest(sql="SELECT 1"))
    assert result.row_count == 1
    assert materialization.count == 1
    assert materialization.last_materialized is integration
    port.query(RelationalQueryRequest(sql="SELECT 2"))
    assert materialization.count == 1
    assert isinstance(materialization.last_materialized, RelationalStore)
    assert materialization.last_materialized is integration


def test_tenant_mismatch_fails_before_materialization() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    pinning = _RecordingPinningStore()
    resolution = ExecutionBoundIntegrationResolution(
        pinning_store=pinning,
        materialization=materialization,
    )
    port = ExecutionBoundConfiguredRelationalStorePort(
        tenant_id="wrong-tenant",
        execution_id=mint_execution_id(),
        adoption=_adoption(tenant="t1"),
        resolution=resolution,
    )
    with pytest.raises(Exception):
        port.query(RelationalQueryRequest(sql="SELECT 1"))
    assert materialization.count == 0
    assert pinning.pins == []
