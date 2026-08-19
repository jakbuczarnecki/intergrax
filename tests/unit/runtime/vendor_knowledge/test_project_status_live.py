# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.bundle import create_project_status_integration
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.live import (
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    LiveCapabilityExecutionContextV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
)
from intergrax.runtime.vendor_knowledge.live.project_status import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
    ProjectStatusReadLiveHandlerV1,
    ProjectStatusReadLiveRequestV1,
    build_project_status_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.live.registration import publish_live_registration_bundles
from local_workspace_application.workspaces.hybrid_ask_execution import LiveCapabilityExecutorV1
from local_workspace_application.workspaces.hybrid_ask_policy import (
    ExecutableLiveCallV1,
    ResolvedLiveResourceScopeV1,
)
from proof_infrastructure.controlled_project_status_service.lifecycle import (
    ControlledProjectStatusServer,
)
from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
_TENANT = "tenant-proof"
_CONNECTION = "conn.project-status"


@pytest.fixture
def project_status_server() -> ControlledProjectStatusServer:
    server = ControlledProjectStatusServer.start()
    yield server
    server.stop()


def _budget() -> EffectiveLiveCallBudgetV1:
    return EffectiveLiveCallBudgetV1(
        max_live_calls=1,
        max_total_duration_ms=30_000,
        max_result_items=1,
        max_result_bytes=65_536,
        max_provider_pages=1,
        max_provider_requests=1,
        max_upstream_items=1,
        max_provider_page_size=1,
        max_content_bytes_per_item=16_384,
    )


def _executable_call() -> ExecutableLiveCallV1:
    return ExecutableLiveCallV1(
        call_id="project-status-call",
        capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
        contract_version="1",
        connection_ref=_CONNECTION,
        live_access_binding_id="binding-project-status",
        validated_request=ProjectStatusReadLiveRequestV1(
            project_id=ORION_FIXTURE_PROJECT_ID,
        ),
        effective_budget=_budget(),
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=PROJECT_STATUS_SOURCE_KIND,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(),
    )


class _ConnectionResolver:
    def __init__(self, integration: object) -> None:
        self._integration = integration
        self.calls: list[tuple[str, str, str, IntegrationCategory]] = []

    def resolve(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> object:
        self.calls.append((tenant_id, connection_ref, provider_id, integration_kind))
        return self._integration


async def test_handler_reaches_real_http_service_and_normalizes_item(
    project_status_server: ControlledProjectStatusServer,
) -> None:
    project_status_server.store.reset_read_request_count()
    integration = create_project_status_integration(
        base_url=project_status_server.base_url,
    )
    handler = ProjectStatusReadLiveHandlerV1()
    call = ExecutableLiveCallV1(
        call_id="handler-call",
        capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
        contract_version="1",
        connection_ref=_CONNECTION,
        live_access_binding_id="binding-project-status",
        validated_request=ProjectStatusReadLiveRequestV1(
            project_id=ORION_FIXTURE_PROJECT_ID,
        ),
        effective_budget=_budget(),
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=PROJECT_STATUS_SOURCE_KIND,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(),
    )
    context = LiveCapabilityExecutionContextV1(
        run_id="run-handler",
        tenant_id=_TENANT,
        workspace_id="workspace-proof",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=999999.0,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )

    result = await handler.execute(
        integration=integration,
        call=call,
        context=context,
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert len(result.items) == 1
    payload = json.loads(result.items[0].content)
    assert payload["project_id"] == ORION_FIXTURE_PROJECT_ID
    assert payload["blockers"][0]["status"] == ProjectBlockerStatusV1.OPEN.value
    count = httpx.get(
        f"{project_status_server.base_url}/control/request-count",
        timeout=2.0,
    ).json()["read_request_count"]
    assert count == 1


async def test_live_capability_executor_uses_resolver_and_registered_handler(
    project_status_server: ControlledProjectStatusServer,
) -> None:
    project_status_server.store.reset_read_request_count()
    integration = create_project_status_integration(
        base_url=project_status_server.base_url,
    )
    resolver = _ConnectionResolver(integration)
    published = publish_live_registration_bundles(
        build_project_status_live_registration_bundles()
    )
    executor = LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=resolver,
        clock=lambda: _NOW,
    )

    result = await executor.execute(
        run_id="run-executor",
        tenant_id=_TENANT,
        workspace_id="workspace-proof",
        call=_executable_call(),
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.source_kind == PROJECT_STATUS_SOURCE_KIND
    assert result.receipt is None
    assert resolver.calls == [
        (_TENANT, _CONNECTION, PROJECT_STATUS_PROVIDER_ID, IntegrationCategory.ISSUE_TRACKER)
    ]
    payload = json.loads(result.items[0].content)
    assert payload["blockers"][0]["id"] == ORION_FIXTURE_BLOCKER_ID
    assert payload["blockers"][0]["status"] == ProjectBlockerStatusV1.OPEN.value
    assert httpx.get(
        f"{project_status_server.base_url}/control/request-count",
        timeout=2.0,
    ).json()["read_request_count"] == 1


async def test_state_change_open_to_closed_without_provider_reconfiguration(
    project_status_server: ControlledProjectStatusServer,
) -> None:
    integration = create_project_status_integration(
        base_url=project_status_server.base_url,
    )
    resolver = _ConnectionResolver(integration)
    published = publish_live_registration_bundles(
        build_project_status_live_registration_bundles()
    )
    executor = LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=resolver,
        clock=lambda: _NOW,
    )

    first = await executor.execute(
        run_id="run-open",
        tenant_id=_TENANT,
        workspace_id="workspace-proof",
        call=_executable_call(),
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )
    open_payload = json.loads(first.items[0].content)
    assert open_payload["blockers"][0]["status"] == ProjectBlockerStatusV1.OPEN.value

    control = httpx.put(
        f"{project_status_server.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
        timeout=2.0,
    )
    assert control.status_code == 200

    project_status_server.store.reset_read_request_count()
    second = await executor.execute(
        run_id="run-closed",
        tenant_id=_TENANT,
        workspace_id="workspace-proof",
        call=_executable_call(),
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )
    closed_payload = json.loads(second.items[0].content)
    assert closed_payload["blockers"][0]["status"] == ProjectBlockerStatusV1.CLOSED.value
    assert httpx.get(
        f"{project_status_server.base_url}/control/request-count",
        timeout=2.0,
    ).json()["read_request_count"] == 1


async def test_not_found_maps_to_live_provider_not_found(
    project_status_server: ControlledProjectStatusServer,
) -> None:
    integration = create_project_status_integration(
        base_url=project_status_server.base_url,
    )
    handler = ProjectStatusReadLiveHandlerV1()
    call = ExecutableLiveCallV1(
        call_id="missing-project",
        capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
        contract_version="1",
        connection_ref=_CONNECTION,
        live_access_binding_id="binding-project-status",
        validated_request=ProjectStatusReadLiveRequestV1(project_id="UNKNOWN"),
        effective_budget=_budget(),
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind=PROJECT_STATUS_SOURCE_KIND,
        resolved_resource_scope=ResolvedLiveResourceScopeV1(),
    )
    context = LiveCapabilityExecutionContextV1(
        run_id="run-not-found",
        tenant_id=_TENANT,
        workspace_id="workspace-proof",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=999999.0,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )
    result = await handler.execute(integration=integration, call=call, context=context)
    assert result.normalized_outcome is LiveExecutionOutcomeV1.FAILED
    assert result.error_code == "live_provider_not_found"
