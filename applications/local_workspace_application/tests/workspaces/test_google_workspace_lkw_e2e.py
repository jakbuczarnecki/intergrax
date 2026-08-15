# © Artur Czarnecki. All rights reserved.

"""Google Workspace Calendar/Docs/Sheets application proof."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from typing import Mapping

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from applications.local_workspace_application.tests.workspaces.rag_e2e_support import (
    _NOW,
    _TENANT,
    _WORKSPACE,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.lkw_task_enricher import (
    build_lkw_combined_task_enricher,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.workspace_routes import (
    mount_managed_workspace_routes,
)
from local_workspace_application.workspaces.connected_source_models import (
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)

from tests.unit.integrations.providers.collaboration_suite.google_workspace.test_calendar import (
    _event as _calendar_event,
    _page as _calendar_page,
)
from tests.unit.integrations.providers.collaboration_suite.google_workspace.test_docs import (
    _document_payload,
    _document_tab,
    _paragraph_block,
    _tab,
    _text_run,
)
from tests.unit.integrations.providers.collaboration_suite.google_workspace.test_sheets import (
    _minimal_payload as _sheets_payload,
)

pytestmark = pytest.mark.unit

_GOOGLE_CONNECTION = "conn.google"
_GOOGLE_CREDENTIAL_REF = "secrets/tenant-a/google"
_CALENDAR_ID = "team@example.com"
_DOC_ID = "doc-known-1"
_SHEET_ID = "sheet-known-1"
_CALENDAR_MARKER = "calendar application proof marker"
_DOC_MARKER = "docs application proof marker"
_SHEET_MARKER = "sheets cell application proof marker"


def _docs_payload(*, title: str, text: str) -> dict[str, object]:
    end = len(text) + 1
    return _document_payload(
        document_id=_DOC_ID,
        title=title,
        tabs=[
            _tab(
                "tab-1",
                "Main",
                0,
                0,
                _document_tab(
                    [_paragraph_block(1, end, [_text_run(text, 1, end)])]
                ),
            )
        ],
    )


def _sheets_payload_with_marker(marker: str) -> dict[str, object]:
    payload = copy.deepcopy(_sheets_payload())
    payload["spreadsheetId"] = _SHEET_ID
    properties = dict(payload["properties"])
    properties["title"] = "Known application spreadsheet"
    payload["properties"] = properties
    first_sheet = copy.deepcopy(payload["sheets"][0])
    first_row = first_sheet["data"][0]["rowData"][0]
    first_cell = first_row["values"][0]
    first_cell["userEnteredValue"]["stringValue"] = marker
    first_sheet["data"] = [{"rowData": [{"values": [first_cell]}]}]
    payload["sheets"] = [first_sheet]
    return payload


def _calendar_event_payload(*, marker: str, cancelled: bool = False) -> dict[str, object]:
    payload = _calendar_event(
        event_id="calendar-known-1",
        status="cancelled" if cancelled else "confirmed",
        cancelled=cancelled,
    )
    if not cancelled:
        payload["summary"] = marker
        payload["description"] = marker
    return payload


@dataclass
class _GoogleTransport:
    docs: dict[str, object]
    sheets: dict[str, object]
    calendar_cancelled: bool = False
    calendar_marker: str = _CALENDAR_MARKER

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        _ = params, headers
        if source_kind is GoogleWorkspaceSourceKind.DOCS:
            return copy.deepcopy(self.docs)
        if source_kind is GoogleWorkspaceSourceKind.SHEETS:
            return copy.deepcopy(self.sheets)
        event = _calendar_event_payload(
            marker=self.calendar_marker,
            cancelled=self.calendar_cancelled,
        )
        if "/events/" in relative_path:
            return event
        return _calendar_page(events=[event])


@dataclass(frozen=True)
class _GoogleClientFamily:
    _transport: _GoogleTransport

    @property
    def transport(self) -> _GoogleTransport:
        return self._transport


class _GoogleClientFactory:
    def __init__(self, transport: _GoogleTransport) -> None:
        self.transport = transport
        self.calls: list[dict[str, str]] = []

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> _GoogleClientFamily:
        self.calls.append(dict(credential_material))
        return _GoogleClientFamily(self.transport)


class _GoogleSecretsStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        _ = version
        self.calls.append(path)
        if path == _GOOGLE_CREDENTIAL_REF:
            return json.dumps({"access_token": "opaque-test-token"})
        return json.dumps({"app_token": "xapp-test", "bot_token": "xoxb-test"})

    def put_secret(self, path: str, value: str) -> None:
        _ = path, value

    def delete_secret(self, path: str) -> None:
        _ = path


def _restart_with_google(
    rag_e2e_env,
    *,
    transport: _GoogleTransport,
) -> tuple[FastAPI, object]:
    repo: ManagedWorkspaceRepository = rag_e2e_env["repo"]
    harness_runtime = rag_e2e_env["harness_runtime"]
    old_settings: LocalWorkspaceBackendSettings = rag_e2e_env["settings"]
    connection_repository = DocumentStoreTenantConnectionRepository(repo.document_store)
    TenantConnectionService(
        tenant_id=_TENANT,
        repository=connection_repository,
    ).create(
        TenantConnection(
            connection_ref=_GOOGLE_CONNECTION,
            tenant_id=_TENANT,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            safe_display_name="Google Workspace",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref=_GOOGLE_CREDENTIAL_REF,
            validated_secret_free_config={},
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id="att-google",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_GOOGLE_CONNECTION,
            safe_display_label="Google Workspace",
            status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
            mutation_id="mut-google",
            effective_revision=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repo.put_workspace(
        Workspace(
            workspace_id="workspace-b",
            tenant_id=_TENANT,
            name="isolated workspace",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )

    settings = replace(
        old_settings,
        tenant_connection_bootstrap_tenant_ids=(_TENANT,),
        slack_tenant_id="",
        connected_source_slack_connection_ref="",
    )
    google_factory = _GoogleClientFactory(transport)
    secrets = _GoogleSecretsStore()
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        slack_runtime_builder=lambda _config: rag_e2e_env["integration"],
        google_client_factory=google_factory,
    )
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.transition_to_ready()
    lifecycle.set_executor_available(True)
    environment = build_local_workspace_environment_profile(settings)
    task_enricher = build_lkw_combined_task_enricher(
        environment,
        default_capability="local.workspace.search",
        agent_checkpoint_store=harness_runtime.agent_checkpoint_store,
        compensation_queue_store=harness_runtime.compensation_queue_store,
        idempotency_store=harness_runtime.reliability.idempotency_store,
    )
    task_executor = LocalWorkspaceTaskExecutor(
        harness_runtime.nexus_loop,
        task_enricher=task_enricher,
        readiness=lifecycle,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=task_executor,
        settings=settings,
        repository=repo,
        tenant_connection_secrets_store=secrets,
        tenant_connection_factory_registry=factory_registry,
        llm_adapter=rag_e2e_env["llm"],
        vectorstore_manager=harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager,
    )
    return app, (google_factory, secrets)


def _discover(client: TestClient, resource_type: str) -> dict[str, object]:
    response = client.get(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/connections/"
        f"{_GOOGLE_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": resource_type, "limit": 10},
    )
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["resource_type"] == resource_type
    return item


def _create(
    client: TestClient,
    *,
    candidate: str,
    expected_revision: int,
    idempotency: str,
) -> dict[str, object]:
    response = client.post(
        "/v1/local_workspace/workspaces/workspace-1/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": f"WKC/{expected_revision}",
            "Idempotency-Key": idempotency,
        },
        json={
            "connection_ref": _GOOGLE_CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": "2026-01-01T00:00:00+00:00",
            "root_latest": "2026-02-01T00:00:00+00:00",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _sync_and_drain(client: TestClient, runtime, repo, source: dict[str, object]) -> None:
    response = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{source['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert response.status_code == 202, response.text
    operation_id = response.json()["operation_id"]
    for _ in range(128):
        runtime.worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            break
    final = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
    assert final is not None and final.status.value == "completed", (
        f"status={final.status.value if final is not None else None} "
        f"error={final.error if final is not None else None}"
    )


def test_google_workspace_rehydrated_calendar_docs_sheets_search_ask(rag_e2e_env) -> None:
    transport = _GoogleTransport(
        docs=_docs_payload(title="Known Docs", text=_DOC_MARKER),
        sheets=_sheets_payload_with_marker(_SHEET_MARKER),
    )
    app, runtime_handles = _restart_with_google(
        rag_e2e_env,
        transport=transport,
    )
    google_factory, secrets = runtime_handles
    with TestClient(app) as client:
        wiring = app.state.lkw_connected_source_wiring
        runtime = app.state.lkw_managed_workspace_sync_runtime
        repo = rag_e2e_env["repo"]
        wiring.google_known_resource_catalog.register(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_GOOGLE_CONNECTION,
            resource_type=RemoteResourceTypeV1.GOOGLE_WORKSPACE_CALENDAR,
            remote_resource_id=_CALENDAR_ID,
            safe_display_label="Known Calendar",
        )
        wiring.google_known_resource_catalog.register(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_GOOGLE_CONNECTION,
            resource_type=RemoteResourceTypeV1.GOOGLE_WORKSPACE_DOCS,
            remote_resource_id=_DOC_ID,
            safe_display_label="Known Docs",
        )
        wiring.google_known_resource_catalog.register(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_GOOGLE_CONNECTION,
            resource_type=RemoteResourceTypeV1.GOOGLE_WORKSPACE_SHEETS,
            remote_resource_id=_SHEET_ID,
            safe_display_label="Known Sheets",
        )
        assert secrets.calls.count(_GOOGLE_CREDENTIAL_REF) == 1
        connection = wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_GOOGLE_CONNECTION,
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        )
        assert connection is not None

        calendar = _discover(client, "google_workspace_calendar")
        docs = _discover(client, "google_workspace_docs")
        sheets = _discover(client, "google_workspace_sheets")
        assert len(google_factory.calls) == 1
        sources = (
            _create(client, candidate=calendar["opaque_candidate_ref"], expected_revision=1, idempotency="google-calendar"),
            _create(client, candidate=docs["opaque_candidate_ref"], expected_revision=2, idempotency="google-docs"),
            _create(client, candidate=sheets["opaque_candidate_ref"], expected_revision=3, idempotency="google-sheets"),
        )
        calendar_source, docs_source, _sheets_source = sources
        bindings = [
            wiring.tenant_binding_port.get_binding(
                tenant_id=_TENANT,
                binding_id=source["knowledge_source_binding_ref"],
            )
            for source in sources
        ]
        assert {binding.connection_ref for binding in bindings if binding is not None} == {
            _GOOGLE_CONNECTION
        }
        assert {binding.source_kind for binding in bindings if binding is not None} == {
            GOOGLE_CALENDAR_SOURCE_KIND,
            GOOGLE_DOCS_SOURCE_KIND,
            GOOGLE_SHEETS_SOURCE_KIND,
        }

        for source in sources:
            _sync_and_drain(client, runtime, repo, source)

        markers = (_CALENDAR_MARKER, _DOC_MARKER, _SHEET_MARKER)
        for source, marker in zip(sources, markers, strict=True):
            search = client.post(
                f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
                headers={"X-Tenant-Id": _TENANT},
                json={"query": marker, "limit": 10},
            )
            assert search.status_code == 200, search.text
            results = search.json()["results"]
            assert results
            assert any(
                hit["source_id"] == source["source_id"] and marker in hit["snippet"]
                for hit in results
            )
            rag_e2e_env["llm"]._fixed_text = json.dumps(
                {
                    "status": "completed",
                    "answer": marker,
                    "used_evidence_ids": ["E1"],
                }
            )
            ask = client.post(
                f"/v1/local_workspace/workspaces/{_WORKSPACE}/ask",
                headers={
                    "X-Tenant-Id": _TENANT,
                    "Idempotency-Key": f"google-ask-{source['source_id']}",
                },
                json={"question": marker},
            )
            assert ask.status_code == 200, ask.text
            assert ask.json()["citations"]
            assert ask.json()["citations"][0]["source_id"] == source["source_id"]

        docs_ref = next(
            ref
            for ref in repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
            if ref.source_id == docs_source["source_id"]
        )
        transport.docs = _docs_payload(title="Known Docs revised", text="revised docs marker")
        _sync_and_drain(client, runtime, repo, docs_source)
        revised_docs_ref = repo.get_document_ref(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            document_id=docs_ref.document_id,
        )
        assert revised_docs_ref is not None
        assert revised_docs_ref.document_id == docs_ref.document_id
        assert revised_docs_ref.content_hash != docs_ref.content_hash

        transport.calendar_marker = "calendar revised marker"
        _sync_and_drain(client, runtime, repo, calendar_source)
        transport.calendar_cancelled = True
        _sync_and_drain(client, runtime, repo, calendar_source)

        old_calendar_search = client.post(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": _CALENDAR_MARKER, "limit": 10},
        )
        assert old_calendar_search.status_code == 200
        assert not old_calendar_search.json()["results"]
        cross_source_search = client.post(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": _DOC_MARKER, "limit": 10},
        )
        cross_source_results = cross_source_search.json()["results"]
        assert any(
            hit["source_id"] == docs_source["source_id"]
            and _DOC_MARKER in hit["snippet"]
            for hit in cross_source_results
        )
        assert all(
            _DOC_MARKER not in hit["snippet"]
            or hit["source_id"] == docs_source["source_id"]
            for hit in cross_source_results
        )
        other_workspace = client.post(
            "/v1/local_workspace/workspaces/workspace-b/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": _DOC_MARKER, "limit": 10},
        )
        assert other_workspace.status_code == 200
        assert other_workspace.json()["results"] == []
        other_tenant = client.post(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": "tenant-other"},
            json={"query": _DOC_MARKER, "limit": 10},
        )
        assert other_tenant.status_code in {403, 404}
