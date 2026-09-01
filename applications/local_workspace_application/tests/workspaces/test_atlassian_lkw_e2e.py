# © Artur Czarnecki. All rights reserved.

"""Restarted Jira and Confluence application Search/Ask proof."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.lkw_task_enricher import (
    build_lkw_combined_task_enricher,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.execution_wiring import build_lkw_host_task_execution
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
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
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

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-other"
_WORKSPACE = "workspace-1"
_OTHER_WORKSPACE = "workspace-b"
_JIRA_CONNECTION = "conn.jira"
_CONFLUENCE_CONNECTION = "conn.confluence"
_JIRA_CREDENTIAL_REF = "secrets/tenant-a/jira"
_CONFLUENCE_CREDENTIAL_REF = "secrets/tenant-a/confluence"
_JIRA_PROJECT = "PROJ"
_CONFLUENCE_SPACE = "10000"
_JIRA_MARKER = "JIRA_BODY_MARKER_4D"
_CONFLUENCE_MARKER = "CONFLUENCE_BODY_MARKER_4D"


class _Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _jira_issue_payload() -> dict[str, object]:
    return {
        "id": "12345",
        "key": "PROJ-1",
        "fields": {
            "summary": "Bounded Jira issue",
            "description": _JIRA_MARKER,
            "status": {"id": "3", "name": "In Progress"},
            "issuetype": {"id": "10001", "name": "Task"},
            "project": {"id": "10000", "key": _JIRA_PROJECT, "name": "Platform Project"},
            "priority": {"name": "High"},
            "labels": ["readiness"],
            "components": [],
            "assignee": None,
            "reporter": None,
            "resolution": None,
            "created": "2026-01-01T10:00:00.000+0000",
            "updated": "2026-01-02T10:00:00.000+0000",
        },
    }


class _JiraHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def post(self, path: str, *, json: dict[str, object]) -> _Response:
        self.calls.append(("POST", path))
        assert json["jql"] == f'project = "{_JIRA_PROJECT}" ORDER BY id ASC'
        return _Response({"issues": [_jira_issue_payload()], "isLast": True})

    def get(self, path: str, *, params: dict[str, object] | None = None) -> _Response:
        self.calls.append(("GET", path))
        assert path == "/issue/PROJ-1"
        return _Response(_jira_issue_payload())


def _confluence_page_payload(*, include_body: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "20001",
        "status": "current",
        "title": "Bounded Confluence page",
        "spaceId": _CONFLUENCE_SPACE,
        "createdAt": "2026-01-01T10:00:00.000Z",
        "version": {
            "number": 3,
            "createdAt": "2026-01-02T10:00:00.000Z",
        },
        "parentId": None,
    }
    if include_body:
        payload["body"] = {
            "storage": {
                "value": f"<p>{_CONFLUENCE_MARKER}</p>",
            }
        }
    return payload


class _ConfluenceHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> _Response:
        self.calls.append(("GET", path))
        if "/spaces/10000/pages" in path:
            return _Response(
                {
                    "results": [_confluence_page_payload(include_body=False)],
                    "_links": {},
                }
            )
        assert path.endswith("/pages/20001")
        assert params == {"body-format": "storage", "version": 3}
        return _Response(_confluence_page_payload(include_body=True))


class _SecretsStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        del version
        self.calls.append(path)
        if path == _JIRA_CREDENTIAL_REF:
            return json.dumps({"email": "jira@example.com", "api_token": "jira-token"})
        if path == _CONFLUENCE_CREDENTIAL_REF:
            return json.dumps(
                {"email": "confluence@example.com", "api_token": "confluence-token"}
            )
        raise KeyError(path)

    def put_secret(self, path: str, value: str) -> None:
        del path, value

    def delete_secret(self, path: str) -> None:
        del path


class _HttpFactories:
    def __init__(self) -> None:
        self.jira = _JiraHttpClient()
        self.confluence = _ConfluenceHttpClient()
        self.jira_calls = 0
        self.confluence_calls = 0

    def jira_factory(self, config: Any) -> _JiraHttpClient:
        assert config.api_token == "jira-token"
        self.jira_calls += 1
        return self.jira

    def confluence_factory(self, config: Any) -> _ConfluenceHttpClient:
        assert config.api_token == "confluence-token"
        self.confluence_calls += 1
        return self.confluence


def _connection(
    *,
    connection_ref: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    credential_ref: str,
    base_url: str,
) -> TenantConnection:
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id=_TENANT,
        provider_id=provider_id,
        integration_kind=integration_kind,
        safe_display_name=provider_id.title(),
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref=credential_ref,
        validated_secret_free_config={"base_url": base_url},
        configuration_version=1,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _restart_application(
    rag_e2e_env: dict[str, object],
) -> tuple[FastAPI, _HttpFactories, _SecretsStore]:
    repo: ManagedWorkspaceRepository = rag_e2e_env["repo"]  # type: ignore[assignment]
    harness_runtime = rag_e2e_env["harness_runtime"]
    old_settings: LocalWorkspaceBackendSettings = rag_e2e_env["settings"]
    connection_repository = DocumentStoreTenantConnectionRepository(repo.document_store)
    connection_service = TenantConnectionService(
        tenant_id=_TENANT,
        repository=connection_repository,
    )
    connection_service.create(
        _connection(
            connection_ref=_JIRA_CONNECTION,
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            credential_ref=_JIRA_CREDENTIAL_REF,
            base_url="https://jira.example.test",
        )
    )
    connection_service.create(
        _connection(
            connection_ref=_CONFLUENCE_CONNECTION,
            provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
            credential_ref=_CONFLUENCE_CREDENTIAL_REF,
            base_url="https://confluence.example.test/wiki",
        )
    )
    for attachment_id, connection_ref, label in (
        ("att-jira", _JIRA_CONNECTION, "Jira"),
        ("att-confluence", _CONFLUENCE_CONNECTION, "Confluence"),
    ):
        repo.put_knowledge_connection_attachment_version_if_absent(
            WorkspaceConnectionAttachment(
                attachment_id=attachment_id,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                connection_ref=connection_ref,
                safe_display_label=label,
                status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
                mutation_id=f"mutation-{attachment_id}",
                effective_revision=1,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    repo.put_workspace(
        Workspace(
            workspace_id=_OTHER_WORKSPACE,
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
    factories = _HttpFactories()
    secrets = _SecretsStore()
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        jira_http_client_factory=factories.jira_factory,
        confluence_http_client_factory=factories.confluence_factory,
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
        build_lkw_host_task_execution(harness_runtime.nexus_loop, environment),
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
    return app, factories, secrets


def _discover(
    client: TestClient,
    *,
    connection_ref: str,
    resource_type: RemoteResourceTypeV1,
) -> dict[str, object]:
    response = client.get(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/connections/"
        f"{connection_ref}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": resource_type.value, "limit": 10},
    )
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert len(items) == 1
    return items[0]


def _create(
    client: TestClient,
    *,
    candidate: str,
    expected_revision: int,
    idempotency_key: str,
) -> dict[str, object]:
    response = client.post(
        f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": f"WKC/{expected_revision}",
            "Idempotency-Key": idempotency_key,
        },
        json={
            "connection_ref": (
                _JIRA_CONNECTION
                if idempotency_key.startswith("jira")
                else _CONFLUENCE_CONNECTION
            ),
            "opaque_candidate_ref": candidate,
            "root_oldest": "2026-01-01T00:00:00+00:00",
            "root_latest": "2026-02-01T00:00:00+00:00",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _sync_and_drain(
    client: TestClient,
    runtime: Any,
    repo: ManagedWorkspaceRepository,
    source: dict[str, object],
) -> None:
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


def _assert_search_and_ask(
    *,
    client: TestClient,
    llm: Any,
    workspace_id: str,
    marker: str,
    source: dict[str, object],
    ask_key: str,
) -> None:
    search = client.post(
        f"/v1/local_workspace/workspaces/{workspace_id}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={"query": marker, "limit": 10},
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert any(
        hit["source_id"] == source["source_id"] and marker in hit["snippet"]
        for hit in results
    )
    llm._fixed_text = json.dumps(
        {
            "status": "completed",
            "answer": marker,
            "used_evidence_ids": ["E1"],
        }
    )
    ask = client.post(
        f"/v1/local_workspace/workspaces/{workspace_id}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": ask_key},
        json={"question": marker},
    )
    assert ask.status_code == 200, ask.text
    citations = ask.json()["citations"]
    assert citations
    assert citations[0]["source_id"] == source["source_id"]


def test_atlassian_rehydrated_jira_and_confluence_search_ask(rag_e2e_env) -> None:
    app, factories, secrets = _restart_application(rag_e2e_env)
    with TestClient(app) as client:
        wiring = app.state.lkw_connected_source_wiring
        runtime = app.state.lkw_managed_workspace_sync_runtime
        repo: ManagedWorkspaceRepository = rag_e2e_env["repo"]
        assert secrets.calls.count(_JIRA_CREDENTIAL_REF) == 1
        assert secrets.calls.count(_CONFLUENCE_CREDENTIAL_REF) == 1
        assert factories.jira_calls == 1
        assert factories.confluence_calls == 1

        jira_integration = wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_JIRA_CONNECTION,
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        )
        confluence_integration = wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_CONFLUENCE_CONNECTION,
            provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        )
        assert jira_integration is not confluence_integration
        with pytest.raises(VendorKnowledgeError):
            wiring.connection_registry.resolve(
                tenant_id=_TENANT,
                connection_ref=_JIRA_CONNECTION,
                provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
                integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
            )
        with pytest.raises(VendorKnowledgeError):
            wiring.connection_registry.resolve(
                tenant_id=_TENANT,
                connection_ref=_CONFLUENCE_CONNECTION,
                provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
            )

        wiring.jira_known_project_catalog.register(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_JIRA_CONNECTION,
            project_key=_JIRA_PROJECT,
            safe_display_label="Platform Project",
        )
        wiring.confluence_known_space_catalog.register(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONFLUENCE_CONNECTION,
            space_id=_CONFLUENCE_SPACE,
            safe_display_label="Engineering Space",
        )
        jira_candidate = _discover(
            client,
            connection_ref=_JIRA_CONNECTION,
            resource_type=RemoteResourceTypeV1.JIRA_PROJECT,
        )
        confluence_candidate = _discover(
            client,
            connection_ref=_CONFLUENCE_CONNECTION,
            resource_type=RemoteResourceTypeV1.CONFLUENCE_SPACE,
        )
        jira_source = _create(
            client,
            candidate=jira_candidate["opaque_candidate_ref"],
            expected_revision=1,
            idempotency_key="jira-source",
        )
        confluence_source = _create(
            client,
            candidate=confluence_candidate["opaque_candidate_ref"],
            expected_revision=2,
            idempotency_key="confluence-source",
        )
        _sync_and_drain(client, runtime, repo, jira_source)
        _sync_and_drain(client, runtime, repo, confluence_source)

        _assert_search_and_ask(
            client=client,
            llm=rag_e2e_env["llm"],
            workspace_id=_WORKSPACE,
            marker=_JIRA_MARKER,
            source=jira_source,
            ask_key="jira-ask",
        )
        _assert_search_and_ask(
            client=client,
            llm=rag_e2e_env["llm"],
            workspace_id=_WORKSPACE,
            marker=_CONFLUENCE_MARKER,
            source=confluence_source,
            ask_key="confluence-ask",
        )

        unknown_jira = wiring.opaque_ref_codec.encode_jira_project_candidate(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_JIRA_CONNECTION,
            project_key="OTHER",
            safe_display_label="Unknown project",
        )
        unknown_project = client.post(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
            headers={
                "X-Tenant-Id": _TENANT,
                "If-Match": "WKC/3",
                "Idempotency-Key": "unknown-jira-project",
            },
            json={
                "connection_ref": _JIRA_CONNECTION,
                "opaque_candidate_ref": unknown_jira,
                "root_oldest": "2026-01-01T00:00:00+00:00",
                "root_latest": "2026-02-01T00:00:00+00:00",
            },
        )
        assert 400 <= unknown_project.status_code < 500
        unknown_confluence = wiring.opaque_ref_codec.encode_confluence_space_candidate(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONFLUENCE_CONNECTION,
            space_id="99999",
            safe_display_label="Unknown space",
        )
        unknown_space = client.post(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
            headers={
                "X-Tenant-Id": _TENANT,
                "If-Match": "WKC/3",
                "Idempotency-Key": "unknown-confluence-space",
            },
            json={
                "connection_ref": _CONFLUENCE_CONNECTION,
                "opaque_candidate_ref": unknown_confluence,
                "root_oldest": "2026-01-01T00:00:00+00:00",
                "root_latest": "2026-02-01T00:00:00+00:00",
            },
        )
        assert 400 <= unknown_space.status_code < 500
        wrong_resource = client.get(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/knowledge/connections/"
            f"{_CONFLUENCE_CONNECTION}/remote-resources",
            headers={"X-Tenant-Id": _TENANT},
            params={"resource_type": RemoteResourceTypeV1.JIRA_PROJECT.value},
        )
        assert wrong_resource.status_code == 200
        assert wrong_resource.json()["items"] == []
        other_workspace = client.post(
            f"/v1/local_workspace/workspaces/{_OTHER_WORKSPACE}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": _JIRA_MARKER, "limit": 10},
        )
        assert other_workspace.status_code == 200
        assert other_workspace.json()["results"] == []
        other_tenant = client.post(
            f"/v1/local_workspace/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": _OTHER_TENANT},
            json={"query": _JIRA_MARKER, "limit": 10},
        )
        assert other_tenant.status_code in {403, 404}
