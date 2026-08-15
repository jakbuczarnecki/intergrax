# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.6 P6 integration factories (32 harness slugs)."""

from __future__ import annotations

import os
import math
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage
from intergrax.integrations._shared.health import http_ping_ok
from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig, QueueIntegrationConfig
from intergrax.integrations._shared.p2.factories import _open_httpx_client, _resolve
from intergrax.integrations._shared.p3.clients import RestVectorStoreIntegration, build_rest_search_provider
from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations._shared.p5.factories import (
    _feature_flag_factory,
    _http_obs,
    _issue_tracker_factory,
    _sql_store_factory,
)
from intergrax.integrations._shared.p5.clients import HttpCiCdBackend, RestSecretsStore
from intergrax.integrations._shared.p7.clients import (
    HttpBillingMeterBackend,
    HttpCrmBackend,
    HttpIdentityProviderBackend,
    HttpMlInferenceHostBackend,
    HttpSandboxHostBackend,
    HttpSecurityScannerBackend,
    HttpSpeechProviderBackend,
    HttpVisionServingBackend,
    HttpWorkflowOrchestratorBackend,
    SpeechAdapterBackend,
)
from intergrax.integrations.contracts.billing_meter import BillingMeterBackend
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.contracts.crm import CrmBackend
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.ml_inference_host import MlInferenceHostBackend
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    effective_filter,
    native_hit,
    provider_metadata,
    validate_query,
    validate_records,
    validate_scope,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreContractError
from intergrax.integrations.contracts.vision_serving import VisionServingBackend
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter
from intergrax.websearch.schemas.search_hit import SearchHit


def _security_scanner_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    scan_image_path: str,
    scan_repo_path: str,
    health_path: str = "/",
    security_scanner: Optional[SecurityScannerBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecurityScannerBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def scan_image(self, image_ref: str) -> dict[str, Any]:
                response = http.post(scan_image_path, json={"image": image_ref, "ref": image_ref})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"target": image_ref}

            def scan_repo(self, repo_path: str) -> dict[str, Any]:
                response = http.post(scan_repo_path, json={"path": repo_path, "repo": repo_path})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"target": repo_path}

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=security_scanner,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpSecurityScannerBackend(c, provider=provider),
    )


def _sandbox_host_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    sandbox_host: Optional[SandboxHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SandboxHostBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def create_session(self) -> dict[str, Any]:
                response = http.post("/sessions")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"session_id": ""}

            def exec(self, session_id: str, command: str) -> dict[str, Any]:
                response = http.post(
                    f"/sessions/{session_id}/exec",
                    json={"command": command},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"exit_code": 0}

            def upload_artifact(
                self,
                session_id: str,
                *,
                local_path: str,
                remote_name: str,
            ) -> dict[str, Any]:
                response = http.post(
                    f"/sessions/{session_id}/artifacts",
                    json={"local_path": local_path, "remote_name": remote_name},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"artifact_id": remote_name}

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=sandbox_host,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpSandboxHostBackend(c, provider=provider),
    )


def _identity_provider_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    identity_provider: Optional[IdentityProviderBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IdentityProviderBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def verify_token(self, token: str) -> dict[str, Any]:
                response = http.get("/userinfo", headers={"Authorization": f"Bearer {token}"})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"sub": token}

            def userinfo(self, token: str) -> dict[str, Any]:
                return self.verify_token(token)

            def list_tenants(self, *, limit: int = 50) -> list[dict[str, Any]]:
                response = http.get("/tenants", params={"limit": limit})
                if response.status_code >= 400:
                    return []
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return list(payload)[:limit]
                if isinstance(payload, dict):
                    rows = payload.get("tenants") or payload.get("items") or []
                    return list(rows)[:limit]
                return []

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=identity_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpIdentityProviderBackend(c, provider=provider),
    )


def _ci_cd_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)
        org = config.org
        repo = config.repo

        class _Client:
            def get_workflow_run(self, run_id: str) -> dict[str, Any]:
                if provider == "argocd":
                    response = http.get(f"/api/v1/applications/{org}/events", params={"run": run_id})
                elif provider == "buildkite":
                    response = http.get(f"/v2/organizations/{org}/pipelines/{repo}/builds/{run_id}")
                elif provider == "jenkins":
                    response = http.get(f"/job/{repo}/{run_id}/api/json")
                else:
                    response = http.get(f"/runs/{run_id}")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"id": run_id}

            def list_check_suites(self, *, ref: str, limit: int = 20) -> list[dict[str, Any]]:
                if provider == "argocd":
                    response = http.get(f"/api/v1/applications/{org}", params={"revision": ref})
                elif provider == "buildkite":
                    response = http.get(
                        f"/v2/organizations/{org}/pipelines/{repo}/builds",
                        params={"branch": ref, "per_page": limit},
                    )
                elif provider == "jenkins":
                    response = http.get(f"/job/{repo}/api/json", params={"tree": "builds[number,result]"})
                else:
                    return []
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload[:limit]
                if isinstance(payload, dict):
                    for key in ("builds", "items", "results", "events"):
                        rows = payload.get(key)
                        if isinstance(rows, list):
                            return list(rows)[:limit]
                return []

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=ci_cd,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpCiCdBackend(c, provider=provider),
    )


def _workflow_orchestrator_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    workflow_orchestrator: Optional[WorkflowOrchestratorBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WorkflowOrchestratorBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def trigger_run(self, workflow_id: str, *, parameters: dict[str, str]) -> dict[str, Any]:
                if provider == "prefect":
                    response = http.post(f"/api/deployments/{workflow_id}/create_flow_run", json={"parameters": parameters})
                else:
                    response = http.post(f"/api/v1/dags/{workflow_id}/dagRuns", json={"conf": parameters})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"run_id": workflow_id}

            def poll_status(self, run_id: str) -> dict[str, Any]:
                if provider == "prefect":
                    response = http.get(f"/api/flow_runs/{run_id}")
                else:
                    response = http.get(f"/api/v1/dags/~/dagRuns/{run_id}")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"run_id": run_id}

            def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
                if provider == "prefect":
                    response = http.get(f"/api/flow_runs/{run_id}/logs", params={"limit": tail_lines})
                else:
                    response = http.get(f"/api/v1/dags/~/dagRuns/{run_id}/logs/{run_id}", params={"limit": tail_lines})
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return str(payload.get("logs") or payload.get("content") or "")
                return str(payload)

            def list_runs(self, *, workflow_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
                if provider == "prefect":
                    params: dict[str, Any] = {"limit": limit}
                    if workflow_id:
                        params["deployment_id"] = workflow_id
                    response = http.get("/api/flow_runs", params=params)
                else:
                    dag = workflow_id or "~"
                    response = http.get(f"/api/v1/dags/{dag}/dagRuns", params={"limit": limit})
                response.raise_for_status()
                payload = response.json()
                rows = payload if isinstance(payload, list) else list(
                    payload.get("dag_runs") or payload.get("flow_runs") or []
                )
                return [
                    dict(item) if isinstance(item, dict) else {"run_id": str(item)}
                    for item in rows[:limit]
                ]

            def cancel_run(self, run_id: str) -> dict[str, Any]:
                if provider == "prefect":
                    response = http.post(
                        f"/api/flow_runs/{run_id}/set_state",
                        json={"state": {"type": "CANCELLED"}},
                    )
                else:
                    response = http.patch(
                        f"/api/v1/dags/~/dagRuns/{run_id}",
                        json={"state": "failed"},
                    )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"run_id": run_id, "status": "cancelled"}

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=workflow_orchestrator,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpWorkflowOrchestratorBackend(c, provider=provider),
    )


def _vision_serving_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/v2/health/ready",
    vision_serving: Optional[VisionServingBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VisionServingBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def predict(self, model_name: str, *, input_uri: str) -> dict[str, Any]:
                response = http.post(
                    f"/v2/models/{model_name}/infer",
                    json={"inputs": [{"uri": input_uri}]},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"predictions": []}

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=vision_serving,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpVisionServingBackend(c, provider=provider),
    )


def _ml_inference_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    ml_inference_host: Optional[MlInferenceHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MlInferenceHostBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def predict(self, model_ref: str, *, inputs: Mapping[str, Any]) -> dict[str, Any]:
                response = http.post(
                    f"/v1/models/{model_ref}/predictions",
                    json={"input": dict(inputs)},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"output": payload}

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=ml_inference_host,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpMlInferenceHostBackend(c, provider=provider),
    )


def _billing_meter_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    billing_meter: Optional[BillingMeterBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BillingMeterBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def list_meter_events(self, *, customer_id: str, limit: int = 50) -> list[dict[str, Any]]:
                response = http.get(
                    "/v1/billing/meter_events",
                    params={"customer": customer_id, "limit": limit},
                )
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("data") if isinstance(payload, dict) else payload
                return list(rows or [])[:limit]

            def submit_meter_event(self, *, customer_id: str, metric: str, quantity: float) -> dict[str, Any]:
                response = http.post(
                    "/v1/billing/meter_events",
                    json={"customer": customer_id, "event_name": metric, "value": quantity},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"customer_id": customer_id}

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=billing_meter,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpBillingMeterBackend(c, provider=provider),
    )


def _crm_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    health_path: str = "/",
    crm: Optional[CrmBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CrmBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def get_account(self, account_id: str) -> dict[str, Any]:
                if provider == "salesforce":
                    response = http.get(f"/services/data/v59.0/sobjects/Account/{account_id}")
                else:
                    response = http.get(f"/crm/v3/objects/companies/{account_id}")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"account_id": account_id}

            def list_contacts(self, *, account_id: str, limit: int = 50) -> list[dict[str, Any]]:
                if provider == "salesforce":
                    response = http.get(
                        "/services/data/v59.0/query",
                        params={"q": f"SELECT Id,Name,Email FROM Contact WHERE AccountId='{account_id}' LIMIT {limit}"},
                    )
                else:
                    response = http.get(
                        "/crm/v3/objects/contacts",
                        params={"associations.company": account_id, "limit": limit},
                    )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload[:limit]
                if isinstance(payload, dict):
                    rows = payload.get("records") or payload.get("results") or []
                    return list(rows)[:limit]
                return []

            def list_tickets(self, *, account_id: str, limit: int = 50) -> list[dict[str, Any]]:
                if provider == "salesforce":
                    response = http.get(
                        "/services/data/v59.0/query",
                        params={"q": f"SELECT Id,Subject,Status FROM Case WHERE AccountId='{account_id}' LIMIT {limit}"},
                    )
                else:
                    response = http.get(
                        "/crm/v3/objects/tickets",
                        params={"associations.company": account_id, "limit": limit},
                    )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload[:limit]
                if isinstance(payload, dict):
                    rows = payload.get("records") or payload.get("results") or []
                    return list(rows)[:limit]
                return []

            def health(self) -> bool:
                return http_ping_ok(http, path=health_path)

        return _Client()

    return _resolve(
        implementation=crm,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpCrmBackend(c, provider=provider),
    )


class _TypesenseHttpVectorStore(VectorStore):
    def __init__(self, http: Any, *, collection: str) -> None:
        self._http = http
        self._collection = collection
        self._tenant_id: str | None = None

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        if self._tenant_id is None:
            self._tenant_id = scope.tenant_id
        validated = validate_records(
            records,
            scope=scope,
            tenant_id=self._tenant_id,
        )
        if not validated:
            return []
        rows: list[dict[str, Any]] = []
        for record in validated:
            metadata = provider_metadata(record.document, scope=scope)
            user_fields = {
                key: value
                for key, value in metadata.items()
                if key not in {"id", "content", "embedding", "metadata"}
            }
            rows.append(
                {
                    "id": record.vector_id,
                    "content": record.document.content,
                    "embedding": record.embedding.tolist(),
                    "metadata": metadata,
                    **user_fields,
                }
            )
        response = self._http.post(f"/collections/{self._collection}/documents/import", json=rows)
        response.raise_for_status()
        return [record.vector_id for record in validated]

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        vector, limit = validate_query(query_embedding, top_k=top_k)
        validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)
        conditions = effective_filter(scope, metadata_filter).conditions
        body: dict[str, Any] = {
            "q": "*",
            "vector_query": f"embedding:([{','.join(str(v) for v in vector)}], k:{limit})",
            "per_page": limit,
        }
        if conditions:
            body["filter_by"] = " && ".join(self._filter_condition(key, value) for key, value in conditions.items())
        response = self._http.post(f"/collections/{self._collection}/documents/search", json=body)
        response.raise_for_status()
        payload = response.json()
        hits: list[VectorStoreHit] = []
        for rank, row in enumerate(list((payload.get("hits") if isinstance(payload, dict) else []) or [])[:limit]):
            document = dict(row.get("document") or row)
            metadata = dict(document.get("metadata") or row.get("metadata") or {})
            metadata.update(
                {
                    key: row[key]
                    for key in ("schema_version", "document_id", "root_document_id", "parent_document_id",
                                "tenant_id", "namespace", "workspace_id", "source_kind", "source_id",
                                "source_parent_id", "provider_id", "source_revision", "source_uri", "content_hash")
                    if key in row and key not in metadata
                }
            )
            hits.append(
                native_hit(
                    vector_id=str(document.get("id") or ""),
                    content=str(document.get("content") or ""),
                    metadata=metadata,
                    similarity_score=self._score(row),
                    rank=rank,
                    scope=scope,
                    embedding=document.get("embedding") if include_embeddings else None,
                )
            )
        return hits

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)
        if ids:
            raise VectorStoreContractError("typesense scoped delete is unsupported")

    def count(self, *, scope: VectorStoreScope) -> int:
        validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)
        raise VectorStoreContractError("typesense scoped count is unsupported")

    def health(self) -> bool:
        return http_ping_ok(self._http, path="/health")

    @staticmethod
    def _filter_condition(key: str, value: object) -> str:
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise VectorStoreContractError("typesense filter contains an invalid field")
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise VectorStoreContractError("typesense filter value must be finite")
            literal = str(value)
        elif isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("`", "\\`")
            literal = f"`{escaped}`"
        else:
            raise VectorStoreContractError("typesense filter value is unsupported")
        return f"{key}:={literal}"

    @staticmethod
    def _score(row: Mapping[str, Any]) -> float:
        distance = row.get("vector_distance")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            return 1.0 / (1.0 + max(0.0, float(distance)))
        text_match = row.get("text_match")
        if isinstance(text_match, (int, float)) and not isinstance(text_match, bool):
            return max(0.0, min(1.0, float(text_match)))
        return 0.0


def _algolia_hits(query: str, payload: Mapping[str, Any], limit: int) -> Sequence[SearchHit]:
    hits: list[SearchHit] = []
    rows = payload.get("hits") if isinstance(payload, dict) else []
    for idx, row in enumerate(list(rows or [])[:limit]):
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                provider="algolia",
                query_issued=query,
                rank=idx + 1,
                title=str(row.get("title") or row.get("name") or ""),
                url=str(row.get("url") or row.get("permalink") or ""),
                snippet=str(row.get("snippet") or row.get("description") or ""),
            )
        )
    return hits


# --- H-INT-10: security + secrets ---


def create_trivy_security_scanner(
    *,
    security_scanner: Optional[SecurityScannerBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecurityScannerBackend:
    return _security_scanner_factory(
        env_prefix="INTERGRAX_TRIVY",
        provider="trivy",
        default_url="http://127.0.0.1:4954",
        scan_image_path="/scan/image",
        scan_repo_path="/scan/repo",
        health_path="/healthz",
        security_scanner=security_scanner,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_snyk_security_scanner(
    *,
    security_scanner: Optional[SecurityScannerBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecurityScannerBackend:
    return _security_scanner_factory(
        env_prefix="INTERGRAX_SNYK",
        provider="snyk",
        default_url="https://api.snyk.io",
        scan_image_path="/v1/container/image",
        scan_repo_path="/v1/code/analysis",
        health_path="/v1/health",
        security_scanner=security_scanner,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_semgrep_security_scanner(
    *,
    security_scanner: Optional[SecurityScannerBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecurityScannerBackend:
    return _security_scanner_factory(
        env_prefix="INTERGRAX_SEMGREP",
        provider="semgrep",
        default_url="https://semgrep.dev",
        scan_image_path="/api/scan/image",
        scan_repo_path="/api/scan",
        health_path="/api/health",
        security_scanner=security_scanner,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_infisical_secrets_store(
    *,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    config = HttpIntegrationConfig.from_env("INTERGRAX_INFISICAL", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://app.infisical.com")

        class _Client:
            def read_secret(self, path: str, *, version: Optional[str] = None) -> str:
                del version
                response = http.get(f"/api/v3/secrets/{path}")
                response.raise_for_status()
                payload = response.json()
                return str((payload.get("secret") if isinstance(payload, dict) else payload) or "")

            def write_secret(self, path: str, value: str) -> None:
                http.post(f"/api/v3/secrets/{path}", json={"secret": value})

            def delete_secret(self, path: str) -> None:
                http.delete(f"/api/v3/secrets/{path}")

            def health(self) -> bool:
                try:
                    response = http.get("/api/v3/status")
                    return response.status_code < 500
                except Exception:
                    return False

        return _Client()

    return _resolve(
        implementation=secrets_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: RestSecretsStore(c),
    )


# --- H-INT-11: sandbox hosts ---


def create_e2b_sandbox_host(
    *,
    sandbox_host: Optional[SandboxHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SandboxHostBackend:
    return _sandbox_host_factory(
        env_prefix="INTERGRAX_E2B",
        provider="e2b",
        default_url="https://api.e2b.dev",
        health_path="/health",
        sandbox_host=sandbox_host,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_modal_sandbox_host(
    *,
    sandbox_host: Optional[SandboxHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SandboxHostBackend:
    return _sandbox_host_factory(
        env_prefix="INTERGRAX_MODAL",
        provider="modal",
        default_url="https://api.modal.com",
        health_path="/health",
        sandbox_host=sandbox_host,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_daytona_sandbox_host(
    *,
    sandbox_host: Optional[SandboxHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SandboxHostBackend:
    return _sandbox_host_factory(
        env_prefix="INTERGRAX_DAYTONA",
        provider="daytona",
        default_url="https://app.daytona.io",
        health_path="/health",
        sandbox_host=sandbox_host,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


# --- H-INT-12: identity providers ---


def create_auth0_identity_provider(
    *,
    identity_provider: Optional[IdentityProviderBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IdentityProviderBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_AUTH0", **config_overrides)
    default_url = config.base_url or (f"https://{config.org}.auth0.com" if config.org else "https://your-tenant.auth0.com")
    return _identity_provider_factory(
        env_prefix="INTERGRAX_AUTH0",
        provider="auth0",
        default_url=default_url,
        health_path="/.well-known/openid-configuration",
        identity_provider=identity_provider,
        client=client,
        client_factory=client_factory,
        base_url=default_url,
        **config_overrides,
    )


def create_keycloak_identity_provider(
    *,
    identity_provider: Optional[IdentityProviderBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IdentityProviderBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_KEYCLOAK", **config_overrides)
    return _identity_provider_factory(
        env_prefix="INTERGRAX_KEYCLOAK",
        provider="keycloak",
        default_url=config.base_url or "http://127.0.0.1:8088",
        health_path="/health/ready",
        identity_provider=identity_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_workos_identity_provider(
    *,
    identity_provider: Optional[IdentityProviderBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IdentityProviderBackend:
    return _identity_provider_factory(
        env_prefix="INTERGRAX_WORKOS",
        provider="workos",
        default_url="https://api.workos.com",
        health_path="/health",
        identity_provider=identity_provider,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


# --- H-INT-13: GitOps CI/CD ---


def create_argocd_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    return _ci_cd_factory(
        env_prefix="INTERGRAX_ARGOCD",
        provider="argocd",
        default_url="http://127.0.0.1:8080",
        health_path="/healthz",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_buildkite_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    return _ci_cd_factory(
        env_prefix="INTERGRAX_BUILDKITE",
        provider="buildkite",
        default_url="https://api.buildkite.com",
        health_path="/v2/user",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_jenkins_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_JENKINS", **config_overrides)
    return _ci_cd_factory(
        env_prefix="INTERGRAX_JENKINS",
        provider="jenkins",
        default_url=config.base_url or "http://127.0.0.1:8080",
        health_path="/api/json",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


# --- H-INT-14: speech providers ---


def create_elevenlabs_speech_provider(
    *,
    speech_provider: Optional[SpeechProviderBackend] = None,
    adapter: Optional[SpeechAdapter] = None,
    adapter_factory: Optional[Callable[[], SpeechAdapter]] = None,
    **config_overrides: object,
) -> SpeechProviderBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_ELEVENLABS", **config_overrides)

    def _open() -> SpeechAdapter:
        from intergrax.speech_adapters.providers.elevenlabs_speech import ElevenLabsSpeechAdapter

        return ElevenLabsSpeechAdapter(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.elevenlabs.io/v1",
        )

    return _resolve(
        implementation=speech_provider,
        backend=adapter,
        backend_factory=adapter_factory,
        open_fn=_open,
        adapter_fn=lambda a: SpeechAdapterBackend(a, slug="elevenlabs"),
    )


def create_deepgram_speech_provider(
    *,
    speech_provider: Optional[SpeechProviderBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SpeechProviderBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_DEEPGRAM", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.deepgram.com")

        class _Client:
            def synthesize(self, text: str, *, voice_id: str = "default") -> dict[str, Any]:
                response = http.post(
                    "/v1/speak",
                    json={"text": text, "voice": voice_id},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"audio_uri": ""}

            def transcribe(self, audio_uri: str) -> dict[str, Any]:
                response = http.post(
                    "/v1/listen",
                    json={"url": audio_uri},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"transcript": ""}

            def health(self) -> bool:
                return http_ping_ok(http, path="/v1/projects")

        return _Client()

    return _resolve(
        implementation=speech_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpSpeechProviderBackend(c, provider="deepgram"),
    )


# --- H-INT-15: enterprise ops ---


def create_newrelic_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_NEWRELIC",
        provider="newrelic",
        default_url="https://api.newrelic.com",
        instant_path="/v2/accounts/query",
        range_path="/v2/accounts/query",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_splunk_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_SPLUNK",
        provider="splunk",
        default_url="https://localhost:8089",
        instant_path="/services/search/v2/jobs/export",
        range_path="/services/search/v2/jobs/export",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_zendesk_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    return _issue_tracker_factory(
        env_prefix="INTERGRAX_ZENDESK",
        provider="zendesk",
        search_path="/api/v2/search.json",
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_statsig_feature_flag(
    *,
    feature_flag: Optional[FeatureFlagBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> FeatureFlagBackend:
    return _feature_flag_factory(
        env_prefix="INTERGRAX_STATSIG",
        provider="statsig",
        feature_flag=feature_flag,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


# --- H-INT-16: data / workflow ---


def create_prefect_workflow_orchestrator(
    *,
    workflow_orchestrator: Optional[WorkflowOrchestratorBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WorkflowOrchestratorBackend:
    return _workflow_orchestrator_factory(
        env_prefix="INTERGRAX_PREFECT",
        provider="prefect",
        default_url="http://127.0.0.1:4200",
        health_path="/api/health",
        workflow_orchestrator=workflow_orchestrator,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_airflow_workflow_orchestrator(
    *,
    workflow_orchestrator: Optional[WorkflowOrchestratorBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> WorkflowOrchestratorBackend:
    return _workflow_orchestrator_factory(
        env_prefix="INTERGRAX_AIRFLOW",
        provider="airflow",
        default_url="http://127.0.0.1:8086",
        health_path="/health",
        workflow_orchestrator=workflow_orchestrator,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_typesense_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    config = VectorIntegrationConfig.from_env("INTERGRAX_TYPESENSE", **config_overrides)

    def _open() -> VectorStore:
        http_config = HttpIntegrationConfig.from_env("INTERGRAX_TYPESENSE", **config_overrides)
        http = _open_httpx_client(
            http_config,
            default_url=config.url or http_config.base_url or "http://127.0.0.1:8108",
        )
        return _TypesenseHttpVectorStore(http, collection=config.collection)

    inner = client if client is not None else (client_factory() if client_factory else _open())
    if isinstance(inner, VectorStore):
        return RestVectorStoreIntegration(config, inner)
    return RestVectorStoreIntegration(config, _TypesenseHttpVectorStore(inner, collection=config.collection))


def create_neon_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    return _sql_store_factory(
        prefix="INTERGRAX_NEON",
        factory_name="create_neon_relational_store",
        driver="psycopg",
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        **config_overrides,
    )


def create_pulsar_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    kv_store: Optional[Any] = None,
    **config_overrides: object,
) -> MessageBus:
    if message_bus is not None:
        return message_bus
    from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus

    overrides = dict(config_overrides)
    config = QueueIntegrationConfig.from_env("INTERGRAX_PULSAR", **config_overrides)
    if config.connection_string:
        overrides["bootstrap_servers"] = config.connection_string
    if config.topic:
        overrides["topic"] = config.topic
    return create_kafka_message_bus(kv_store=kv_store, **overrides)


def create_confluent_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    kv_store: Optional[Any] = None,
    **config_overrides: object,
) -> MessageBus:
    if message_bus is not None:
        return message_bus
    from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus

    overrides = dict(config_overrides)
    config = QueueIntegrationConfig.from_env("INTERGRAX_CONFLUENT", **config_overrides)
    if config.connection_string:
        overrides["bootstrap_servers"] = config.connection_string
    if config.topic:
        overrides["topic"] = config.topic
    return create_kafka_message_bus(kv_store=kv_store, **overrides)


# --- H-INT-17: reserve slugs ---


def create_algolia_search_provider(
    *,
    search_provider: Optional[SearchProvider] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SearchProvider:
    config = HttpIntegrationConfig.from_env("INTERGRAX_ALGOLIA", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://algolia.net")
        index = config.repo or config.org or "intergrax"

        class _Client:
            def search(self, query: str, limit: int) -> dict[str, Any]:
                response = http.post(
                    f"/1/indexes/{index}/query",
                    json={"params": f"query={query}&hitsPerPage={limit}"},
                )
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"hits": []}

            def health(self) -> bool:
                return http_ping_ok(http, path="/1/indexes")

        return _Client()

    def _adapter(c: Any) -> SearchProvider:
        return build_rest_search_provider(
            provider="algolia",
            search_fn=lambda q, limit: c.search(q, limit),
            hits_fn=_algolia_hits,
        )

    return _resolve(
        implementation=search_provider,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


def create_backblaze_b2_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    s3_client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage

    endpoint = str(config_overrides.get("endpoint") or os.environ.get("INTERGRAX_BACKBLAZE_B2_ENDPOINT", "")).strip()
    access_key = str(config_overrides.get("access_key") or os.environ.get("INTERGRAX_BACKBLAZE_B2_ACCESS_KEY", "")).strip()
    secret_key = str(config_overrides.get("secret_key") or os.environ.get("INTERGRAX_BACKBLAZE_B2_SECRET_KEY", "")).strip()
    bucket = str(config_overrides.get("bucket") or os.environ.get("INTERGRAX_BACKBLAZE_B2_BUCKET", "intergrax")).strip()

    class _B2Config:
        def __init__(self) -> None:
            self.bucket = bucket
            self.prefix = str(config_overrides.get("prefix") or os.environ.get("INTERGRAX_BACKBLAZE_B2_PREFIX", ""))

        def object_key(self, key: str) -> str:
            normalized = key.lstrip("/")
            prefix = self.prefix.strip("/")
            return f"{prefix}/{normalized}" if prefix else normalized

        def require_bucket(self) -> str:
            if not self.bucket:
                from intergrax.integrations.contracts.base import IntegrationConfigurationError

                raise IntegrationConfigurationError("Backblaze B2 requires bucket (INTERGRAX_BACKBLAZE_B2_BUCKET)")
            return self.bucket

    config = _B2Config()

    def _open() -> Any:
        try:
            import boto3
        except ImportError as exc:
            from intergrax.integrations.contracts.base import IntegrationConfigurationError

            raise IntegrationConfigurationError("Backblaze B2 requires boto3") from exc
        return boto3.client(
            "s3",
            endpoint_url=endpoint or "https://s3.us-west-002.backblazeb2.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-west-002",
        )

    resolved = s3_client if s3_client is not None else (client_factory() if client_factory else _open())
    return CatalogObjectStorage(config, resolved, factory_name="create_backblaze_b2_object_storage")


def create_triton_vision_serving(
    *,
    vision_serving: Optional[VisionServingBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VisionServingBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_TRITON", **config_overrides)
    return _vision_serving_factory(
        env_prefix="INTERGRAX_TRITON",
        provider="triton",
        default_url=config.base_url or "http://127.0.0.1:8000",
        health_path="/v2/health/ready",
        vision_serving=vision_serving,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_replicate_ml_inference_host(
    *,
    ml_inference_host: Optional[MlInferenceHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> MlInferenceHostBackend:
    return _ml_inference_factory(
        env_prefix="INTERGRAX_REPLICATE",
        provider="replicate",
        default_url="https://api.replicate.com",
        health_path="/v1/account",
        ml_inference_host=ml_inference_host,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_stripe_billing_meter(
    *,
    billing_meter: Optional[BillingMeterBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> BillingMeterBackend:
    return _billing_meter_factory(
        env_prefix="INTERGRAX_STRIPE",
        provider="stripe",
        default_url="https://api.stripe.com",
        health_path="/v1/account",
        billing_meter=billing_meter,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_salesforce_crm(
    *,
    crm: Optional[CrmBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CrmBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_SALESFORCE", **config_overrides)
    return _crm_factory(
        env_prefix="INTERGRAX_SALESFORCE",
        provider="salesforce",
        default_url=config.base_url or "https://login.salesforce.com",
        health_path="/services/data",
        crm=crm,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_hubspot_crm(
    *,
    crm: Optional[CrmBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CrmBackend:
    return _crm_factory(
        env_prefix="INTERGRAX_HUBSPOT",
        provider="hubspot",
        default_url="https://api.hubapi.com",
        health_path="/crm/v3/objects/contacts",
        crm=crm,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


__all__ = [
    "create_airflow_workflow_orchestrator",
    "create_algolia_search_provider",
    "create_argocd_ci_cd",
    "create_auth0_identity_provider",
    "create_backblaze_b2_object_storage",
    "create_buildkite_ci_cd",
    "create_confluent_message_bus",
    "create_daytona_sandbox_host",
    "create_deepgram_speech_provider",
    "create_e2b_sandbox_host",
    "create_elevenlabs_speech_provider",
    "create_hubspot_crm",
    "create_infisical_secrets_store",
    "create_jenkins_ci_cd",
    "create_keycloak_identity_provider",
    "create_modal_sandbox_host",
    "create_neon_relational_store",
    "create_newrelic_observability_backend",
    "create_prefect_workflow_orchestrator",
    "create_pulsar_message_bus",
    "create_replicate_ml_inference_host",
    "create_salesforce_crm",
    "create_semgrep_security_scanner",
    "create_snyk_security_scanner",
    "create_splunk_observability_backend",
    "create_statsig_feature_flag",
    "create_stripe_billing_meter",
    "create_triton_vision_serving",
    "create_trivy_security_scanner",
    "create_typesense_vector_store",
    "create_workos_identity_provider",
    "create_zendesk_issue_tracker",
]
