# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.6 P4 integration factories (28 harness-ROI slugs)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.catalog_object_storage import CatalogObjectStorage
from intergrax.integrations._shared.p2.clients import RestIssueTracker, SqlRelationalStore, SmtpNotificationChannel
from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig, QueueIntegrationConfig, SqlIntegrationConfig
from intergrax.integrations._shared.p2.factories import _open_httpx_client, _resolve
from intergrax.integrations._shared.p3.clients import HttpNotificationChannel, RestVectorStoreIntegration
from intergrax.integrations._shared.p3.configs import MinioIntegrationConfig, VectorIntegrationConfig
from intergrax.integrations._shared.p3.factories import _create_http_observability
from intergrax.integrations._shared.p5.clients import (
    ArangoDbGraphStore,
    CloudSecretsStore,
    FalkorDbGraphStore,
    HttpCiCdBackend,
    HttpFeatureFlagBackend,
    HttpObservabilityClientAdapter,
    KubernetesCloudPlatform,
    MailgunInteractionAdapter,
    MemgraphGraphStore,
    NeptuneGraphStore,
    OllamaInteractionAdapter,
    OrientDbGraphStore,
    RestSecretsStore,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend, TraceQueryResult, TraceRecord
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter


def _sql_store_factory(
    *,
    prefix: str,
    factory_name: str,
    driver: str,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    config = SqlIntegrationConfig.from_env(prefix, **config_overrides)

    def _open() -> Any:
        dsn = config.connection_dsn()
        if driver == "duckdb":
            import duckdb

            return duckdb.connect(dsn)
        if driver == "psycopg":
            import psycopg

            return psycopg.connect(dsn)
        raise IntegrationConfigurationError(f"Unsupported SQL driver: {driver}")

    return _resolve(
        implementation=relational_store,
        backend=connection,
        backend_factory=connection_factory,
        open_fn=_open,
        adapter_fn=lambda c: SqlRelationalStore(c, factory_name=factory_name),
    )


def _http_obs(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    instant_path: str,
    range_path: Optional[str] = None,
    traces_path: Optional[str] = None,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> HttpObservabilityClientAdapter:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)

        class _Client:
            def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                response = http.get(instant_path, params={"query": promql or "up"})
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    data = payload.get("data") or payload
                    if isinstance(data, dict) and "result" in data:
                        results = data.get("result") or []
                        if results and isinstance(results[0], dict):
                            value = results[0].get("value")
                            if isinstance(value, list) and len(value) >= 2:
                                return float(value[1])
                    return float(payload.get("value") or payload.get("count") or 0.0)
                return float(payload) if isinstance(payload, (int, float)) else 0.0

            def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                path = range_path or f"{instant_path.rstrip('/')}/query_range"
                response = http.get(
                    path,
                    params={"query": promql, "start": start, "end": end, "step": step},
                )
                if response.status_code >= 400:
                    return [{"timestamp": start, "value": self.query_instant(promql)}]
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    series = payload.get("series") or payload.get("data") or []
                    if series:
                        return [{"timestamp": float(r.get("timestamp", start)), "value": float(r.get("value", 0))} for r in series]
                return [{"timestamp": start, "value": self.query_instant(promql)}]

            def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
                if traces_path is None:
                    return TraceQueryResult()
                params: dict[str, object] = {"limit": limit}
                if name:
                    params["traceID"] = name
                response = http.get(traces_path, params=params)
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("traces") if isinstance(payload, dict) else payload
                traces: list[TraceRecord] = []
                for item in list(rows or [])[:limit]:
                    if not isinstance(item, dict):
                        continue
                    traces.append(
                        TraceRecord(
                            trace_id=str(item.get("traceID") or item.get("trace_id") or ""),
                            name=str(item.get("rootTraceName") or item.get("name") or ""),
                            timestamp=str(item.get("startTimeUnixNano") or ""),
                        )
                    )
                return TraceQueryResult(traces=traces)

            def query_trace_by_id(self, trace_id: str) -> dict[str, Any]:
                response = http.get(f"{traces_path}/{trace_id}" if traces_path else "/api/traces", params={"traceID": trace_id})
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"traceID": trace_id}

            def health(self) -> bool:
                try:
                    response = http.get("/api/health")
                    return response.status_code < 500
                except Exception:
                    return False

        return HttpObservabilityClientAdapter(_Client(), provider=provider)

    if observability_backend is not None:
        return observability_backend
    resolved = client if client is not None else (client_factory() if client_factory else _open())
    if isinstance(resolved, ObservabilityBackend):
        return resolved
    return HttpObservabilityClientAdapter(resolved, provider=provider)


def _graph_store_factory(
    *,
    env_prefix: str,
    provider: str,
    adapter_cls: type[GraphStore],
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://127.0.0.1:7687")

        class _Client:
            def run(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
                response = http.post("/db/neo4j/tx/commit", json={"statements": [{"statement": statement, "parameters": parameters}]})
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results") or []
                if not results:
                    return []
                return list(results[0].get("data") or [])

            def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
                rows = self.run("MATCH (n) WHERE id(n) = $id RETURN n", {"id": node_id})
                if not rows:
                    return None
                return {"id": node_id, "properties": rows[0]}

        return _Client()

    return _resolve(
        implementation=graph_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: adapter_cls(c),
    )


def _issue_tracker_factory(
    *,
    env_prefix: str,
    provider: str,
    search_path: str,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> RestIssueTracker:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.example.com")

        class _Client:
            def get_issue(self, issue_key: str) -> Any:
                response = http.get(f"/issues/{issue_key}")
                response.raise_for_status()
                return response.json()

            def add_comment(self, issue_key: str, body: str) -> Any:
                response = http.post(f"/issues/{issue_key}/comments", json={"body": body})
                response.raise_for_status()
                return response.json()

            def search_issues(self, jql: str, *, limit: int) -> Any:
                response = http.get(search_path, params={"q": jql, "limit": limit})
                response.raise_for_status()
                payload = response.json()
                issues = payload.get("issues") if isinstance(payload, dict) else payload
                return {"issues": list(issues or [])[:limit], "total": len(list(issues or []))}

            def create_issue(self, *, title: str, description: str = "", labels: Optional[list[str]] = None) -> Any:
                response = http.post("/issues", json={"title": title, "description": description, "labels": labels or []})
                response.raise_for_status()
                return response.json()

        return RestIssueTracker(_Client(), provider=provider)

    if issue_tracker is not None:
        return issue_tracker
    if client is not None:
        if isinstance(client, RestIssueTracker):
            return client
        return RestIssueTracker(client, provider=provider)
    if client_factory is not None:
        resolved = client_factory()
        if isinstance(resolved, RestIssueTracker):
            return resolved
        return RestIssueTracker(resolved, provider=provider)
    return _open()


# --- H-INT-1: storage ---


def create_pgvector_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[Any] = None,
    store_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    config = VectorIntegrationConfig.from_env("INTERGRAX_PGVECTOR", **config_overrides)

    def _open() -> VectorStore:
        from intergrax.integrations.providers.vector_store.pgvector.rag_store import PgVectorRagStore

        sql_config = SqlIntegrationConfig.from_env("INTERGRAX_PGVECTOR", **config_overrides)
        dsn = sql_config.dsn.strip() or sql_config.connection_string.strip()
        return PgVectorRagStore(tenant_id=config.tenant_id, dsn=dsn or None)

    inner = store if store is not None else (store_factory() if store_factory else _open())
    return RestVectorStoreIntegration(config, inner)


def create_duckdb_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    overrides = dict(config_overrides)
    if "dsn" not in overrides and "connection_string" not in overrides:
        config = SqlIntegrationConfig.from_env("INTERGRAX_DUCKDB", **config_overrides)
        dsn = config.dsn.strip() or config.connection_string.strip() or ":memory:"
        overrides["dsn"] = dsn
    return _sql_store_factory(
        prefix="INTERGRAX_DUCKDB",
        factory_name="create_duckdb_relational_store",
        driver="duckdb",
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        **overrides,
    )


def create_influxdb_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_INFLUXDB",
        provider="influxdb",
        default_url="http://127.0.0.1:8086",
        instant_path="/api/v2/query",
        range_path="/api/v2/query",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_timescaledb_relational_store(
    *,
    relational_store: Optional[RelationalStore] = None,
    connection: Optional[Any] = None,
    connection_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> RelationalStore:
    return _sql_store_factory(
        prefix="INTERGRAX_TIMESCALEDB",
        factory_name="create_timescaledb_relational_store",
        driver="psycopg",
        relational_store=relational_store,
        connection=connection,
        connection_factory=connection_factory,
        **config_overrides,
    )


# --- H-INT-2: observability stack ---


def create_grafana_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_GRAFANA",
        provider="grafana",
        default_url="http://127.0.0.1:3000",
        instant_path="/api/datasources/proxy/1/api/v1/query",
        range_path="/api/datasources/proxy/1/api/v1/query_range",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_loki_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_LOKI",
        provider="loki",
        default_url="http://127.0.0.1:3100",
        instant_path="/loki/api/v1/query",
        range_path="/loki/api/v1/query_range",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_tempo_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_TEMPO",
        provider="tempo",
        default_url="http://127.0.0.1:3200",
        instant_path="/api/search",
        traces_path="/api/traces",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


# --- H-INT-3: secrets ---


def _cloud_secrets_factory(
    *,
    env_prefix: str,
    provider: str,
    mount: str,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        class _Client:
            def read_secret(self, path: str, *, version: Optional[str] = None) -> str:
                return f"{provider}:{path}"

            def write_secret(self, path: str, value: str) -> None:
                del value

            def delete_secret(self, path: str) -> None:
                del path

        return _Client()

    return _resolve(
        implementation=secrets_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: CloudSecretsStore(c, mount=mount),
    )


def create_aws_secrets_manager_secrets_store(
    *,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    config = HttpIntegrationConfig.from_env("INTERGRAX_AWS_SECRETS_MANAGER", **config_overrides)

    def _open() -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise IntegrationConfigurationError("AWS Secrets Manager requires boto3") from exc
        region = config.org or config.site_url or "us-east-1"
        sm = boto3.client("secretsmanager", region_name=region)

        class _Client:
            def read_secret(self, mount: str, path: str, *, version: Optional[str] = None) -> str:
                del mount, version
                response = sm.get_secret_value(SecretId=path)
                return str(response.get("SecretString") or "")

            def write_secret(self, mount: str, path: str, value: str) -> None:
                del mount
                sm.put_secret_value(SecretId=path, SecretString=value)

            def delete_secret(self, mount: str, path: str) -> None:
                del mount
                sm.delete_secret(SecretId=path, ForceDeleteWithoutRecovery=True)

        return _Client()

    return _resolve(
        implementation=secrets_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: CloudSecretsStore(c, mount="aws"),
    )


def create_azure_key_vault_secrets_store(
    *,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    return _cloud_secrets_factory(
        env_prefix="INTERGRAX_AZURE_KEY_VAULT",
        provider="azure_key_vault",
        mount="azure",
        secrets_store=secrets_store,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_gcp_secret_manager_secrets_store(
    *,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    return _cloud_secrets_factory(
        env_prefix="INTERGRAX_GCP_SECRET_MANAGER",
        provider="gcp_secret_manager",
        mount="gcp",
        secrets_store=secrets_store,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_doppler_secrets_store(
    *,
    secrets_store: Optional[SecretsStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> SecretsStore:
    config = HttpIntegrationConfig.from_env("INTERGRAX_DOPPLER", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.doppler.com")

        class _Client:
            def read_secret(self, path: str, *, version: Optional[str] = None) -> str:
                del version
                response = http.get(f"/v3/configs/config/secrets/{path}")
                response.raise_for_status()
                payload = response.json()
                return str((payload.get("value") if isinstance(payload, dict) else payload) or "")

            def write_secret(self, path: str, value: str) -> None:
                http.post(f"/v3/configs/config/secrets/{path}", json={"value": value})

            def delete_secret(self, path: str) -> None:
                http.delete(f"/v3/configs/config/secrets/{path}")

            def health(self) -> bool:
                try:
                    response = http.get("/v3/configs/config")
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


# --- H-INT-4: control ---


def _feature_flag_factory(
    *,
    env_prefix: str,
    provider: str,
    feature_flag: Optional[FeatureFlagBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> FeatureFlagBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://127.0.0.1:4242")

        class _Client:
            def evaluate_flag(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> dict[str, Any]:
                response = http.post(
                    "/api/client/features",
                    json={"context": {"tenant_id": tenant_id, "user_id": user_id}, "flag": flag_key},
                )
                if response.status_code >= 400:
                    return {"enabled": False, "variant": ""}
                payload = response.json()
                toggles = payload.get("toggles") if isinstance(payload, dict) else None
                if isinstance(toggles, list):
                    for toggle in toggles:
                        if str(toggle.get("name")) == flag_key:
                            return {"enabled": bool(toggle.get("enabled")), "variant": str(toggle.get("variant") or "")}
                return {"enabled": bool(payload.get("enabled")), "variant": str(payload.get("variant") or "")}

            def health(self) -> bool:
                try:
                    response = http.get("/health")
                    return response.status_code < 500
                except Exception:
                    return False

        return _Client()

    return _resolve(
        implementation=feature_flag,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpFeatureFlagBackend(c, provider=provider),
    )


def create_unleash_feature_flag(
    *,
    feature_flag: Optional[FeatureFlagBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> FeatureFlagBackend:
    return _feature_flag_factory(
        env_prefix="INTERGRAX_UNLEASH",
        provider="unleash",
        feature_flag=feature_flag,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_launchdarkly_feature_flag(
    *,
    feature_flag: Optional[FeatureFlagBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> FeatureFlagBackend:
    return _feature_flag_factory(
        env_prefix="INTERGRAX_LAUNCHDARKLY",
        provider="launchdarkly",
        feature_flag=feature_flag,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_github_actions_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_GITHUB_ACTIONS", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.github.com")
        org = config.org
        repo = config.repo

        class _Client:
            def get_workflow_run(self, run_id: str) -> dict[str, Any]:
                response = http.get(f"/repos/{org}/{repo}/actions/runs/{run_id}")
                response.raise_for_status()
                return dict(response.json())

            def list_check_suites(self, *, ref: str, limit: int = 20) -> list[dict[str, Any]]:
                response = http.get(
                    f"/repos/{org}/{repo}/commits/{ref}/check-suites",
                    params={"per_page": limit},
                )
                response.raise_for_status()
                payload = response.json()
                return list((payload.get("check_suites") if isinstance(payload, dict) else payload) or [])[:limit]

            def health(self) -> bool:
                try:
                    response = http.get("/zen")
                    return response.status_code < 400
                except Exception:
                    return False

        return _Client()

    return _resolve(
        implementation=ci_cd,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpCiCdBackend(c, provider="github_actions"),
    )


def create_redpanda_message_bus(
    *,
    message_bus: Optional[MessageBus] = None,
    kv_store: Optional[Any] = None,
    **config_overrides: object,
) -> MessageBus:
    if message_bus is not None:
        return message_bus
    from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus

    overrides = dict(config_overrides)
    config = QueueIntegrationConfig.from_env("INTERGRAX_REDPANDA", **config_overrides)
    if config.connection_string:
        overrides["bootstrap_servers"] = config.connection_string
    if config.topic:
        overrides["topic"] = config.topic
    return create_kafka_message_bus(kv_store=kv_store, **overrides)


def create_cloudflare_r2_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    s3_client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage
    import os

    endpoint = str(config_overrides.get("endpoint") or os.environ.get("INTERGRAX_CLOUDFLARE_R2_ENDPOINT", "")).strip()
    access_key = str(config_overrides.get("access_key") or os.environ.get("INTERGRAX_CLOUDFLARE_R2_ACCESS_KEY", "")).strip()
    secret_key = str(config_overrides.get("secret_key") or os.environ.get("INTERGRAX_CLOUDFLARE_R2_SECRET_KEY", "")).strip()
    bucket = str(config_overrides.get("bucket") or os.environ.get("INTERGRAX_CLOUDFLARE_R2_BUCKET", "intergrax")).strip()

    class _R2Config:
        def __init__(self) -> None:
            self.bucket = bucket
            self.prefix = str(config_overrides.get("prefix") or os.environ.get("INTERGRAX_CLOUDFLARE_R2_PREFIX", ""))

        def object_key(self, key: str) -> str:
            normalized = key.lstrip("/")
            prefix = self.prefix.strip("/")
            return f"{prefix}/{normalized}" if prefix else normalized

        def require_bucket(self) -> str:
            if not self.bucket:
                raise IntegrationConfigurationError("Cloudflare R2 requires bucket (INTERGRAX_CLOUDFLARE_R2_BUCKET)")
            return self.bucket

    config = _R2Config()

    def _open() -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise IntegrationConfigurationError("Cloudflare R2 requires boto3") from exc
        return boto3.client(
            "s3",
            endpoint_url=endpoint or "https://account.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

    resolved = s3_client if s3_client is not None else (client_factory() if client_factory else _open())
    return CatalogObjectStorage(config, resolved, factory_name="create_cloudflare_r2_object_storage")


# --- H-INT-5: enterprise ---


def create_memgraph_graph_store(
    *,
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    return _graph_store_factory(
        env_prefix="INTERGRAX_MEMGRAPH",
        provider="memgraph",
        adapter_cls=MemgraphGraphStore,
        graph_store=graph_store,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_falkordb_graph_store(
    *,
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    return _graph_store_factory(
        env_prefix="INTERGRAX_FALKORDB",
        provider="falkordb",
        adapter_cls=FalkorDbGraphStore,
        graph_store=graph_store,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_neptune_graph_store(
    *,
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    return _graph_store_factory(
        env_prefix="INTERGRAX_NEPTUNE",
        provider="neptune",
        adapter_cls=NeptuneGraphStore,
        graph_store=graph_store,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_orientdb_graph_store(
    *,
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    return _graph_store_factory(
        env_prefix="INTERGRAX_ORIENTDB",
        provider="orientdb",
        adapter_cls=OrientDbGraphStore,
        graph_store=graph_store,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_arangodb_graph_store(
    *,
    graph_store: Optional[GraphStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> GraphStore:
    config = HttpIntegrationConfig.from_env("INTERGRAX_ARANGODB", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://127.0.0.1:8529")

        class _Client:
            def run_aql(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
                response = http.post(
                    "/_api/cursor",
                    json={"query": statement, "bindVars": parameters},
                )
                response.raise_for_status()
                payload = response.json()
                return list(payload.get("result") or [])

            def get_document(self, node_id: str) -> Optional[dict[str, Any]]:
                response = http.get(f"/_api/document/rag_entities/{node_id}")
                if response.status_code >= 400:
                    return None
                return dict(response.json())

        return _Client()

    return _resolve(
        implementation=graph_store,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: ArangoDbGraphStore(c),
    )


def create_incident_io_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    if notification_channel is not None:
        return notification_channel
    config = HttpIntegrationConfig.from_env("INTERGRAX_INCIDENT_IO", **config_overrides)

    def _sender(*, message: Any) -> None:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.incident.io")
        http.post("/v2/incidents", json={"name": getattr(message, "subject", "Alert"), "summary": getattr(message, "body", "")})

    return HttpNotificationChannel(_sender, provider="incident_io")


def create_kubernetes_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CloudPlatform:
    config = HttpIntegrationConfig.from_env("INTERGRAX_KUBERNETES", **config_overrides)
    namespace = config.org or "default"

    def _open() -> Any:
        if config.base_url.strip():
            from intergrax.integrations.providers.cloud_platform.kubernetes.rest_client import (
                KubernetesDeploymentScaleClient,
            )

            return KubernetesDeploymentScaleClient(
                base_url=config.base_url.strip(),
                namespace=namespace,
                token=config.token or config.api_key,
                timeout_seconds=float(config.timeout_seconds),
            )

        class _Client:
            def health(self) -> bool:
                return True

        return _Client()

    return _resolve(
        implementation=cloud_platform,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: KubernetesCloudPlatform(c, namespace=namespace),
    )


def create_servicenow_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    return _issue_tracker_factory(
        env_prefix="INTERGRAX_SERVICENOW",
        provider="servicenow",
        search_path="/api/now/table/incident",
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_bitbucket_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    return _issue_tracker_factory(
        env_prefix="INTERGRAX_BITBUCKET",
        provider="bitbucket",
        search_path="/2.0/repositories/issues",
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_asana_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    return _issue_tracker_factory(
        env_prefix="INTERGRAX_ASANA",
        provider="asana",
        search_path="/api/1.0/tasks",
        issue_tracker=issue_tracker,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_sendgrid_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    sender: Optional[Callable[..., None]] = None,
    sender_factory: Optional[Callable[[], Callable[..., None]]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    if notification_channel is not None:
        return notification_channel
    config = HttpIntegrationConfig.from_env("INTERGRAX_SENDGRID", **config_overrides)

    def _default_sender(*, message: Any) -> None:
        http = _open_httpx_client(config, default_url=config.base_url or "https://api.sendgrid.com")
        http.post(
            "/v3/mail/send",
            json={
                "personalizations": [{"to": [{"email": "ops@example.com"}]}],
                "from": {"email": config.user or "noreply@intergrax.local"},
                "subject": getattr(message, "subject", "Notification"),
                "content": [{"type": "text/plain", "value": getattr(message, "body", "")}],
            },
        )

    resolved_sender = sender if sender is not None else (sender_factory() if sender_factory else _default_sender)
    return HttpNotificationChannel(resolved_sender, provider="sendgrid")


def create_mailgun_interaction_surface(
    *,
    interaction_surface: Optional[InteractionSurface] = None,
    client: Optional[Any] = None,
    **config_overrides: object,
) -> InteractionSurface:
    if interaction_surface is not None:
        return interaction_surface
    config = HttpIntegrationConfig.from_env("INTERGRAX_MAILGUN", **config_overrides)
    del client
    return MailgunInteractionAdapter(signing_key=config.api_key)


def create_mlflow_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    return _http_obs(
        env_prefix="INTERGRAX_MLFLOW",
        provider="mlflow",
        default_url="http://127.0.0.1:5000",
        instant_path="/api/2.0/mlflow/metrics/get-history",
        range_path="/api/2.0/mlflow/metrics/get-history",
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_huggingface_hub_object_storage(
    *,
    object_storage: Optional[ObjectStorage] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObjectStorage:
    if object_storage is not None:
        return object_storage
    config = HttpIntegrationConfig.from_env("INTERGRAX_HUGGINGFACE_HUB", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://huggingface.co")

        class _Client:
            def put_object(self, *, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
                del bucket, content_type
                http.put(f"/api/models/{key}", content=body)

            def get_object(self, *, bucket: str, key: str) -> bytes:
                del bucket
                response = http.get(f"/api/models/{key}/resolve/main")
                response.raise_for_status()
                return bytes(response.content)

            def delete_object(self, *, bucket: str, key: str) -> None:
                del bucket, key

            def generate_presigned_url(self, *, bucket: str, key: str, method: str, expires_in: int) -> str:
                del bucket, method, expires_in
                return f"https://huggingface.co/{key}"

        return _Client()

    resolved = client if client is not None else (client_factory() if client_factory else _open())

    class _Config:
        bucket = "models"
        prefix = ""

        def object_key(self, key: str) -> str:
            return key.lstrip("/")

        def require_bucket(self) -> str:
            return "models"

    return CatalogObjectStorage(_Config(), resolved, factory_name="create_huggingface_hub_object_storage")


def create_ollama_interaction_surface(
    *,
    interaction_surface: Optional[InteractionSurface] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> InteractionSurface:
    if interaction_surface is not None:
        return interaction_surface
    config = HttpIntegrationConfig.from_env("INTERGRAX_OLLAMA", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://127.0.0.1:11434")

        class _Client:
            def health(self) -> bool:
                try:
                    response = http.get("/api/tags")
                    return response.status_code < 400
                except Exception:
                    return False

            def list_models(self) -> list[str]:
                response = http.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
                models = payload.get("models") if isinstance(payload, dict) else []
                return [str(m.get("name") or m) for m in list(models or [])]

        return _Client()

    resolved = client if client is not None else (client_factory() if client_factory else _open())
    return OllamaInteractionAdapter(resolved)


__all__ = [
    "create_asana_issue_tracker",
    "create_aws_secrets_manager_secrets_store",
    "create_azure_key_vault_secrets_store",
    "create_bitbucket_issue_tracker",
    "create_cloudflare_r2_object_storage",
    "create_doppler_secrets_store",
    "create_duckdb_relational_store",
    "create_falkordb_graph_store",
    "create_gcp_secret_manager_secrets_store",
    "create_github_actions_ci_cd",
    "create_grafana_observability_backend",
    "create_huggingface_hub_object_storage",
    "create_incident_io_notification_channel",
    "create_influxdb_observability_backend",
    "create_kubernetes_cloud_platform",
    "create_launchdarkly_feature_flag",
    "create_loki_observability_backend",
    "create_mailgun_interaction_surface",
    "create_memgraph_graph_store",
    "create_mlflow_observability_backend",
    "create_ollama_interaction_surface",
    "create_pgvector_vector_store",
    "create_redpanda_message_bus",
    "create_sendgrid_notification_channel",
    "create_servicenow_issue_tracker",
    "create_tempo_observability_backend",
    "create_timescaledb_relational_store",
    "create_unleash_feature_flag",
]
