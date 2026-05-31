# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.8 harness gap integration factories."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.p2.clients import RestIssueTracker
from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations._shared.p2.factories import _open_httpx_client, _resolve
from intergrax.integrations._shared.p3.clients import HttpNotificationChannel, RestVectorStoreIntegration
from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations._shared.p3.factories import _create_http_observability
from intergrax.integrations._shared.p4.configs import OpenSearchIntegrationConfig
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend, TraceQueryResult, TraceRecord
from intergrax.integrations.contracts.vector_store import VectorStore


def _trace_rows(payload: Any, *, limit: int) -> TraceQueryResult:
    rows = payload.get("data") if isinstance(payload, dict) else payload
    traces: list[TraceRecord] = []
    for item in list(rows or [])[:limit]:
        if not isinstance(item, dict):
            continue
        traces.append(
            TraceRecord(
                trace_id=str(item.get("id") or item.get("trace_id") or item.get("traceId") or ""),
                name=str(item.get("name") or item.get("run_name") or ""),
                timestamp=str(item.get("timestamp") or item.get("created_at") or item.get("createdAt") or "") or None,
                metadata={k: v for k, v in item.items() if k not in {"id", "trace_id", "traceId", "name", "timestamp", "created_at", "createdAt"}},
            )
        )
    return TraceQueryResult(traces=traces)


def _http_obs_factory(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    instant_path: str = "/metrics",
    traces_path: Optional[str] = None,
) -> Callable[..., ObservabilityBackend]:
    def factory(
        *,
        observability_backend: Optional[ObservabilityBackend] = None,
        client: Optional[Any] = None,
        client_factory: Optional[Callable[[], Any]] = None,
        **config_overrides: object,
    ) -> ObservabilityBackend:
        config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

        def _open() -> Any:
            http = _open_httpx_client(config, default_url=config.base_url or default_url)

            class _Client:
                def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                    response = http.get(instant_path, params={"query": promql or "up"})
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict):
                        return float(payload.get("value") or payload.get("count") or len(payload.get("data") or []))
                    return float(payload) if isinstance(payload, (int, float)) else 0.0

                def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                    response = http.get(
                        f"{instant_path}/range",
                        params={"query": promql, "start": start, "end": end, "step": step},
                    )
                    if response.status_code >= 400:
                        return [{"timestamp": start, "value": self.query_instant(promql)}]
                    response.raise_for_status()
                    return list(response.json().get("series") or [])

                def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
                    if traces_path is None:
                        return TraceQueryResult()
                    params: dict[str, object] = {"limit": limit}
                    if name:
                        params["name"] = name
                    response = http.get(traces_path, params=params)
                    response.raise_for_status()
                    return _trace_rows(response.json(), limit=limit)

            return _Client()

        return _create_http_observability(
            observability_backend=observability_backend,
            client=client,
            client_factory=client_factory,
            config=config,
            provider=provider,
            open_fn=_open,
        )

    return factory


create_langsmith_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_LANGSMITH",
    provider="langsmith",
    default_url="https://api.smith.langchain.com",
    instant_path="/api/v1/sessions/count",
    traces_path="/api/v1/runs",
)
create_helicone_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_HELICONE",
    provider="helicone",
    default_url="https://api.helicone.ai",
    instant_path="/v1/request/query",
    traces_path="/v1/request/query",
)
create_posthog_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_POSTHOG",
    provider="posthog",
    default_url="https://app.posthog.com",
    instant_path="/api/projects/@current/insights/trend/",
)
create_braintrust_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_BRAINTRUST",
    provider="braintrust",
    default_url="https://api.braintrust.dev",
    instant_path="/v1/experiment/metrics",
    traces_path="/v1/project/logs",
)
create_signoz_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_SIGNOZ",
    provider="signoz",
    default_url="http://localhost:8080",
    instant_path="/api/v1/metrics",
    traces_path="/api/v1/traces",
)
create_honeycomb_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_HONEYCOMB",
    provider="honeycomb",
    default_url="https://api.honeycomb.io",
    instant_path="/1/metrics/",
)
create_arize_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_ARIZE",
    provider="arize",
    default_url="https://api.arize.com",
    instant_path="/v1/metrics",
    traces_path="/v1/traces",
)
create_phoenix_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_PHOENIX",
    provider="phoenix",
    default_url="http://localhost:6006",
    instant_path="/v1/metrics",
    traces_path="/v1/traces",
)
create_wandb_observability_backend = _http_obs_factory(
    env_prefix="INTERGRAX_WANDB",
    provider="wandb",
    default_url="https://api.wandb.ai",
    instant_path="/graphql",
)


def create_opensearch_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    from intergrax.integrations.providers.observability_backend.elasticsearch.config import ElasticsearchIntegrationConfig
    from intergrax.integrations.providers.observability_backend.elasticsearch.opens import open_elasticsearch_observability_backend

    os_config = OpenSearchIntegrationConfig.from_env(**config_overrides)
    es_config = ElasticsearchIntegrationConfig.model_validate(os_config.model_dump())
    if observability_backend is not None:
        return observability_backend
    if client is not None:
        from intergrax.integrations.providers.observability_backend.elasticsearch.adapter import ElasticsearchObservabilityBackend

        return ElasticsearchObservabilityBackend(client)
    if client_factory is not None:
        from intergrax.integrations.providers.observability_backend.elasticsearch.adapter import ElasticsearchObservabilityBackend

        return ElasticsearchObservabilityBackend(client_factory())
    return open_elasticsearch_observability_backend(es_config)


def create_vespa_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> VectorStore:
    if vector_store is not None:
        return vector_store
    config = VectorIntegrationConfig.from_env("INTERGRAX_VESPA", **config_overrides)

    def _open() -> Any:
        return _open_httpx_client(config, default_url=config.require_url())

    def _adapter(raw: Any) -> VectorStore:
        from intergrax.integrations._shared.p4.vector_adapters import VespaVectorFacade

        return RestVectorStoreIntegration(config, VespaVectorFacade(raw, collection=config.collection, tenant_id=config.tenant_id))

    return _resolve(
        implementation=None,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=_adapter,
    )


def create_gitlab_issue_tracker(
    *,
    issue_tracker: Optional[IssueTracker] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> IssueTracker:
    config = HttpIntegrationConfig.from_env("INTERGRAX_GITLAB", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "https://gitlab.com/api/v4")
        project = config.repo or config.org

        class _Client:
            def get_issue(self, issue_key: str) -> dict[str, Any]:
                iid = issue_key.split("#")[-1] if "#" in issue_key else issue_key
                response = http.get(f"/projects/{project}/issues/{iid}")
                response.raise_for_status()
                row = response.json()
                return {
                    "key": str(row.get("iid") or issue_key),
                    "summary": str(row.get("title") or ""),
                    "description": str(row.get("description") or ""),
                    "status": str(row.get("state") or ""),
                    "url": str(row.get("web_url") or ""),
                }

            def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
                iid = issue_key.split("#")[-1] if "#" in issue_key else issue_key
                response = http.post(f"/projects/{project}/issues/{iid}/notes", json={"body": body})
                response.raise_for_status()
                return {"id": str(response.json().get("id") or ""), "author": "bot"}

            def search_issues(self, jql: str, *, limit: int) -> list[dict[str, Any]]:
                response = http.get(f"/projects/{project}/issues", params={"search": jql, "per_page": limit})
                response.raise_for_status()
                return [
                    {
                        "key": str(row.get("iid") or ""),
                        "summary": str(row.get("title") or ""),
                        "status": str(row.get("state") or ""),
                    }
                    for row in response.json()
                ]

        return _Client()

    return _resolve(
        implementation=issue_tracker,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: RestIssueTracker(c, provider="gitlab"),
    )


def _escalation_channel_factory(*, env_prefix: str, provider: str, default_url: str) -> Callable[..., NotificationChannel]:
    def factory(
        *,
        notification_channel: Optional[NotificationChannel] = None,
        client: Optional[Any] = None,
        client_factory: Optional[Callable[[], Any]] = None,
        **config_overrides: object,
    ) -> NotificationChannel:
        if notification_channel is not None:
            return notification_channel
        config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

        def _sender(*, message: Any) -> None:
            http = _open_httpx_client(config, default_url=config.base_url or default_url)
            if provider == "pagerduty":
                payload = {
                    "routing_key": config.api_key or config.token,
                    "event_action": "trigger",
                    "payload": {
                        "summary": str(getattr(message, "subject", None) or message.task_id),
                        "severity": "error",
                        "source": "intergrax",
                        "custom_details": {"body": str(getattr(message, "body", ""))},
                    },
                }
                response = http.post("/v2/enqueue", json=payload)
            else:
                payload = {
                    "message": str(getattr(message, "body", "")),
                    "alias": str(getattr(message, "task_id", "intergrax")),
                    "description": str(getattr(message, "subject", "")),
                    "responders": [{"name": str(message.metadata.get("responder") or "ops"), "type": "team"}],
                }
                response = http.post("/v2/alerts", json=payload)
            response.raise_for_status()

        _ = client, client_factory
        return HttpNotificationChannel(_sender, provider=provider)

    return factory


create_pagerduty_notification_channel = _escalation_channel_factory(
    env_prefix="INTERGRAX_PAGERDUTY",
    provider="pagerduty",
    default_url="https://events.pagerduty.com",
)
create_opsgenie_notification_channel = _escalation_channel_factory(
    env_prefix="INTERGRAX_OPSGENIE",
    provider="opsgenie",
    default_url="https://api.opsgenie.com",
)


__all__ = [
    "create_arize_observability_backend",
    "create_braintrust_observability_backend",
    "create_gitlab_issue_tracker",
    "create_helicone_observability_backend",
    "create_honeycomb_observability_backend",
    "create_langsmith_observability_backend",
    "create_opensearch_observability_backend",
    "create_opsgenie_notification_channel",
    "create_pagerduty_notification_channel",
    "create_phoenix_observability_backend",
    "create_posthog_observability_backend",
    "create_signoz_observability_backend",
    "create_vespa_vector_store",
    "create_wandb_observability_backend",
]
