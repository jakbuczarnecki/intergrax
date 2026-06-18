# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.8 harness gap integration factories."""

from __future__ import annotations
from intergrax.utils import attribute_access

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


def _lazy_bundle(module_path: str, factory_name: str) -> Callable[..., Any]:
    def factory(*args: Any, **kwargs: Any) -> Any:
        import importlib

        module = importlib.import_module(module_path)
        return attribute_access.optional(module, factory_name)(*args, **kwargs)

    return factory


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


create_langsmith_observability_backend = _lazy_bundle(
    "intergrax.integrations.providers.observability_backend.langsmith.bundle",
    "create_langsmith_observability_backend",
)
create_braintrust_observability_backend = _lazy_bundle(
    "intergrax.integrations.providers.observability_backend.braintrust.bundle",
    "create_braintrust_observability_backend",
)
create_opensearch_observability_backend = _lazy_bundle(
    "intergrax.integrations.providers.observability_backend.opensearch.bundle",
    "create_opensearch_observability_backend",
)
create_vespa_vector_store = _lazy_bundle(
    "intergrax.integrations.providers.vector_store.vespa.bundle",
    "create_vespa_vector_store",
)
create_gitlab_issue_tracker = _lazy_bundle(
    "intergrax.integrations.providers.issue_tracker.gitlab.bundle",
    "create_gitlab_issue_tracker",
)
create_pagerduty_notification_channel = _lazy_bundle(
    "intergrax.integrations.providers.notification_channel.pagerduty.bundle",
    "create_pagerduty_notification_channel",
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
                        "summary": str(attribute_access.optional(message, "subject", None) or message.task_id),
                        "severity": "error",
                        "source": "intergrax",
                        "custom_details": {"body": str(attribute_access.optional(message, "body", ""))},
                    },
                }
                response = http.post("/v2/enqueue", json=payload)
            else:
                payload = {
                    "message": str(attribute_access.optional(message, "body", "")),
                    "alias": str(attribute_access.optional(message, "task_id", "intergrax")),
                    "description": str(attribute_access.optional(message, "subject", "")),
                    "responders": [{"name": str(message.metadata.get("responder") or "ops"), "type": "team"}],
                }
                response = http.post("/v2/alerts", json=payload)
            response.raise_for_status()

        _ = client, client_factory
        return HttpNotificationChannel(_sender, provider=provider)

    return factory


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
