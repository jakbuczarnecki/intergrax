# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.6 P5 integration factories (8 greenfield harness slugs)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.health import http_ping_ok
from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations._shared.p2.factories import _open_httpx_client, _resolve
from intergrax.integrations._shared.p3.clients import HttpNotificationChannel
from intergrax.integrations._shared.p5.clients import HttpCiCdBackend, HttpObservabilityClientAdapter, KubernetesCloudPlatform
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend, TraceQueryResult


def _create_ci_cd_backend(
    *,
    env_prefix: str,
    provider: str,
    default_url: str,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    health_path: str = "/",
    **config_overrides: object,
) -> CiCdBackend:
    config = HttpIntegrationConfig.from_env(env_prefix, **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or default_url)
        org = config.org
        repo = config.repo
        project = config.org or org

        class _Client:
            def get_workflow_run(self, run_id: str) -> dict[str, Any]:
                if provider == "gitlab_ci":
                    response = http.get(f"/api/v4/projects/{project}/pipelines/{run_id}")
                elif provider == "circleci":
                    response = http.get(f"/api/v2/project/{project}/{repo}/pipeline/{run_id}")
                elif provider == "azure_pipelines":
                    response = http.get(
                        f"/{org}/{repo}/_apis/pipelines/runs/{run_id}",
                        params={"api-version": "7.0"},
                    )
                elif provider == "codecov":
                    response = http.get(f"/api/v2/github/{org}/{repo}/commits/{run_id}")
                else:
                    raise IntegrationConfigurationError(f"Unsupported CI provider: {provider}")
                response.raise_for_status()
                payload = response.json()
                return dict(payload) if isinstance(payload, dict) else {"id": run_id}

            def list_check_suites(self, *, ref: str, limit: int = 20) -> list[dict[str, Any]]:
                if provider == "gitlab_ci":
                    response = http.get(
                        f"/api/v4/projects/{project}/pipelines",
                        params={"ref": ref, "per_page": limit},
                    )
                elif provider == "circleci":
                    response = http.get(
                        f"/api/v2/project/{project}/{repo}/pipeline",
                        params={"branch": ref, "page-token": limit},
                    )
                elif provider == "azure_pipelines":
                    response = http.get(
                        f"/{org}/{repo}/_apis/pipelines",
                        params={"api-version": "7.0", "$top": limit},
                    )
                elif provider == "codecov":
                    response = http.get(
                        f"/api/v2/github/{org}/{repo}/commits",
                        params={"branch": ref, "page_size": limit},
                    )
                else:
                    return []
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload[:limit]
                if isinstance(payload, dict):
                    for key in ("items", "results", "commits", "pipelines", "value"):
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


def create_gitlab_ci_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    return _create_ci_cd_backend(
        env_prefix="INTERGRAX_GITLAB_CI",
        provider="gitlab_ci",
        default_url="https://gitlab.com",
        health_path="/api/v4/version",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_circleci_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    return _create_ci_cd_backend(
        env_prefix="INTERGRAX_CIRCLECI",
        provider="circleci",
        default_url="https://circleci.com",
        health_path="/api/v2/me",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_azure_pipelines_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    return _create_ci_cd_backend(
        env_prefix="INTERGRAX_AZURE_PIPELINES",
        provider="azure_pipelines",
        default_url="https://dev.azure.com",
        health_path="/",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_codecov_ci_cd(
    *,
    ci_cd: Optional[CiCdBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CiCdBackend:
    return _create_ci_cd_backend(
        env_prefix="INTERGRAX_CODECOV",
        provider="codecov",
        default_url="https://api.codecov.io",
        health_path="/api/v2",
        ci_cd=ci_cd,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )


def create_mailpit_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    if notification_channel is not None:
        return notification_channel
    config = HttpIntegrationConfig.from_env("INTERGRAX_MAILPIT", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://localhost:8025")

        class _Client:
            def health(self) -> bool:
                return http_ping_ok(http, path="/api/v1/info")

        return _Client()

    resolved_client = client if client is not None else (client_factory() if client_factory else _open())

    def _sender(*, message: Any) -> None:
        del message

    return HttpNotificationChannel(_sender, provider="mailpit", health_client=resolved_client)


def create_localstack_cloud_platform(
    *,
    cloud_platform: Optional[CloudPlatform] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> CloudPlatform:
    config = HttpIntegrationConfig.from_env("INTERGRAX_LOCALSTACK", **config_overrides)
    namespace = config.org or "default"

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://localhost:4566")

        class _Client:
            def health(self) -> bool:
                return http_ping_ok(http, path="/_localstack/health")

        return _Client()

    return _resolve(
        implementation=cloud_platform,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: KubernetesCloudPlatform(c, namespace=namespace, slug="localstack"),
    )


def create_grafana_oncall_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> NotificationChannel:
    if notification_channel is not None:
        return notification_channel
    config = HttpIntegrationConfig.from_env("INTERGRAX_GRAFANA_ONCALL", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://localhost:8080")

        class _Client:
            def health(self) -> bool:
                return http_ping_ok(http, path="/api/internal/v1/health/")

        return _Client()

    resolved_client = client if client is not None else (client_factory() if client_factory else _open())

    def _sender(*, message: Any) -> None:
        from intergrax.runtime.notifications.models import NotificationMessage

        if not isinstance(message, NotificationMessage):
            raise IntegrationConfigurationError("Grafana OnCall expects NotificationMessage")
        http = _open_httpx_client(config, default_url=config.base_url or "http://localhost:8080")
        http.post(
            "/api/v1/integrations/alertmanager/",
            json={
                "title": str(message.subject or message.task_id),
                "message": str(message.body or ""),
            },
        )

    return HttpNotificationChannel(_sender, provider="grafana_oncall", health_client=resolved_client)


def create_opentelemetry_collector_observability_backend(
    *,
    observability_backend: Optional[ObservabilityBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> ObservabilityBackend:
    config = HttpIntegrationConfig.from_env("INTERGRAX_OPENTELEMETRY_COLLECTOR", **config_overrides)

    def _open() -> Any:
        http = _open_httpx_client(config, default_url=config.base_url or "http://localhost:13133")

        class _Client:
            def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
                del promql, eval_time
                return 1.0 if http_ping_ok(http, path="/") else 0.0

            def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
                del promql, step
                value = 1.0 if http_ping_ok(http, path="/") else 0.0
                return [{"timestamp": start, "value": value}, {"timestamp": end, "value": value}]

            def health(self) -> bool:
                return http_ping_ok(http, path="/")

        return _Client()

    return _resolve(
        implementation=observability_backend,
        backend=client,
        backend_factory=client_factory,
        open_fn=_open,
        adapter_fn=lambda c: HttpObservabilityClientAdapter(c, provider="opentelemetry_collector"),
    )


__all__ = [
    "create_azure_pipelines_ci_cd",
    "create_circleci_ci_cd",
    "create_codecov_ci_cd",
    "create_gitlab_ci_ci_cd",
    "create_grafana_oncall_notification_channel",
    "create_localstack_cloud_platform",
    "create_mailpit_notification_channel",
    "create_opentelemetry_collector_observability_backend",
]
