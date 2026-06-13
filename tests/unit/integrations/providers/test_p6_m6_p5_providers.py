# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.6 P5 integration providers and harness presets."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_ci_cd_backend,
    assert_cloud_platform,
    assert_issue_tracker,
    assert_notification_channel,
    assert_observability_backend,
)
from intergrax.integrations.contracts.base import HealthStatus, IntegrationCategory, IntegrationStatus
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.integrations.registry.harness_lab_health import health_check_harness_m6_p5_probes
from intergrax.integrations.registry.harness_lab_stack import HARNESS_M6_P5_PROBE_SLUGS
from intergrax.integrations.registry import presets
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

M6_P5_GREENFIELD_SLUGS = (
    "gitlab_ci",
    "circleci",
    "azure_pipelines",
    "mailpit",
    "localstack",
    "codecov",
    "grafana_oncall",
    "opentelemetry_collector",
)

M6_P5_HARDEN_SLUGS = (
    "prometheus",
    "clickhouse",
    "vault",
    "pagerduty",
    "github",
    "langfuse",
    "phoenix",
    "braintrust",
    "mlflow",
    "influxdb",
    "timescaledb",
    "temporal",
    "redpanda",
    "minio",
    "s3",
    "neo4j",
    "mongodb",
    "elasticsearch",
    "nats",
    "chroma",
    "weaviate",
    "launchdarkly",
    "signoz",
    "snowflake",
    "supabase",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeCiClient:
    def get_workflow_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "completed", "conclusion": "success", "url": "https://ci/run/1"}

    def list_check_suites(self, *, ref: str, limit: int = 20) -> list[dict[str, Any]]:
        del limit
        return [{"id": "1", "name": ref, "status": "completed", "conclusion": "success"}]

    def health(self) -> bool:
        return True


class _FakeHealthClient:
    def health(self) -> bool:
        return True


class _FakeObsClient:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
        del promql, eval_time
        return 1.0

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
        del promql, step
        return [{"timestamp": start, "value": 1.0}, {"timestamp": end, "value": 1.0}]

    def health(self) -> bool:
        return True


class _FakeIssueClient:
    def get_issue(self, issue_key: str) -> dict[str, Any]:
        return {"key": issue_key, "summary": "Task", "status": "open"}

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        return {"id": "c1", "body": body, "issue": issue_key}

    def search_issues(self, jql: str, *, limit: int) -> list[dict[str, Any]]:
        del limit
        return [{"key": "1", "summary": jql, "status": "open"}]

    def health(self) -> bool:
        return True


class _FakeVaultClient:
    def read_secret(self, mount: str, path: str, *, version: Optional[str] = None) -> str:
        del mount, version
        return f"secret:{path}"

    def write_secret(self, mount: str, path: str, value: str) -> None:
        del mount, path, value

    def delete_secret(self, mount: str, path: str) -> None:
        del mount, path

    def health(self) -> bool:
        return True


@pytest.mark.parametrize("slug", M6_P5_GREENFIELD_SLUGS)
def test_m6_p5_greenfield_registered(slug: str) -> None:
    register_default_integrations()
    meta = catalog_snapshot()[slug]
    assert meta.status is IntegrationStatus.STABLE


@pytest.mark.parametrize("slug", M6_P5_HARDEN_SLUGS)
def test_m6_p5_harden_promoted_stable(slug: str) -> None:
    register_default_integrations()
    meta = catalog_snapshot()[slug]
    assert meta.status is IntegrationStatus.STABLE


def test_p6_ci_cd_providers() -> None:
    from intergrax.integrations.providers.ci_cd.azure_pipelines.bundle import create_azure_pipelines_ci_cd
    from intergrax.integrations.providers.ci_cd.circleci.bundle import create_circleci_ci_cd
    from intergrax.integrations.providers.ci_cd.codecov.bundle import create_codecov_ci_cd
    from intergrax.integrations.providers.ci_cd.gitlab_ci.bundle import create_gitlab_ci_ci_cd

    for factory in (
        create_gitlab_ci_ci_cd,
        create_circleci_ci_cd,
        create_azure_pipelines_ci_cd,
        create_codecov_ci_cd,
    ):
        backend = factory(client=_FakeCiClient())
        assert_ci_cd_backend(backend)
        health = backend.health()
        assert isinstance(health, HealthStatus)
        assert health.healthy is True


def test_p6_mailpit_and_grafana_oncall() -> None:
    from intergrax.integrations.providers.notification_channel.grafana_oncall.bundle import (
        create_grafana_oncall_notification_channel,
    )
    from intergrax.integrations.providers.notification_channel.mailpit.bundle import create_mailpit_notification_channel

    mailpit = create_mailpit_notification_channel(client=_FakeHealthClient())
    assert_notification_channel(mailpit)
    assert mailpit.health().slug == "mailpit"

    oncall = create_grafana_oncall_notification_channel(client=_FakeHealthClient())
    assert_notification_channel(oncall)
    assert oncall.health().slug == "grafana_oncall"


def test_p6_localstack_cloud_platform() -> None:
    from intergrax.integrations.providers.cloud_platform.localstack.bundle import create_localstack_cloud_platform

    platform = create_localstack_cloud_platform(client=_FakeHealthClient())
    assert_cloud_platform(platform)
    assert platform.health().slug == "localstack"


def test_p6_opentelemetry_collector_observability() -> None:
    from intergrax.integrations.providers.observability_backend.opentelemetry_collector.bundle import (
        create_opentelemetry_collector_observability_backend,
    )

    backend = create_opentelemetry_collector_observability_backend(client=_FakeObsClient())
    assert_observability_backend(backend)
    assert backend.health().slug == "opentelemetry_collector"


def test_p5_harden_health_probes() -> None:
    from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker
    from intergrax.integrations.providers.observability_backend.prometheus.adapter import PrometheusObservabilityBackend
    from intergrax.integrations.providers.observability_backend.prometheus.client import PrometheusRestClient
    from intergrax.integrations.providers.observability_backend.prometheus.config import PrometheusIntegrationConfig
    from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store

    class _Http:
        def get(self, path: str, **kwargs: object) -> object:
            del kwargs, path

            class _Resp:
                status_code = 200

                def json(self) -> dict[str, object]:
                    return {"status": "success", "data": {"resultType": "vector", "result": []}}

            return _Resp()

    config = PrometheusIntegrationConfig(base_url="http://localhost:9090")
    client = PrometheusRestClient(config, http_client=_Http())
    prom_backend = PrometheusObservabilityBackend(client)
    assert prom_backend.health().healthy is True

    github = create_github_issue_tracker(client=_FakeIssueClient())
    assert_issue_tracker(github)
    assert github.health().slug == "github"

    vault = create_vault_secrets_store(client=_FakeVaultClient())
    assert vault.health() is not False


def test_harness_p5_presets_bind_categories() -> None:
    metrics = presets.harness_metrics_stack()
    assert metrics.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND.value) == "prometheus"

    eval_stack = presets.harness_eval_stack()
    assert eval_stack.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND.value) == "langfuse"
    assert eval_stack.slug_for_category(IntegrationCategory.RELATIONAL_STORE.value) == "duckdb"

    async_stack = presets.harness_async_stack()
    assert async_stack.slug_for_category(IntegrationCategory.MESSAGE_BUS.value) in {"redpanda", "kafka"}

    ci_stack = presets.harness_ci_stack()
    assert ci_stack.slug_for_category(IntegrationCategory.CI_CD.value) == "github_actions"
    assert ci_stack.slug_for_category(IntegrationCategory.ISSUE_TRACKER.value) == "github"


def test_harness_m6_p5_probe_slugs_subset_of_catalog() -> None:
    register_default_integrations()
    catalog = catalog_snapshot()
    for slug in HARNESS_M6_P5_PROBE_SLUGS:
        assert slug in catalog


def test_harness_m6_p5_health_with_injected_clients() -> None:
    register_default_integrations()
    results = health_check_harness_m6_p5_probes()
    assert len(results) == len(HARNESS_M6_P5_PROBE_SLUGS)
    assert all(isinstance(item, HealthStatus) for item in results)


def test_integration_profile_harness_metrics_preset_roundtrip() -> None:
    register_default_integrations()
    profile = presets.harness_metrics_stack()
    restored = IntegrationProfile.model_validate(profile.model_dump(mode="json"))
    assert restored.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND.value) == "prometheus"
