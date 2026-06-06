# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.6 P4 integration providers (28 slugs)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_ci_cd_backend,
    assert_cloud_platform,
    assert_feature_flag_backend,
    assert_graph_store,
    assert_interaction_surface,
    assert_issue_tracker,
    assert_message_bus,
    assert_notification_channel,
    assert_object_storage,
    assert_observability_backend,
    assert_relational_store,
    assert_secrets_store,
    assert_vector_store,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.ci_cd import CheckSuiteRecord, WorkflowRunRecord
from intergrax.integrations.contracts.feature_flag import FeatureFlagEvaluation
from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.contracts.observability_backend import MetricQueryResult, MetricPoint, MetricSeries, TraceQueryResult, TraceRecord
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.notifications.models import NotificationMessage

pytestmark = pytest.mark.unit

M6_P4_SLUGS = (
    "pgvector",
    "duckdb",
    "influxdb",
    "timescaledb",
    "grafana",
    "loki",
    "tempo",
    "aws_secrets_manager",
    "azure_key_vault",
    "gcp_secret_manager",
    "doppler",
    "unleash",
    "launchdarkly",
    "github_actions",
    "redpanda",
    "cloudflare_r2",
    "memgraph",
    "falkordb",
    "incident_io",
    "kubernetes",
    "servicenow",
    "bitbucket",
    "asana",
    "sendgrid",
    "mailgun",
    "mlflow",
    "huggingface_hub",
    "ollama",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeObsClient:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={}, points=[MetricPoint(timestamp=1.0, value=3.0)])],
        )

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
        return [{"timestamp": start, "value": 3.0}]

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        return TraceQueryResult(traces=[TraceRecord(trace_id="tr-1", name=name or "span")])


class _FakeSqlConnection:
    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        del sql, params

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        del sql, params
        return [("ok",)]

    def commit(self) -> None:
        return None


class _FakeSecretsClient:
    def read_secret(self, path: str, *, version: Optional[str] = None) -> str:
        del version
        return f"secret:{path}"

    def write_secret(self, path: str, value: str) -> None:
        del path, value

    def delete_secret(self, path: str) -> None:
        del path


class _FakeCloudSecretsClient:
    def read_secret(self, mount: str, path: str, *, version: Optional[str] = None) -> str:
        del version
        return f"{mount}:{path}"

    def write_secret(self, mount: str, path: str, value: str) -> None:
        del mount, path, value

    def delete_secret(self, mount: str, path: str) -> None:
        del mount, path


class _FakeFlagClient:
    def evaluate_flag(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> dict[str, Any]:
        del tenant_id, user_id
        return {"enabled": flag_key == "adaptive.observe", "variant": "on"}


class _FakeCiClient:
    def get_workflow_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "completed", "conclusion": "success", "url": "https://github/run/1"}

    def list_check_suites(self, *, ref: str, limit: int = 20) -> list[dict[str, Any]]:
        del limit
        return [{"id": "1", "name": ref, "status": "completed", "conclusion": "success"}]


class _FakeGraphClient:
    def run(self, statement: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        del statement, parameters
        return [{"n": {"id": "1"}}]

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        return {"id": node_id, "labels": ["Entity"], "properties": {"name": "node"}}


class _FakeIssueClient:
    def get_issue(self, issue_key: str) -> Any:
        return {"key": issue_key, "summary": "Task", "status": "open"}

    def add_comment(self, issue_key: str, body: str) -> Any:
        return {"id": "c1", "body": body, "issue": issue_key}

    def search_issues(self, jql: str, *, limit: int) -> Any:
        return {"issues": [{"key": "1", "summary": jql, "status": "open"}], "total": 1}

    def create_issue(self, *, title: str, description: str = "", labels: Optional[list[str]] = None) -> Any:
        del description, labels
        return {"key": "99", "summary": title, "status": "open"}


class _FakeObjectClient:
    def put_object(self, *, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
        del bucket, key, body, content_type

    def get_object(self, *, bucket: str, key: str) -> bytes:
        del bucket
        return b"artifact"

    def delete_object(self, *, bucket: str, key: str) -> None:
        del bucket, key

    def generate_presigned_url(self, *, bucket: str, key: str, method: str, expires_in: int) -> str:
        del bucket, method, expires_in
        return f"https://storage/{key}"


class _FakeOllamaClient:
    def health(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["llama3"]


class _FakeMessageBus:
    def enqueue(self, request: Any) -> Any:
        del request
        from intergrax.integrations.contracts.message_bus import TaskHandle

        return TaskHandle(task_id="task-1", provider="redpanda", tenant_id="t1")

    def get_status(self, task_id: str) -> Any:
        from intergrax.integrations.contracts.message_bus import TaskStatus

        del task_id
        return TaskStatus.COMPLETED

    def get_result(self, task_id: str) -> Any:
        from intergrax.integrations.contracts.message_bus import TaskResult

        del task_id
        return TaskResult(task_id="task-1", output={"ok": True})


@pytest.mark.parametrize("slug", M6_P4_SLUGS)
def test_m6_p4_slugs_registered(slug: str) -> None:
    register_default_integrations()
    assert slug in catalog_snapshot()


def test_pgvector_vector_store() -> None:
    from intergrax.integrations.providers.vector_store.pgvector.bundle import create_pgvector_vector_store

    store = create_pgvector_vector_store()
    assert_vector_store(store)
    assert store.count() == 0


def test_duckdb_relational_store() -> None:
    from intergrax.integrations.providers.relational_store.duckdb.bundle import create_duckdb_relational_store

    store = create_duckdb_relational_store(connection=_FakeSqlConnection(), dsn=":memory:")
    assert_relational_store(store)
    store.connect()
    store.execute("SELECT 1")


def test_observability_stack() -> None:
    from intergrax.integrations.providers.observability_backend.grafana.bundle import create_grafana_observability_backend
    from intergrax.integrations.providers.observability_backend.influxdb.bundle import create_influxdb_observability_backend
    from intergrax.integrations.providers.observability_backend.loki.bundle import create_loki_observability_backend
    from intergrax.integrations.providers.observability_backend.mlflow.bundle import create_mlflow_observability_backend
    from intergrax.integrations.providers.observability_backend.tempo.bundle import create_tempo_observability_backend

    client = _FakeObsClient()
    for factory in (
        create_influxdb_observability_backend,
        create_grafana_observability_backend,
        create_loki_observability_backend,
        create_tempo_observability_backend,
        create_mlflow_observability_backend,
    ):
        backend = factory(observability_backend=client)  # type: ignore[arg-type]
        assert_observability_backend(backend)
        assert backend.query_instant("up").series[0].points[0].value == 3.0


def test_secrets_stores() -> None:
    from intergrax.integrations.providers.secrets_store.aws_secrets_manager.bundle import create_aws_secrets_manager_secrets_store
    from intergrax.integrations.providers.secrets_store.azure_key_vault.bundle import create_azure_key_vault_secrets_store
    from intergrax.integrations.providers.secrets_store.doppler.bundle import create_doppler_secrets_store
    from intergrax.integrations.providers.secrets_store.gcp_secret_manager.bundle import create_gcp_secret_manager_secrets_store

    aws = create_aws_secrets_manager_secrets_store(client=_FakeCloudSecretsClient())
    azure = create_azure_key_vault_secrets_store(client=_FakeCloudSecretsClient())
    gcp = create_gcp_secret_manager_secrets_store(client=_FakeCloudSecretsClient())
    doppler = create_doppler_secrets_store(client=_FakeSecretsClient())
    for store in (aws, azure, gcp, doppler):
        assert_secrets_store(store)
        assert store.get_secret("harness/token")


def test_feature_flags_and_ci_cd() -> None:
    from intergrax.integrations.providers.ci_cd.github_actions.bundle import create_github_actions_ci_cd
    from intergrax.integrations.providers.feature_flag.launchdarkly.bundle import create_launchdarkly_feature_flag
    from intergrax.integrations.providers.feature_flag.unleash.bundle import create_unleash_feature_flag

    unleash = create_unleash_feature_flag(client=_FakeFlagClient())
    ld = create_launchdarkly_feature_flag(client=_FakeFlagClient())
    for backend in (unleash, ld):
        assert_feature_flag_backend(backend)
        assert backend.is_enabled("adaptive.observe", tenant_id="t1")
        evaluation = backend.evaluate("adaptive.observe", tenant_id="t1")
        assert isinstance(evaluation, FeatureFlagEvaluation)

    ci = create_github_actions_ci_cd(client=_FakeCiClient())
    assert_ci_cd_backend(ci)
    run = ci.get_workflow_run("42")
    assert isinstance(run, WorkflowRunRecord)
    suites = ci.list_check_suites(ref="main")
    assert isinstance(suites[0], CheckSuiteRecord)


def test_redpanda_message_bus() -> None:
    from intergrax.integrations.contracts.message_bus import TaskRequest
    from intergrax.integrations.providers.message_bus.redpanda.bundle import create_redpanda_message_bus

    bus = create_redpanda_message_bus(message_bus=_FakeMessageBus())
    handle = bus.enqueue(
        TaskRequest(tenant_id="t1", run_id="run-1", task_name="echo", payload=b"{}"),
    )
    assert handle.task_id == "task-1"


def test_cloudflare_r2_object_storage() -> None:
    from intergrax.integrations.providers.object_storage.cloudflare_r2.bundle import create_cloudflare_r2_object_storage

    storage = create_cloudflare_r2_object_storage(client=_FakeObjectClient())
    assert_object_storage(storage)


def test_graph_stores() -> None:
    from intergrax.integrations.providers.graph_store.falkordb.bundle import create_falkordb_graph_store
    from intergrax.integrations.providers.graph_store.memgraph.bundle import create_memgraph_graph_store

    for factory in (create_memgraph_graph_store, create_falkordb_graph_store):
        store = factory(client=_FakeGraphClient())
        assert_graph_store(store)
        assert store.get_node("1") is not None


@pytest.mark.asyncio
async def test_incident_io_and_sendgrid() -> None:
    from intergrax.integrations._shared.p3.clients import HttpNotificationChannel
    from intergrax.integrations.providers.notification_channel.incident_io.bundle import create_incident_io_notification_channel
    from intergrax.integrations.providers.notification_channel.sendgrid.bundle import create_sendgrid_notification_channel

    sent: list[Any] = []

    def _sender(*, message: Any) -> None:
        sent.append(message)

    incident = create_incident_io_notification_channel(
        notification_channel=HttpNotificationChannel(_sender, provider="incident_io"),
    )
    sendgrid = create_sendgrid_notification_channel(sender=_sender)
    assert_notification_channel(incident)
    assert_notification_channel(sendgrid)
    await incident.notify(
        NotificationMessage(tenant_id="t1", channel="ops", task_id="task-1", subject="Alert", body="Harness")
    )
    assert len(sent) == 1


def test_kubernetes_cloud_platform() -> None:
    from intergrax.integrations.providers.cloud_platform.kubernetes.bundle import create_kubernetes_cloud_platform

    class _K8s:
        def health(self) -> bool:
            return True

    platform = create_kubernetes_cloud_platform(client=_K8s())
    assert_cloud_platform(platform)
    assert platform.resolve("observability_backend") == "prometheus"


def test_issue_trackers() -> None:
    from intergrax.integrations.providers.issue_tracker.asana.bundle import create_asana_issue_tracker
    from intergrax.integrations.providers.issue_tracker.bitbucket.bundle import create_bitbucket_issue_tracker
    from intergrax.integrations.providers.issue_tracker.servicenow.bundle import create_servicenow_issue_tracker

    for factory in (create_servicenow_issue_tracker, create_bitbucket_issue_tracker, create_asana_issue_tracker):
        tracker = factory(client=_FakeIssueClient())
        assert_issue_tracker(tracker)
        assert tracker.get_issue("PRJ-1").summary == "Task"


def test_mailgun_and_ollama_interaction_surfaces() -> None:
    from intergrax.integrations.providers.interaction_surface.mailgun.bundle import create_mailgun_interaction_surface
    from intergrax.integrations.providers.interaction_surface.ollama.bundle import create_ollama_interaction_surface

    mailgun = create_mailgun_interaction_surface()
    assert_interaction_surface(mailgun)
    payload = {"sender": "user@example.com", "body-plain": "hello"}
    assert mailgun.can_handle(payload)
    inbound = mailgun.to_inbound(payload, tenant_id="t1", user_id="user@example.com")
    assert inbound.message == "hello"

    ollama = create_ollama_interaction_surface(client=_FakeOllamaClient())
    assert_interaction_surface(ollama)
    assert ollama.to_inbound({"model": "llama3", "prompt": "hi"}, tenant_id="t1", user_id="u1").channel == "ollama"


def test_huggingface_hub_object_storage() -> None:
    from intergrax.integrations.providers.object_storage.huggingface_hub.bundle import create_huggingface_hub_object_storage

    storage = create_huggingface_hub_object_storage(client=_FakeObjectClient())
    assert_object_storage(storage)


def test_profile_resolve_new_categories() -> None:
    register_default_integrations()
    from intergrax.integrations.providers.ci_cd.github_actions.bundle import create_github_actions_ci_cd
    from intergrax.integrations.providers.feature_flag.unleash.bundle import create_unleash_feature_flag

    flags = create_unleash_feature_flag(client=_FakeFlagClient())
    ci = create_github_actions_ci_cd(client=_FakeCiClient())
    assert_feature_flag_backend(flags)
    assert_ci_cd_backend(ci)
    profile = IntegrationProfile(feature_flag="unleash", ci_cd="github_actions")
    assert profile.slug_for_category(IntegrationCategory.FEATURE_FLAG) == "unleash"
    assert profile.slug_for_category(IntegrationCategory.CI_CD) == "github_actions"
