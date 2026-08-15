# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.9 harness follow-up tests — full adapters, tools, slash_command."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.interactions.adapters.slash_command_adapter import SlashCommandInteractionAdapter
from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker
from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel
from intergrax.integrations.providers.observability_backend.braintrust.bundle import create_braintrust_observability_backend
from intergrax.integrations.providers.observability_backend.langsmith.bundle import create_langsmith_observability_backend
from intergrax.integrations.providers.observability_backend.opensearch.bundle import create_opensearch_observability_backend
from intergrax.integrations.providers.vector_store.vespa.adapter import _VespaVectorStore
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog, list_slugs
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.braintrust.service import braintrust_log_eval
from intergrax.tools.providers.braintrust.contracts import BraintrustLogEvalInput
from intergrax.tools.providers.gitlab.service import gitlab_create_issue
from intergrax.tools.providers.gitlab.contracts import GitLabCreateIssueInput
from intergrax.tools.providers.pagerduty.service import pagerduty_trigger_incident
from intergrax.tools.providers.pagerduty.contracts import PagerDutyTriggerIncidentInput
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeLangSmithClient:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> Any:
        from intergrax.integrations.contracts.observability_backend import MetricPoint, MetricQueryResult, MetricSeries

        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={}, points=[MetricPoint(timestamp=1.0, value=3.0)])],
        )

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> Any:
        return self.query_instant(promql, eval_time=start)

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> Any:
        from intergrax.integrations.contracts.observability_backend import TraceQueryResult, TraceRecord

        return TraceQueryResult(traces=[TraceRecord(trace_id="ls-1", name="agent-run")])


class _FakeGitlabClient:
    def get_issue(self, issue_key: str) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueRecord

        return IssueRecord(key=issue_key, summary="Bug", url="https://gitlab/1")

    def add_comment(self, issue_key: str, body: str) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueComment

        return IssueComment(id="1", body=body)

    def search_issues(self, jql: str, *, limit: int) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueSearchResult

        return IssueSearchResult()

    def create_issue(self, *, title: str, description: str = "", labels: Optional[list[str]] = None) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueRecord

        return IssueRecord(key="99", summary=title, description=description, url="https://gitlab/99")


class _FakePagerDutyClient:
    def trigger_incident(self, **kwargs: Any) -> str:
        return "dedup-abc"

    def acknowledge_incident(self, *, dedup_key: str, note: str | None = None) -> None:
        _ = dedup_key, note

    def send_notification(self, *, subject: str, body: str, task_id: str) -> None:
        _ = subject, body, task_id


class _FakeBraintrustClient:
    def log_eval(self, *, name: str, score: float, metadata: Any = None, project: Any = None) -> str:
        return "log-1"

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> Any:
        from intergrax.integrations.contracts.observability_backend import MetricPoint, MetricQueryResult, MetricSeries

        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={}, points=[MetricPoint(timestamp=1.0, value=1.0)])],
        )

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> Any:
        return self.query_instant(promql)

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> Any:
        from intergrax.integrations.contracts.observability_backend import TraceQueryResult

        return TraceQueryResult()


def test_full_langsmith_adapter() -> None:
    from intergrax.integrations.providers.observability_backend.langsmith.integration import (
        LangsmithObservabilityIntegration,
    )

    backend = LangsmithObservabilityIntegration.from_client(_FakeLangSmithClient())  # type: ignore[arg-type]
    assert backend.query_instant("sessions").series[0].points[0].value == 3.0
    assert backend.query_traces(limit=1).traces[0].trace_id == "ls-1"


def test_gitlab_create_issue_tool() -> None:
    class _Tracker:
        def create_issue(self, *, title: str, description: str = "", labels: Optional[list[str]] = None) -> Any:
            from intergrax.integrations.contracts.issue_tracker import IssueRecord

            return IssueRecord(key="99", summary=title, description=description, url="https://gitlab/99")

    ctx = ToolWiringContext(issue_tracker=_Tracker())
    out = gitlab_create_issue(ctx, GitLabCreateIssueInput(title="Harness failure", description="details"))
    assert out.issue.key == "99"


def test_pagerduty_trigger_incident_tool() -> None:
    from intergrax.integrations.providers.notification_channel.pagerduty.adapter import _PagerDutyNotificationChannel

    channel = _PagerDutyNotificationChannel(_FakePagerDutyClient())  # type: ignore[arg-type]
    ctx = ToolWiringContext(notification_channel=channel)
    out = pagerduty_trigger_incident(
        ctx,
        PagerDutyTriggerIncidentInput(summary="Agent run failed"),
    )
    assert out.dedup_key == "dedup-abc"


def test_braintrust_log_eval_tool() -> None:
    from intergrax.integrations.providers.observability_backend.braintrust.integration import (
        BraintrustObservabilityIntegration,
    )

    backend = BraintrustObservabilityIntegration.from_client(_FakeBraintrustClient())  # type: ignore[arg-type]
    ctx = ToolWiringContext(observability_backend=backend)
    out = braintrust_log_eval(ctx, BraintrustLogEvalInput(name="accuracy", score=0.9))
    assert out.log_id == "log-1"


def test_slash_command_not_registered_as_provider() -> None:
    register_default_integrations()
    assert "slash_command" not in list_slugs()


def test_slash_command_adapter_handles_payload() -> None:
    adapter = SlashCommandInteractionAdapter()
    assert adapter.can_handle({"command": "/research", "text": "hello"})
    inbound = adapter.to_inbound({"text": "/echo.basic ping"}, tenant_id="t1", user_id="u1")
    assert inbound.capability == "echo.basic"


def test_harness_lab_profile() -> None:
    profile = IntegrationProfile.harness_lab()
    assert profile.observability_backend is not None
    assert profile.observability_backend.resolved_slug() == "sentry"
    assert profile.notification_channel is not None
    assert profile.notification_channel.resolved_slug() == "pagerduty"


def test_opensearch_index_document() -> None:
    class _FakeHttp:
        def post(self, path: str, json: dict[str, Any]) -> Any:
            class _Resp:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, str]:
                    return {"_id": "doc-1"}

            return _Resp()

        def head(self, path: str) -> Any:
            class _Resp:
                status_code = 404

            return _Resp()

        def put(self, path: str, json: dict[str, Any]) -> Any:
            class _Resp:
                def raise_for_status(self) -> None:
                    return None

            return _Resp()

    from intergrax.integrations.providers.observability_backend.opensearch.client import OpenSearchRestClient
    from intergrax.integrations.providers.observability_backend.opensearch.config import OpenSearchIntegrationConfig

    client = OpenSearchRestClient(OpenSearchIntegrationConfig(), http_client=_FakeHttp())
    doc_id = client.index_document(index="logs", document={"message": "hello"})
    assert doc_id == "doc-1"


def test_vespa_feed_and_query() -> None:
    fed: list[str] = []

    class _FakeHttp:
        def post(self, path: str, json: dict[str, Any]) -> Any:
            fed.append(path)

            class _Resp:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, str]:
                    return {"id": "id:default:intergrax::doc-1"}

            return _Resp()

        def delete(self, path: str) -> Any:
            class _Resp:
                status_code = 200

                def raise_for_status(self) -> None:
                    return None

            return _Resp()

    from intergrax.integrations.providers.vector_store.vespa.client import VespaRestClient
    from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope

    client = VespaRestClient(VespaIntegrationConfig(), http_client=_FakeHttp())
    client.feed_document(doc_id="doc-1", fields={"content": "hello"})
    assert fed
    store = _VespaVectorStore(VespaIntegrationConfig(), client)
    assert store.count(scope=VectorStoreScope(tenant_id="default")) >= 0
