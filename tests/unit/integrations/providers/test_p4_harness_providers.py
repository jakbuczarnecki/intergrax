# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.8 harness integration providers."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_issue_tracker,
    assert_notification_channel,
    assert_observability_backend,
    assert_vector_store,
)
from intergrax.integrations.contracts.observability_backend import TraceQueryResult, TraceRecord
from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker
from intergrax.integrations.providers.notification_channel.opsgenie.bundle import create_opsgenie_notification_channel
from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel
from intergrax.integrations.providers.observability_backend.langsmith.bundle import create_langsmith_observability_backend
from intergrax.integrations.providers.observability_backend.opensearch.bundle import create_opensearch_observability_backend
from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.runtime.notifications.models import NotificationMessage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeObsClient:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> Any:
        from intergrax.integrations.contracts.observability_backend import MetricPoint, MetricQueryResult, MetricSeries

        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={}, points=[MetricPoint(timestamp=1.0, value=5.0)])],
        )

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
        return [{"timestamp": start, "value": 5.0}]

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        return TraceQueryResult(traces=[TraceRecord(trace_id="t1", name=name or "run", timestamp="2026-01-01")])


class _FakeGitlabClient:
    def get_issue(self, issue_key: str) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueRecord

        return IssueRecord(key=issue_key, summary="Bug", description="d", status="open", url="https://gitlab/1")

    def add_comment(self, issue_key: str, body: str) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueComment

        return IssueComment(id="c1", body=body, author="bot")

    def search_issues(self, jql: str, *, limit: int) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult

        return IssueSearchResult(
            issues=[IssueRecord(key="1", summary=jql, status="open")],
            total=1,
        )

    def create_issue(self, *, title: str, description: str = "", labels: Optional[list[str]] = None) -> Any:
        from intergrax.integrations.contracts.issue_tracker import IssueRecord

        return IssueRecord(key="99", summary=title, description=description, status="open", url="https://gitlab/99")


def test_langsmith_observability() -> None:
    from intergrax.integrations.providers.observability_backend.langsmith.integration import (
        LangsmithObservabilityIntegration,
    )

    backend = LangsmithObservabilityIntegration.from_client(_FakeObsClient())  # type: ignore[arg-type]
    assert_observability_backend(backend)
    assert backend.query_instant("runs").series[0].points[0].value == 5.0
    traces = backend.query_traces(limit=1)
    assert traces.traces[0].trace_id == "t1"


def test_gitlab_issue_tracker() -> None:
    from intergrax.integrations.providers.issue_tracker.gitlab.adapter import _GitLabIssueTracker

    tracker = _GitLabIssueTracker(_FakeGitlabClient())  # type: ignore[arg-type]
    assert_issue_tracker(tracker)
    assert tracker.get_issue("42").summary == "Bug"


@pytest.mark.asyncio
async def test_pagerduty_and_opsgenie() -> None:
    sent: list[Any] = []

    def _sender(*, message: Any) -> None:
        sent.append(message)

    from intergrax.integrations._shared.p3.clients import HttpNotificationChannel

    for provider in ("pagerduty", "opsgenie"):
        channel = HttpNotificationChannel(_sender, provider=provider)
        assert_notification_channel(channel)
        await channel.notify(
            NotificationMessage(
                tenant_id="t1",
                channel="ops",
                task_id="task-1",
                subject="Alert",
                body="Agent failed",
                metadata={"responder": "oncall"},
            )
        )
    assert len(sent) == 2


def test_vespa_vector_store() -> None:
    from unittest.mock import MagicMock

    from intergrax.integrations.providers.vector_store.vespa.client import VespaRestClient
    from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig

    mock_http = MagicMock()
    mock_http.post.return_value.raise_for_status = MagicMock()
    mock_http.post.return_value.json.return_value = {"id": "doc-1"}
    client = VespaRestClient(VespaIntegrationConfig(collection="coll", tenant_id="lab"), http_client=mock_http)
    store = create_vespa_vector_store(client=client)
    assert_vector_store(store)


def test_register_default_integrations_includes_p4_slugs() -> None:
    register_default_integrations()
    slugs = set(catalog_snapshot().keys())
    for slug in (
        "langsmith",
        "pagerduty",
        "gitlab",
        "vespa",
        "opensearch",
    ):
        assert slug in slugs
