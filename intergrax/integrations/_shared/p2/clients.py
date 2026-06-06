# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""P2/P3 catalog adapters over duck-typed backends."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional, Sequence

from intergrax.integrations._shared.rest_search import hits_from_brave_payload, hits_from_serpapi_payload
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent
from intergrax.integrations.contracts.collaboration_suite import (
    CalendarEventsResult,
    MailListResult,
    MailMessage,
    UserRecord,
)
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.contracts.observability_backend import MetricPoint, MetricQueryResult, MetricSeries, TraceQueryResult
from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.websearch.schemas.search_hit import SearchHit


class DynamoDBDocumentStore:
    def __init__(self, table: Any, *, partition_attr: str, sort_attr: str) -> None:
        self._table = table
        self._partition_attr = partition_attr
        self._sort_attr = sort_attr
        self._closed = False

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        self._require_open()
        item = self._table.get_item(partition_key, row_key)
        if item is None:
            return None
        return DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=dict(item.get("data") or item),
        )

    def put(self, document: DocumentRecord) -> None:
        self._require_open()
        self._table.put_item(
            {
                self._partition_attr: document.partition_key,
                self._sort_attr: document.row_key,
                "data": dict(document.data),
            }
        )

    def delete(self, partition_key: str, row_key: str) -> None:
        self._require_open()
        self._table.delete_item(partition_key, row_key)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        self._require_open()
        rows = self._table.query(partition_key, limit=limit, row_key_prefix=row_key_prefix)
        documents = [
            DocumentRecord(
                partition_key=partition_key,
                row_key=str(row.get(self._sort_attr) or row.get("row_key") or ""),
                data=dict(row.get("data") or row),
            )
            for row in rows
        ]
        return DocumentQueryResult(documents=documents, total=len(documents))

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("DynamoDB document store is closed")


class MemcachedKeyValueCache:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._closed = False

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        self._require_open()
        raw = self._client.get(self._scoped(tenant_id, key))
        return raw if isinstance(raw, bytes) else (str(raw).encode() if raw is not None else None)

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._require_open()
        self._client.set(self._scoped(tenant_id, key), value, ttl_seconds=ttl_seconds)

    def delete(self, tenant_id: str, key: str) -> None:
        self._require_open()
        self._client.delete(self._scoped(tenant_id, key))

    def set_if_absent(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        self._require_open()
        return bool(self._client.set_if_absent(self._scoped(tenant_id, key), value, ttl_seconds=ttl_seconds))

    def close(self) -> None:
        self._closed = True

    @staticmethod
    def _scoped(tenant_id: str, key: str) -> str:
        return f"{tenant_id}:{key}"

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Memcached cache is closed")


class SqlRelationalStore:
    def __init__(self, connection: Any, *, factory_name: str) -> None:
        self._connection = connection
        self._factory_name = factory_name

    def connect(self) -> None:
        if self._connection is None:
            raise IntegrationConfigurationError(
                f"SQL store is closed; create a new store via {self._factory_name}()"
            )

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_connection()
        conn.execute(sql, params)
        if hasattr(conn, "commit"):
            conn.commit()

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        conn = self._require_connection()
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        if self._connection is not None:
            if hasattr(self._connection, "close"):
                self._connection.close()
            self._connection = None

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise IntegrationConfigurationError(
                f"SQL store is closed; create a new store via {self._factory_name}()"
            )
        return self._connection

    def health(self) -> HealthStatus:
        if self._connection is None:
            return HealthStatus(slug=self._factory_name, healthy=False, detail="connection closed")
        try:
            self._connection.execute("SELECT 1")
            return HealthStatus(slug=self._factory_name, healthy=True, detail="sql ping")
        except Exception as exc:  # noqa: BLE001 — health probe surface
            return HealthStatus(slug=self._factory_name, healthy=False, detail=str(exc))


class SmtpNotificationChannel:
    def __init__(self, sender: Callable[..., None], *, from_address: str) -> None:
        self._sender = sender
        self._from_address = from_address

    async def notify(self, message: Any) -> None:
        from intergrax.runtime.notifications.models import NotificationMessage

        if not isinstance(message, NotificationMessage):
            raise IntegrationConfigurationError("SMTP adapter expects NotificationMessage")
        recipient = str(message.metadata.get("to") or message.channel)
        self._sender(
            from_address=self._from_address,
            to=recipient,
            subject=message.subject or f"[{message.task_id}] notification",
            body=message.body,
            metadata=dict(message.metadata),
        )


class OtelObservabilityBackend:
    def __init__(self, exporter: Any) -> None:
        self._exporter = exporter

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        value = float(self._exporter.query_instant(promql, eval_time=eval_time))
        ts = float(eval_time or 0.0)
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={}, points=[MetricPoint(timestamp=ts, value=value)])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        points_raw = self._exporter.query_range(promql, start=start, end=end, step=step)
        points = [
            MetricPoint(timestamp=float(row["timestamp"]), value=float(row["value"]))
            for row in points_raw
            if isinstance(row, dict)
        ]
        return MetricQueryResult(result_type="matrix", series=[MetricSeries(metric={}, points=points)])


    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        _ = limit, name
        from intergrax.integrations.contracts.observability_backend import TraceQueryResult

        return TraceQueryResult()


class RestIssueTracker:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def get_issue(self, issue_key: str) -> IssueRecord:
        payload = self._client.get_issue(issue_key)
        return IssueRecord(
            key=str(payload.get("key") or issue_key),
            summary=str(payload.get("summary") or payload.get("title") or ""),
            description=str(payload.get("description") or payload.get("body") or ""),
            status=str(payload.get("status") or ""),
            assignee=payload.get("assignee"),
            url=str(payload.get("url") or ""),
        )

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        payload = self._client.add_comment(issue_key, body)
        return IssueComment(
            id=str(payload.get("id") or uuid.uuid4()),
            body=body,
            author=payload.get("author"),
        )

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        rows = self._client.search_issues(jql, limit=limit)
        issues = [
            IssueRecord(
                key=str(row.get("key") or row.get("number") or ""),
                summary=str(row.get("summary") or row.get("title") or ""),
                description=str(row.get("description") or ""),
                status=str(row.get("status") or row.get("state") or ""),
                assignee=row.get("assignee"),
                url=str(row.get("url") or row.get("html_url") or ""),
            )
            for row in rows
        ]
        return IssueSearchResult(issues=issues, total=len(issues))

    def health(self) -> HealthStatus:
        from intergrax.integrations._shared.health import probe_client_health

        return probe_client_health(self._client, slug=self._provider)


class RestWikiKnowledge:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get_page(self, page_id: str) -> WikiPageRecord:
        payload = self._client.get_page(page_id)
        return WikiPageRecord(
            id=str(payload.get("id") or page_id),
            title=str(payload.get("title") or ""),
            space_key=str(payload.get("space_key") or payload.get("space") or ""),
            body=str(payload.get("body") or payload.get("content") or ""),
            url=str(payload.get("url") or ""),
            version=payload.get("version"),
        )

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        rows = self._client.search_pages(query, limit=limit)
        pages = [
            WikiPageRecord(
                id=str(row.get("id") or ""),
                title=str(row.get("title") or ""),
                space_key=str(row.get("space_key") or row.get("space") or ""),
                body=str(row.get("body") or row.get("snippet") or ""),
                url=str(row.get("url") or ""),
            )
            for row in rows
        ]
        return WikiSearchResult(pages=pages, total=len(pages))


class GoogleWorkspaceCollaborationSuite:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get_message(self, user_id: str, message_id: str) -> MailMessage:
        payload = self._client.get_message(user_id, message_id)
        return MailMessage(
            id=str(payload.get("id") or message_id),
            subject=str(payload.get("subject") or ""),
            body_preview=str(payload.get("body_preview") or payload.get("snippet") or ""),
            from_address=payload.get("from"),
            received_at=payload.get("received_at"),
        )

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25) -> MailListResult:
        rows = self._client.list_messages(user_id, folder=folder, limit=limit)
        messages = [
            MailMessage(
                id=str(row.get("id") or ""),
                subject=str(row.get("subject") or ""),
                body_preview=str(row.get("body_preview") or row.get("snippet") or ""),
                from_address=row.get("from"),
                received_at=row.get("received_at"),
            )
            for row in rows
        ]
        return MailListResult(messages=messages, total=len(messages))

    def send_mail(self, user_id: str, *, subject: str, body: str, to: Sequence[str]) -> None:
        self._client.send_mail(user_id, subject=subject, body=body, to=list(to))

    def list_calendar_events(
        self,
        user_id: str,
        *,
        start: str,
        end: str,
        limit: int = 50,
    ) -> CalendarEventsResult:
        from intergrax.integrations.contracts.collaboration_suite import CalendarEvent

        rows = self._client.list_calendar_events(user_id, start=start, end=end, limit=limit)
        events = [
            CalendarEvent(
                id=str(row.get("id") or ""),
                subject=str(row.get("subject") or row.get("summary") or ""),
                start=str(row.get("start") or ""),
                end=str(row.get("end") or ""),
                location=str(row.get("location") or ""),
                organizer=row.get("organizer"),
            )
            for row in rows
        ]
        return CalendarEventsResult(events=events, total=len(events))

    def get_user(self, user_id: str) -> UserRecord:
        payload = self._client.get_user(user_id)
        return UserRecord(
            id=str(payload.get("id") or user_id),
            display_name=str(payload.get("display_name") or payload.get("name") or ""),
            email=payload.get("email"),
        )


class RestSearchProvider:
    def __init__(
        self,
        *,
        provider: str,
        search_fn: Callable[[str, int], Mapping[str, Any]],
        hits_fn: Callable[[str, Mapping[str, Any], int], Sequence[SearchHit]],
    ) -> None:
        self._provider = provider
        self._search_fn = search_fn
        self._hits_fn = hits_fn

    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        payload = self._search_fn(query, limit)
        return self._hits_fn(query, payload, limit)


class PlaywrightBrowserAutomation:
    def __init__(self, browser: Any, *, timeout_ms: int) -> None:
        self._browser = browser
        self._timeout_ms = timeout_ms
        self._closed = False

    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        self._require_open()
        page = self._browser.new_page()
        try:
            response = page.goto(url, wait_until=wait_until, timeout=self._timeout_ms)
            status = int(getattr(response, "status", 200) or 200)
            title = str(page.title() or "")
            html = str(page.content() or "")
            text = str(page.inner_text("body") if hasattr(page, "inner_text") else "")
            return PageContent(url=url, title=title, text=text, html=html, status_code=status)
        finally:
            page.close()

    def close(self) -> None:
        if not self._closed:
            if hasattr(self._browser, "close"):
                self._browser.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Playwright browser is closed")


def brave_hits(query: str, payload: Mapping[str, Any], limit: int) -> Sequence[SearchHit]:
    return hits_from_brave_payload(query, payload, limit=limit)


def serpapi_hits(query: str, payload: Mapping[str, Any], limit: int) -> Sequence[SearchHit]:
    return hits_from_serpapi_payload(query, payload, limit=limit)
