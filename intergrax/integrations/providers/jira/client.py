# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira REST client — internal; HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.providers.jira.config import JiraIntegrationConfig


def _extract_adf_text(node: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    if node.get("type") == "text":
        text = node.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    content = node.get("content")
    if isinstance(content, list):
        for child in content:
            if isinstance(child, dict):
                chunks.append(_extract_adf_text(child))
    return " ".join(part for part in chunks if part)


def _plain_description(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return _extract_adf_text(raw)
    return str(raw)


def _issue_from_payload(config: JiraIntegrationConfig, payload: Mapping[str, Any]) -> IssueRecord:
    key = str(payload.get("key") or "")
    fields = payload.get("fields")
    fields_map = fields if isinstance(fields, dict) else {}
    status_obj = fields_map.get("status")
    status = status_obj.get("name") if isinstance(status_obj, dict) else ""
    assignee_obj = fields_map.get("assignee")
    assignee = assignee_obj.get("displayName") if isinstance(assignee_obj, dict) else None
    return IssueRecord(
        key=key,
        summary=str(fields_map.get("summary") or ""),
        description=_plain_description(fields_map.get("description")),
        status=str(status or ""),
        assignee=str(assignee) if assignee else None,
        url=config.issue_url(key) if key else "",
    )


class JiraRestClient:
    """Minimal Jira REST API v3 client — sync HTTP via injected client."""

    def __init__(
        self,
        config: JiraIntegrationConfig,
        *,
        http_client: Any,
    ) -> None:
        if not config.base_url:
            raise IntegrationConfigurationError("Jira base_url is required (INTERGRAX_JIRA_BASE_URL)")
        if not config.email or not config.api_token:
            raise IntegrationConfigurationError(
                "Jira email and api_token are required (INTERGRAX_JIRA_EMAIL, INTERGRAX_JIRA_API_TOKEN)"
            )
        self._config = config
        self._http_client = http_client

    @property
    def config(self) -> JiraIntegrationConfig:
        return self._config

    def get_issue(self, issue_key: str) -> IssueRecord:
        response = self._http_client.get(f"/issue/{issue_key}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Jira get_issue response")
        return _issue_from_payload(self._config, payload)

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": body}],
                    }
                ],
            }
        }
        response = self._http_client.post(f"/issue/{issue_key}/comment", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise IntegrationConfigurationError("Unexpected Jira add_comment response")
        author_obj = data.get("author")
        author = author_obj.get("displayName") if isinstance(author_obj, dict) else None
        return IssueComment(
            id=str(data.get("id") or ""),
            body=_plain_description(data.get("body")),
            author=str(author) if author else None,
        )

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        response = self._http_client.post(
            "/search",
            json={
                "jql": jql,
                "maxResults": max(1, int(limit)),
                "fields": ["summary", "description", "status", "assignee"],
            },
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise IntegrationConfigurationError("Unexpected Jira search response")
        raw_issues = data.get("issues")
        issues = [
            _issue_from_payload(self._config, item)
            for item in raw_issues
            if isinstance(item, dict)
        ]
        total_raw = data.get("total", len(issues))
        total = int(total_raw) if isinstance(total_raw, int) else len(issues)
        return IssueSearchResult(issues=issues, total=total)
