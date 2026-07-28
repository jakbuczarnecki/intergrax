# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira REST client — internal; HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JiraKnowledgeIssue,
    JiraKnowledgeIssuePage,
    parse_jira_knowledge_issue,
    parse_jira_knowledge_issue_page,
    validate_jira_issue_key,
    validate_jira_knowledge_issue_project_scope,
    validate_jira_project_key,
)

_KNOWLEDGE_ISSUE_FIELDS: tuple[str, ...] = (
    "summary",
    "description",
    "status",
    "assignee",
    "reporter",
    "issuetype",
    "project",
    "priority",
    "labels",
    "components",
    "created",
    "updated",
    "resolution",
)


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


def _response_status_code(response: object) -> int | None:
    status_code = response.status_code  # type: ignore[attr-defined]
    return int(status_code) if isinstance(status_code, int) else None


def _raise_for_knowledge_response(response: object, *, operation: str) -> None:
    status_code = _response_status_code(response)
    if status_code is None or status_code < 400:
        return
    if status_code == 429 or status_code >= 500:
        raise IntegrationDependencyError(f"Jira {operation} dependency failure")
    if status_code in {400, 401, 403}:
        raise IntegrationConfigurationError(f"Jira {operation} configuration failure")
    if operation == "get_knowledge_issue" and status_code == 404:
        raise IntegrationDependencyError("Jira issue fetch dependency failure")
    raise IntegrationConfigurationError(f"Jira {operation} configuration failure")


def _execute_knowledge_transport(operation: str, transport_fn: Any) -> object:
    try:
        return transport_fn()
    except (IntegrationConfigurationError, IntegrationDependencyError):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Jira knowledge dependency is unavailable"
        ) from None


def _decode_knowledge_json(response: object) -> dict[str, Any]:
    try:
        json_method = response.json  # type: ignore[attr-defined]
        payload = json_method()
    except Exception:
        raise ValueError("unexpected Jira knowledge response") from None
    if not isinstance(payload, dict):
        raise ValueError("unexpected Jira knowledge response")
    return payload


def _validate_knowledge_page_scope(
    page: JiraKnowledgeIssuePage,
    *,
    project_key: str,
) -> None:
    for issue in page.issues:
        validate_jira_knowledge_issue_project_scope(issue, project_key=project_key)


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

    def search_knowledge_issues(
        self,
        *,
        project_key: str,
        next_page_token: str | None,
        limit: int,
    ) -> JiraKnowledgeIssuePage:
        validated_project_key = validate_jira_project_key(project_key)
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be in range 1..1000")
        body: dict[str, object] = {
            "jql": f'project = "{validated_project_key}" ORDER BY id ASC',
            "maxResults": int(limit),
            "fields": list(_KNOWLEDGE_ISSUE_FIELDS),
        }
        if next_page_token is not None:
            if not isinstance(next_page_token, str):
                raise ValueError("next_page_token must be a string")
            token = next_page_token.strip()
            if not token:
                raise ValueError("next_page_token must be a non-empty string")
            body["nextPageToken"] = token
        response = _execute_knowledge_transport(
            "search_knowledge_issues",
            lambda: self._http_client.post("/search/jql", json=body),
        )
        _raise_for_knowledge_response(response, operation="search_knowledge_issues")
        payload = _decode_knowledge_json(response)
        try:
            page = parse_jira_knowledge_issue_page(
                payload,
                issue_url_builder=self._config.issue_url,
                plain_description=_plain_description,
            )
        except (ValueError, TypeError, ValidationError):
            raise ValueError("unexpected Jira knowledge response") from None
        _validate_knowledge_page_scope(page, project_key=validated_project_key)
        return page

    def get_knowledge_issue(
        self,
        *,
        issue_key: str,
    ) -> JiraKnowledgeIssue:
        validated_issue_key = validate_jira_issue_key(issue_key)
        fields_param = ",".join(_KNOWLEDGE_ISSUE_FIELDS)
        response = _execute_knowledge_transport(
            "get_knowledge_issue",
            lambda: self._http_client.get(
                f"/issue/{validated_issue_key}",
                params={"fields": fields_param},
            ),
        )
        _raise_for_knowledge_response(response, operation="get_knowledge_issue")
        payload = _decode_knowledge_json(response)
        try:
            issue = parse_jira_knowledge_issue(
                payload,
                issue_url=self._config.issue_url(validated_issue_key),
                plain_description=_plain_description,
            )
        except (ValueError, TypeError, ValidationError):
            raise ValueError("unexpected Jira knowledge response") from None
        return issue

    def update_issue(
        self,
        issue_key: str,
        *,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> IssueRecord:
        fields: dict[str, object] = {}
        if summary is not None:
            fields["summary"] = summary
        if assignee is not None:
            fields["assignee"] = {"name": assignee}
        if fields:
            response = self._http_client.put(f"/issue/{issue_key}", json={"fields": fields})
            response.raise_for_status()
        if status is not None:
            transitions = self._http_client.get(f"/issue/{issue_key}/transitions")
            transitions.raise_for_status()
            payload = transitions.json()
            transition_id = None
            if isinstance(payload, dict):
                for item in payload.get("transitions") or []:
                    if not isinstance(item, dict):
                        continue
                    to_obj = item.get("to")
                    to_name = to_obj.get("name") if isinstance(to_obj, dict) else None
                    if str(to_name or "").lower() == status.strip().lower():
                        transition_id = item.get("id")
                        break
            if transition_id is not None:
                self._http_client.post(
                    f"/issue/{issue_key}/transitions",
                    json={"transition": {"id": transition_id}},
                )
        return self.get_issue(issue_key)
