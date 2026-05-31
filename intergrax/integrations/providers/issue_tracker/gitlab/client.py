# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GitLab REST client — HTTP injected from ``opens.py`` only."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.issue_tracker import IssueComment, IssueRecord, IssueSearchResult
from intergrax.integrations.providers.issue_tracker.gitlab.config import GitLabIntegrationConfig


def _issue_from_payload(config: GitLabIntegrationConfig, payload: Mapping[str, Any]) -> IssueRecord:
    iid = str(payload.get("iid") or payload.get("key") or "")
    web_url = str(payload.get("web_url") or "")
    return IssueRecord(
        key=iid,
        summary=str(payload.get("title") or ""),
        description=str(payload.get("description") or ""),
        status=str(payload.get("state") or ""),
        assignee=_assignee_name(payload.get("assignee")),
        url=web_url or (config.issue_url(config.project_id, iid) if iid else ""),
    )


def _assignee_name(raw: object) -> Optional[str]:
    if isinstance(raw, dict):
        name = raw.get("name") or raw.get("username")
        return str(name) if name else None
    return None


class GitLabRestClient:
    """Minimal GitLab Issues API v4 client."""

    def __init__(self, config: GitLabIntegrationConfig, *, http_client: Any) -> None:
        if not config.token:
            raise IntegrationConfigurationError("GitLab token is required (INTERGRAX_GITLAB_TOKEN)")
        if not config.project_id:
            raise IntegrationConfigurationError("GitLab project_id is required (INTERGRAX_GITLAB_REPO)")
        self._config = config
        self._http = http_client

    @property
    def config(self) -> GitLabIntegrationConfig:
        return self._config

    def _project_path(self) -> str:
        return f"/projects/{self._config.encoded_project()}"

    def get_issue(self, issue_key: str) -> IssueRecord:
        iid = issue_key.split("#")[-1] if "#" in issue_key else issue_key
        response = self._http.get(f"{self._project_path()}/issues/{iid}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected GitLab get_issue response")
        return _issue_from_payload(self._config, payload)

    def add_comment(self, issue_key: str, body: str) -> IssueComment:
        iid = issue_key.split("#")[-1] if "#" in issue_key else issue_key
        response = self._http.post(f"{self._project_path()}/issues/{iid}/notes", json={"body": body})
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise IntegrationConfigurationError("Unexpected GitLab add_comment response")
        author = data.get("author")
        author_name = author.get("username") if isinstance(author, dict) else None
        return IssueComment(
            id=str(data.get("id") or ""),
            body=str(data.get("body") or body),
            author=str(author_name) if author_name else None,
        )

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        response = self._http.get(
            f"{self._project_path()}/issues",
            params={"search": jql, "per_page": max(1, int(limit))},
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise IntegrationConfigurationError("Unexpected GitLab search response")
        issues = [_issue_from_payload(self._config, row) for row in rows if isinstance(row, dict)]
        return IssueSearchResult(issues=issues, total=len(issues))

    def create_issue(
        self,
        *,
        title: str,
        description: str = "",
        labels: Optional[list[str]] = None,
    ) -> IssueRecord:
        payload: dict[str, object] = {"title": title, "description": description}
        if labels:
            payload["labels"] = ",".join(labels)
        response = self._http.post(f"{self._project_path()}/issues", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise IntegrationConfigurationError("Unexpected GitLab create_issue response")
        return _issue_from_payload(self._config, data)
