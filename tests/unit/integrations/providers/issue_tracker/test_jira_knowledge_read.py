# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Jira knowledge-read provider surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult
from intergrax.integrations.providers.issue_tracker.jira.adapter import _JiraIssueTracker
from intergrax.integrations.providers.issue_tracker.jira.bundle import create_jira_issue_tracker
from intergrax.integrations.providers.issue_tracker.jira.client import JiraRestClient
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.integration import JiraIssueTrackerIntegration
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JiraKnowledgeIssue,
    JiraKnowledgeIssuePage,
    JiraKnowledgeUser,
    parse_jira_knowledge_issue,
    validate_jira_issue_key,
    validate_jira_project_key,
)

pytestmark = pytest.mark.unit


def _config() -> JiraIntegrationConfig:
    return JiraIntegrationConfig(
        base_url="https://example.atlassian.net",
        email="bot@example.com",
        api_token="top-secret-token",
    )


def _issue_fields(
    *,
    issue_id: str = "10001",
    key: str = "PROJ-1",
    summary: str = "Summary",
    description: object = "Plain description",
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": summary,
            "description": description,
            "status": {"id": "3", "name": "In Progress"},
            "issuetype": {"id": "1", "name": "Task"},
            "project": {"id": "10000", "key": "PROJ", "name": "Project"},
            "priority": {"name": "High"},
            "labels": ["backend"],
            "components": [{"name": "API"}],
            "assignee": {
                "accountId": "acc-1",
                "displayName": "Alex",
                "active": True,
            },
            "reporter": {
                "accountId": "acc-2",
                "displayName": "Reporter",
                "active": True,
            },
            "resolution": {"name": "Done"},
            "created": "2024-01-01T10:00:00.000+0000",
            "updated": "2024-01-02T11:00:00.000+0000",
        },
    }


def _mock_http(*, post_payload: dict | None = None, get_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    post_response = MagicMock()
    post_response.status_code = 200
    post_response.json.return_value = post_payload or {}
    post_response.raise_for_status.return_value = None
    get_response = MagicMock()
    get_response.status_code = 200
    get_response.json.return_value = get_payload or {}
    get_response.raise_for_status.return_value = None
    client.post.return_value = post_response
    client.get.return_value = get_response
    return client


class _CustomIssueTracker:
    def get_issue(self, issue_key: str) -> IssueRecord:
        return IssueRecord(key=issue_key, summary="ok")

    def add_comment(self, issue_key: str, body: str):
        raise NotImplementedError

    def search_issues(self, jql: str, *, limit: int = 50) -> IssueSearchResult:
        return IssueSearchResult(issues=[], total=0)


def test_search_knowledge_issues_uses_search_jql_endpoint() -> None:
    http = _mock_http(
        post_payload={
            "issues": [_issue_fields()],
            "isLast": True,
        }
    )
    client = JiraRestClient(_config(), http_client=http)

    client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=100)

    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "/search/jql"
    assert "/search" != http.post.call_args.args[0]


def test_search_knowledge_issues_does_not_use_legacy_search() -> None:
    http = _mock_http(post_payload={"issues": [], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert http.post.call_args.args[0] != "/search"


def test_search_knowledge_issues_request_contains_project_jql_and_order() -> None:
    http = _mock_http(post_payload={"issues": [], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=25)
    body = http.post.call_args.kwargs["json"]
    assert body["jql"] == 'project = "PROJ" ORDER BY id ASC'
    assert body["maxResults"] == 25


def test_invalid_project_key_rejected_before_http() -> None:
    http = _mock_http()
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="invalid Jira project key"):
        client.search_knowledge_issues(project_key="proj;drop", next_page_token=None, limit=10)
    http.post.assert_not_called()


def test_first_page_without_token() -> None:
    http = _mock_http(post_payload={"issues": [_issue_fields()], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    page = client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    body = http.post.call_args.kwargs["json"]
    assert "nextPageToken" not in body
    assert len(page.issues) == 1


def test_continuation_passes_next_page_token() -> None:
    http = _mock_http(post_payload={"issues": [], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    client.search_knowledge_issues(project_key="PROJ", next_page_token="page-2", limit=10)
    body = http.post.call_args.kwargs["json"]
    assert body["nextPageToken"] == "page-2"


def test_is_last_false_requires_token() -> None:
    http = _mock_http(post_payload={"issues": [], "isLast": False})
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_is_last_true_rejects_token() -> None:
    http = _mock_http(
        post_payload={"issues": [], "isLast": True, "nextPageToken": "leftover"}
    )
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_duplicate_issue_id_rejected() -> None:
    issue = _issue_fields()
    http = _mock_http(post_payload={"issues": [issue, issue], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_missing_issue_id_rejected() -> None:
    payload = _issue_fields()
    del payload["id"]
    http = _mock_http(post_payload={"issues": [payload], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_adf_description_converted_to_plain_text() -> None:
    adf = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ],
    }
    http = _mock_http(
        post_payload={
            "issues": [_issue_fields(description=adf)],
            "isLast": True,
        }
    )
    client = JiraRestClient(_config(), http_client=http)
    page = client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert page.issues[0].description == "Hello"


def test_timestamps_are_timezone_aware() -> None:
    http = _mock_http(post_payload={"issues": [_issue_fields()], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    page = client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert page.issues[0].created_at.tzinfo is not None
    assert page.issues[0].updated_at.tzinfo is not None


def test_get_knowledge_issue_uses_explicit_fields() -> None:
    http = _mock_http(get_payload=_issue_fields())
    client = JiraRestClient(_config(), http_client=http)
    client.get_knowledge_issue(issue_key="PROJ-1")
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == "/issue/PROJ-1"
    fields = http.get.call_args.kwargs["params"]["fields"]
    assert "comment" not in fields
    assert "attachment" not in fields
    assert "summary" in fields
    assert "description" in fields


def test_get_knowledge_issue_rejects_invalid_key_before_http() -> None:
    http = _mock_http()
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="invalid Jira issue key"):
        client.get_knowledge_issue(issue_key="bad key")
    http.get.assert_not_called()


@pytest.mark.parametrize("status_code,expected", [(429, IntegrationDependencyError), (503, IntegrationDependencyError)])
def test_rate_limit_and_5xx_map_to_dependency_error(status_code: int, expected: type[Exception]) -> None:
    http = _mock_http()
    http.post.return_value.status_code = status_code
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(expected):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_client_errors_map_to_configuration_error(status_code: int) -> None:
    http = _mock_http()
    http.post.return_value.status_code = status_code
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_get_issue_404_is_retryable_dependency_failure() -> None:
    http = _mock_http()
    http.get.return_value.status_code = 404
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError):
        client.get_knowledge_issue(issue_key="PROJ-99")


def test_errors_do_not_expose_token_jql_or_response_body() -> None:
    http = _mock_http()
    http.post.return_value.status_code = 400
    http.post.return_value.text = "raw-body-with-secret"
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token="page-2", limit=10)
    message = str(exc_info.value)
    assert "page-2" not in message
    assert "project =" not in message
    assert "raw-body" not in message
    assert "top-secret-token" not in message


def test_jira_issue_tracker_delegates_knowledge_read() -> None:
    http = _mock_http(
        post_payload={"issues": [_issue_fields()], "isLast": True},
        get_payload=_issue_fields(),
    )
    rest = JiraRestClient(_config(), http_client=http)
    tracker = _JiraIssueTracker(rest)
    page = tracker.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=5)
    assert len(page.issues) == 1
    issue = tracker.get_knowledge_issue(issue_key="PROJ-1")
    assert issue.key == "PROJ-1"


def test_jira_integration_delegates_knowledge_read() -> None:
    http = _mock_http(
        post_payload={"issues": [_issue_fields()], "isLast": True},
        get_payload=_issue_fields(),
    )
    rest = JiraRestClient(_config(), http_client=http)
    integration = JiraIssueTrackerIntegration.from_client(_JiraIssueTracker(rest))
    page = integration.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=5)
    assert len(page.issues) == 1
    issue = integration.get_knowledge_issue(issue_key="PROJ-1")
    assert issue.remote_id == "10001"


def test_custom_issue_tracker_without_knowledge_read_fails_closed() -> None:
    integration = JiraIssueTrackerIntegration.from_client(_CustomIssueTracker())
    assert integration.get_issue("PROJ-1").summary == "ok"
    with pytest.raises(IntegrationConfigurationError, match="knowledge read capability"):
        integration.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_project_and_issue_key_validators() -> None:
    assert validate_jira_project_key("PROJ") == "PROJ"
    assert validate_jira_issue_key("PROJ-1") == "PROJ-1"
    with pytest.raises(ValueError):
        validate_jira_project_key("bad key")
    with pytest.raises(ValueError):
        validate_jira_issue_key("PROJ-0")


def test_existing_jira_tracker_still_works() -> None:
    http = _mock_http(
        get_payload={
            "key": "PROJ-1",
            "fields": {
                "summary": "Fix bug",
                "description": "Details",
                "status": {"name": "Open"},
            },
        },
        post_payload={"total": 0, "issues": []},
    )
    tracker = create_jira_issue_tracker(**_config().model_dump(), http_client=http)
    issue = tracker.get_issue("PROJ-1")
    assert issue.key == "PROJ-1"
    result = tracker.search_issues("project = PROJ", limit=5)
    assert result.total == 0


def _valid_issue_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "remote_id": "10001",
        "key": "PROJ-1",
        "summary": "Summary",
        "description": "Description",
        "status_name": "In Progress",
        "issue_type_name": "Task",
        "project_id": "10000",
        "project_key": "PROJ",
        "project_name": "Project",
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
        "web_url": "https://example.atlassian.net/browse/PROJ-1",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("remote_id", ["abc", "0"])
def test_jira_knowledge_issue_rejects_invalid_remote_id(remote_id: str) -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeIssue(**_valid_issue_kwargs(remote_id=remote_id))


def test_jira_knowledge_issue_rejects_malformed_issue_key() -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeIssue(**_valid_issue_kwargs(key="bad key"))


def test_jira_knowledge_issue_rejects_malformed_project_key() -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeIssue(**_valid_issue_kwargs(project_key="bad key"))


def test_jira_knowledge_issue_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeIssue(
            **_valid_issue_kwargs(created_at=datetime(2024, 1, 1, 10, 0))
        )


def test_jira_knowledge_issue_rejects_naive_updated_at() -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeIssue(
            **_valid_issue_kwargs(updated_at=datetime(2024, 1, 2, 11, 0))
        )


def test_jira_knowledge_issue_normalizes_non_utc_aware_datetime() -> None:
    offset = timezone(timedelta(hours=-5))
    issue = JiraKnowledgeIssue(
        **_valid_issue_kwargs(
            created_at=datetime(2024, 1, 1, 10, 0, tzinfo=offset),
            updated_at=datetime(2024, 1, 2, 11, 0, tzinfo=offset),
        )
    )
    assert issue.created_at.tzinfo == timezone.utc
    assert issue.updated_at.tzinfo == timezone.utc


def test_jira_knowledge_issue_page_rejects_duplicate_remote_ids() -> None:
    issue = JiraKnowledgeIssue(**_valid_issue_kwargs())
    with pytest.raises(ValueError, match="duplicate issue id on page"):
        JiraKnowledgeIssuePage(issues=(issue, issue), is_last=True)


def test_jira_knowledge_user_rejects_empty_optional_identity() -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeUser(account_id="   ")
    with pytest.raises(ValueError):
        JiraKnowledgeUser(display_name="")


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_id", 123),
        ("display_name", True),
    ],
)
def test_jira_knowledge_user_rejects_non_string_identity(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        JiraKnowledgeUser(**{field: value})


def test_jira_knowledge_user_trims_valid_string() -> None:
    user = JiraKnowledgeUser(account_id="  acc-1  ", display_name="  Alex  ")
    assert user.account_id == "acc-1"
    assert user.display_name == "Alex"


def test_parser_rejects_integer_account_id() -> None:
    payload = _issue_fields()
    payload["fields"]["assignee"] = {
        "accountId": 123,
        "displayName": "Alex",
        "active": True,
    }
    http = _mock_http(post_payload={"issues": [payload], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response") as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert exc_info.value.__cause__ is None
    assert "123" not in str(exc_info.value)


def test_parser_rejects_list_display_name() -> None:
    payload = _issue_fields()
    payload["fields"]["assignee"] = {
        "accountId": "acc-1",
        "displayName": ["secret-name"],
        "active": True,
    }
    http = _mock_http(post_payload={"issues": [payload], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response") as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert exc_info.value.__cause__ is None
    assert "secret-name" not in str(exc_info.value)


def test_parser_rejects_malformed_timestamps_without_leaking_values() -> None:
    payload = _issue_fields()
    payload["fields"]["created"] = "secret-created-value"
    payload["fields"]["updated"] = "secret-updated-value"
    http = _mock_http(post_payload={"issues": [payload], "isLast": True})
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response") as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert exc_info.value.__cause__ is None
    message = str(exc_info.value)
    assert "secret-created-value" not in message
    assert "secret-updated-value" not in message


def test_get_knowledge_issue_parser_rejects_malformed_timestamps_without_leaking_values() -> None:
    payload = _issue_fields()
    payload["fields"]["created"] = "secret-created-value"
    payload["fields"]["updated"] = "secret-updated-value"
    http = _mock_http(get_payload=payload)
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response") as exc_info:
        client.get_knowledge_issue(issue_key="PROJ-1")
    assert exc_info.value.__cause__ is None
    message = str(exc_info.value)
    assert "secret-created-value" not in message
    assert "secret-updated-value" not in message


def test_parse_jira_knowledge_issue_trims_user_strings() -> None:
    payload = _issue_fields()
    payload["fields"]["assignee"] = {
        "accountId": "  acc-1  ",
        "displayName": "  Alex  ",
        "active": True,
    }
    issue = parse_jira_knowledge_issue(
        payload,
        issue_url="https://example.atlassian.net/browse/PROJ-1",
        plain_description=lambda raw: str(raw or ""),
    )
    assert issue.assignee is not None
    assert issue.assignee.account_id == "acc-1"
    assert issue.assignee.display_name == "Alex"


def test_jira_knowledge_issue_page_rejects_empty_next_page_token() -> None:
    with pytest.raises(ValueError, match="next_page_token"):
        JiraKnowledgeIssuePage(issues=(), next_page_token="", is_last=False)
    with pytest.raises(ValueError, match="next_page_token"):
        JiraKnowledgeIssuePage(issues=(), next_page_token="   ", is_last=False)


def test_jira_knowledge_issue_page_rejects_non_string_next_page_token() -> None:
    with pytest.raises(ValueError, match="next_page_token"):
        JiraKnowledgeIssuePage(issues=(), next_page_token=123, is_last=False)  # type: ignore[arg-type]


def test_jira_knowledge_models_are_frozen_and_extra_forbid() -> None:
    issue = JiraKnowledgeIssue(**_valid_issue_kwargs())
    with pytest.raises(Exception):
        issue.summary = "changed"  # type: ignore[misc]
    with pytest.raises(Exception):
        JiraKnowledgeIssue(**_valid_issue_kwargs(), unexpected=True)  # type: ignore[call-arg]


def _issue_fields_for_project(
    *,
    project_key: str,
    issue_key: str,
) -> dict[str, Any]:
    payload = _issue_fields(key=issue_key)
    payload["fields"]["project"]["key"] = project_key
    return payload


def test_project_scope_mismatch_both_keys_other() -> None:
    http = _mock_http(
        post_payload={
            "issues": [_issue_fields_for_project(project_key="OTHER", issue_key="OTHER-1")],
            "isLast": True,
        }
    )
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="does not belong to requested project") as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    message = str(exc_info.value)
    assert "PROJ" not in message
    assert "OTHER" not in message
    assert "OTHER-1" not in message
    assert "project =" not in message


def test_project_scope_mismatch_project_key_matches_issue_key_other() -> None:
    http = _mock_http(
        post_payload={
            "issues": [_issue_fields_for_project(project_key="PROJ", issue_key="OTHER-1")],
            "isLast": True,
        }
    )
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="does not belong to requested project"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_project_scope_mismatch_issue_key_proj_project_other() -> None:
    http = _mock_http(
        post_payload={
            "issues": [_issue_fields_for_project(project_key="OTHER", issue_key="PROJ-1")],
            "isLast": True,
        }
    )
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="does not belong to requested project"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_post_transport_exception_maps_to_dependency_error() -> None:
    http = _mock_http()
    http.post.side_effect = RuntimeError("network down secret")
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError) as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert exc_info.value.__cause__ is None
    assert "network down" not in str(exc_info.value)


def test_get_transport_exception_maps_to_dependency_error() -> None:
    http = _mock_http()
    http.get.side_effect = RuntimeError("socket reset secret")
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError) as exc_info:
        client.get_knowledge_issue(issue_key="PROJ-1")
    assert exc_info.value.__cause__ is None
    assert "socket reset" not in str(exc_info.value)


def test_json_decode_exception_is_safe() -> None:
    http = _mock_http()
    http.post.return_value.json.side_effect = ValueError("secret body payload")
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Jira knowledge response") as exc_info:
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)
    assert exc_info.value.__cause__ is None
    assert "secret body" not in str(exc_info.value)


def test_unhandled_4xx_maps_to_configuration_error_without_raise_for_status() -> None:
    http = _mock_http()
    http.post.return_value.status_code = 418
    http.post.return_value.raise_for_status.side_effect = AssertionError("raise_for_status called")
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError):
        client.search_knowledge_issues(project_key="PROJ", next_page_token=None, limit=10)


def test_empty_next_page_token_rejected_before_http() -> None:
    http = _mock_http()
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="next_page_token"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token="", limit=10)
    http.post.assert_not_called()


def test_whitespace_next_page_token_rejected_before_http() -> None:
    http = _mock_http()
    client = JiraRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="next_page_token"):
        client.search_knowledge_issues(project_key="PROJ", next_page_token="   ", limit=10)
    http.post.assert_not_called()
