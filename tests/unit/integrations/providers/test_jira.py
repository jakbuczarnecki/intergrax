# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Jira integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_issue_tracker
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.jira.adapter import JiraIssueTracker
from intergrax.integrations.providers.jira.bundle import (
    JiraIntegrationBundle,
    create_jira_integration,
    create_jira_issue_tracker,
)
from intergrax.integrations.providers.jira.config import (
    ENV_JIRA_API_TOKEN,
    ENV_JIRA_BASE_URL,
    ENV_JIRA_EMAIL,
    JiraIntegrationConfig,
)
from intergrax.integrations.providers.jira.register import register_jira_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_JIRA_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "jira"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "JiraRestClient(",
    "integrations.providers.jira.client",
    "integrations.providers.jira.opens",
    "httpx.Client(",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _jira_config() -> JiraIntegrationConfig:
    return JiraIntegrationConfig(
        base_url="https://example.atlassian.net",
        email="bot@example.com",
        api_token="secret",
    )


def _mock_http_client(*, get_payload: dict | None = None, post_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    get_response = MagicMock()
    get_response.json.return_value = get_payload or {}
    get_response.raise_for_status.return_value = None
    post_response = MagicMock()
    post_response.json.return_value = post_payload or {}
    post_response.raise_for_status.return_value = None
    client.get.return_value = get_response
    client.post.return_value = post_response
    return client


def _iter_python_files(*roots: str):
    for root_name in roots:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def test_httpx_client_only_created_in_opens_module() -> None:
    violations: list[str] = []
    for path in _JIRA_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "httpx" in text:
            violations.append(path.name)
    assert violations == []


def test_jira_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _JIRA_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_jira_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_JIRA_BASE_URL, "https://acme.atlassian.net")
    monkeypatch.setenv(ENV_JIRA_EMAIL, "user@acme.com")
    monkeypatch.setenv(ENV_JIRA_API_TOKEN, "token")
    config = JiraIntegrationConfig.from_env()
    assert config.base_url == "https://acme.atlassian.net"
    assert config.api_base_url == "https://acme.atlassian.net/rest/api/3"


def test_jira_config_requires_credentials() -> None:
    with pytest.raises(IntegrationConfigurationError, match="base_url is required"):
        create_jira_issue_tracker(base_url="", email="a", api_token="b", http_client=MagicMock())


def test_get_issue_parses_fields() -> None:
    http = _mock_http_client(
        get_payload={
            "key": "PROJ-1",
            "fields": {
                "summary": "Fix bug",
                "description": "Details here",
                "status": {"name": "In Progress"},
                "assignee": {"displayName": "Alex"},
            },
        }
    )
    tracker = create_jira_issue_tracker(**_jira_config().model_dump(), http_client=http)

    issue = tracker.get_issue("PROJ-1")

    assert issue.key == "PROJ-1"
    assert issue.summary == "Fix bug"
    assert issue.status == "In Progress"
    assert issue.assignee == "Alex"
    assert issue.url == "https://example.atlassian.net/browse/PROJ-1"
    http.get.assert_called_once_with("/issue/PROJ-1")
    assert_issue_tracker(tracker)


def test_add_comment_posts_adf_body() -> None:
    http = _mock_http_client(
        post_payload={
            "id": "10001",
            "body": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Done"}]}]},
            "author": {"displayName": "Bot"},
        }
    )
    tracker = create_jira_issue_tracker(**_jira_config().model_dump(), http_client=http)

    comment = tracker.add_comment("PROJ-1", "Done")

    assert comment.id == "10001"
    assert "Done" in comment.body
    assert comment.author == "Bot"
    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "/issue/PROJ-1/comment"


def test_search_issues_returns_normalized_rows() -> None:
    http = _mock_http_client(
        post_payload={
            "total": 1,
            "issues": [
                {
                    "key": "PROJ-2",
                    "fields": {"summary": "Task", "status": {"name": "Open"}},
                }
            ],
        }
    )
    tracker = create_jira_issue_tracker(**_jira_config().model_dump(), http_client=http)

    result = tracker.search_issues("project = PROJ", limit=10)

    assert result.total == 1
    assert len(result.issues) == 1
    assert result.issues[0].key == "PROJ-2"
    http.post.assert_called_once_with(
        "/search",
        json={
            "jql": "project = PROJ",
            "maxResults": 10,
            "fields": ["summary", "description", "status", "assignee"],
        },
    )


def test_create_jira_integration_bundle() -> None:
    http = _mock_http_client()
    bundle = create_jira_integration(**_jira_config().model_dump(), http_client=http)

    assert isinstance(bundle, JiraIntegrationBundle)
    assert isinstance(bundle.issue_tracker, JiraIssueTracker)


def test_register_and_resolve_via_profile() -> None:
    register_jira_integration()
    profile = IntegrationProfile(issue_tracker=IntegrationSlug.JIRA)
    http = _mock_http_client()

    tracker = resolve(
        IntegrationCategory.ISSUE_TRACKER,
        profile=profile,
        config={**_jira_config().model_dump(), "http_client": http},
    )

    assert_issue_tracker(tracker)
    assert isinstance(tracker, JiraIssueTracker)


def test_register_default_integrations_includes_jira() -> None:
    register_default_integrations()
    profile = IntegrationProfile(issue_tracker=IntegrationSlug.JIRA)
    http = _mock_http_client()

    tracker = resolve(
        IntegrationCategory.ISSUE_TRACKER,
        profile=profile,
        config={**_jira_config().model_dump(), "http_client": http},
    )

    assert isinstance(tracker, JiraIssueTracker)


def test_opens_creates_httpx_client_when_not_injected() -> None:
    config = _jira_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.jira.opens._create_http_client",
        return_value=mock_client,
    ) as create_mock:
        from intergrax.integrations.providers.jira.opens import open_jira_rest_client

        client = open_jira_rest_client(config)

    create_mock.assert_called_once_with(config)
    assert client.config is config
