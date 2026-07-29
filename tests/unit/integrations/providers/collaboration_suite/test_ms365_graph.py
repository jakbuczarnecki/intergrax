# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MS365 Graph integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_collaboration_suite
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.bundle import (
    Ms365GraphIntegrationBundle,
    create_ms365_graph_collaboration_suite,
    create_ms365_graph_integration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    ENV_MS365_CLIENT_ID,
    ENV_MS365_CLIENT_SECRET,
    ENV_MS365_TENANT_ID,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.register import register_ms365_graph_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MS365_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "ms365_graph"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "GraphRestClient(",
    "integrations.providers.ms365_graph.client",
    "integrations.providers.ms365_graph.opens",
    "httpx.Client(",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _ms365_config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
    )


def _mock_http_client(*, get_payload: dict | list | None = None, post_payload: dict | None = None) -> MagicMock:
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
    for path in _MS365_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "httpx" in text:
            violations.append(path.name)
    assert violations == []


def test_ms365_graph_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _MS365_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_ms365_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MS365_TENANT_ID, "tenant-abc")
    monkeypatch.setenv(ENV_MS365_CLIENT_ID, "client-xyz")
    monkeypatch.setenv(ENV_MS365_CLIENT_SECRET, "s3cr3t")
    config = Ms365GraphIntegrationConfig.from_env()
    assert config.tenant_id == "tenant-abc"
    assert "tenant-abc" in config.token_url


def test_ms365_config_requires_credentials() -> None:
    with pytest.raises(IntegrationConfigurationError, match="tenant_id is required"):
        create_ms365_graph_collaboration_suite(
            tenant_id="",
            client_id="a",
            client_secret="b",
            http_client=MagicMock(),
        )


def test_get_message_parses_fields() -> None:
    http = _mock_http_client(
        get_payload={
            "id": "msg-1",
            "subject": "Hello",
            "bodyPreview": "Preview text",
            "from": {"emailAddress": {"address": "sender@example.com"}},
            "receivedDateTime": "2026-05-29T10:00:00Z",
        }
    )
    suite = create_ms365_graph_collaboration_suite(
        **_ms365_config().model_dump(),
        http_client=http,
    )

    message = suite.get_message("user@example.com", "msg-1")

    assert message.id == "msg-1"
    assert message.subject == "Hello"
    assert message.from_address == "sender@example.com"
    http.get.assert_called_once()
    path = http.get.call_args.args[0]
    assert path.startswith("/users/")
    assert path.endswith("/messages/msg-1")
    assert_collaboration_suite(suite)


def test_list_messages_returns_normalized_rows() -> None:
    http = _mock_http_client(
        get_payload={
            "value": [
                {
                    "id": "msg-2",
                    "subject": "Follow up",
                    "bodyPreview": "Body",
                    "from": {"emailAddress": {"address": "a@example.com"}},
                    "receivedDateTime": "2026-05-29T11:00:00Z",
                }
            ]
        }
    )
    suite = create_ms365_graph_collaboration_suite(
        **_ms365_config().model_dump(),
        http_client=http,
    )

    result = suite.list_messages("user@example.com", folder="inbox", limit=10)

    assert result.total == 1
    assert result.messages[0].subject == "Follow up"
    http.get.assert_called_once()
    assert "/mailFolders/inbox/messages" in http.get.call_args.args[0]


def test_send_mail_posts_payload() -> None:
    http = _mock_http_client()
    suite = create_ms365_graph_collaboration_suite(
        **_ms365_config().model_dump(),
        http_client=http,
    )

    suite.send_mail(
        "user@example.com",
        subject="Report",
        body="All good",
        to=["recipient@example.com"],
    )

    http.post.assert_called_once()
    assert http.post.call_args.args[0].endswith("/sendMail")
    payload = http.post.call_args.kwargs["json"]
    assert payload["message"]["subject"] == "Report"
    assert payload["message"]["toRecipients"][0]["emailAddress"]["address"] == "recipient@example.com"


def test_send_mail_requires_recipients() -> None:
    http = _mock_http_client()
    suite = create_ms365_graph_collaboration_suite(
        **_ms365_config().model_dump(),
        http_client=http,
    )

    with pytest.raises(IntegrationConfigurationError, match="at least one recipient"):
        suite.send_mail("user@example.com", subject="x", body="y", to=[])


def test_list_calendar_events_parses_window() -> None:
    http = _mock_http_client(
        get_payload={
            "value": [
                {
                    "id": "evt-1",
                    "subject": "Standup",
                    "start": {"dateTime": "2026-05-29T09:00:00"},
                    "end": {"dateTime": "2026-05-29T09:30:00"},
                    "location": {"displayName": "Teams"},
                    "organizer": {"emailAddress": {"address": "lead@example.com"}},
                }
            ]
        }
    )
    suite = create_ms365_graph_collaboration_suite(
        **_ms365_config().model_dump(),
        http_client=http,
    )

    result = suite.list_calendar_events(
        "user@example.com",
        start="2026-05-29T00:00:00Z",
        end="2026-05-30T00:00:00Z",
        limit=5,
    )

    assert result.total == 1
    assert result.events[0].subject == "Standup"
    assert result.events[0].location == "Teams"
    params = http.get.call_args.kwargs["params"]
    assert params["startDateTime"] == "2026-05-29T00:00:00Z"


def test_get_user_parses_directory_row() -> None:
    http = _mock_http_client(
        get_payload={
            "id": "user-1",
            "displayName": "Alex Example",
            "mail": "alex@example.com",
        }
    )
    suite = create_ms365_graph_collaboration_suite(
        **_ms365_config().model_dump(),
        http_client=http,
    )

    user = suite.get_user("alex@example.com")

    assert user.id == "user-1"
    assert user.display_name == "Alex Example"
    assert user.email == "alex@example.com"


def test_create_ms365_graph_integration_bundle() -> None:
    http = _mock_http_client()
    bundle = create_ms365_graph_integration(**_ms365_config().model_dump(), http_client=http)

    assert isinstance(bundle, Ms365GraphIntegrationBundle)
    assert isinstance(bundle.collaboration_suite, Ms365GraphCollaborationSuiteIntegration)


def test_register_and_resolve_via_profile() -> None:
    register_ms365_graph_integration()
    profile = IntegrationProfile(collaboration_suite="ms365_graph")
    http = _mock_http_client()

    suite = resolve(
        IntegrationCategory.COLLABORATION_SUITE,
        profile=profile,
        config={**_ms365_config().model_dump(), "http_client": http},
    )

    assert_collaboration_suite(suite)
    assert isinstance(suite, Ms365GraphCollaborationSuiteIntegration)


def test_register_default_integrations_includes_ms365_graph() -> None:
    register_default_integrations()
    profile = IntegrationProfile(collaboration_suite="ms365_graph")
    http = _mock_http_client()

    suite = resolve(
        IntegrationCategory.COLLABORATION_SUITE,
        profile=profile,
        config={**_ms365_config().model_dump(), "http_client": http},
    )

    assert isinstance(suite, Ms365GraphCollaborationSuiteIntegration)


def test_opens_creates_httpx_client_when_not_injected() -> None:
    config = _ms365_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.collaboration_suite.ms365_graph.opens._create_http_client",
        return_value=mock_client,
    ) as create_mock:
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.opens import open_graph_rest_client

        client = open_graph_rest_client(config)

    create_mock.assert_called_once_with(config)
    assert client.config is config


def test_opens_accepts_access_token_without_token_fetch() -> None:
    config = _ms365_config()

    with patch("httpx.Client") as client_cls:
        mock_client = MagicMock()
        client_cls.return_value = mock_client
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.opens import open_graph_rest_client

        client = open_graph_rest_client(config, access_token="test-token")

    assert client_cls.call_count == 2
    graph_kwargs = client_cls.call_args_list[0].kwargs
    assert graph_kwargs["follow_redirects"] is False
    assert graph_kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert client.config is config
