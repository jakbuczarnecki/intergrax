# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Confluence integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_wiki_knowledge
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import (
    ConfluenceIntegrationBundle,
    create_confluence_integration,
    create_confluence_wiki_knowledge,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.config import (
    ENV_CONFLUENCE_API_TOKEN,
    ENV_CONFLUENCE_BASE_URL,
    ENV_CONFLUENCE_EMAIL,
    ConfluenceIntegrationConfig,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.register import register_confluence_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFLUENCE_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "confluence"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "ConfluenceRestClient(",
    "integrations.providers.confluence.client",
    "integrations.providers.confluence.opens",
    "httpx.Client(",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _confluence_config() -> ConfluenceIntegrationConfig:
    return ConfluenceIntegrationConfig(
        base_url="https://example.atlassian.net/wiki",
        email="bot@example.com",
        api_token="secret",
    )


def _mock_http_client(*, get_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = get_payload or {}
    response.raise_for_status.return_value = None
    client.get.return_value = response
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
    for path in _CONFLUENCE_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "httpx" in text:
            violations.append(path.name)
    assert violations == []


def test_confluence_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _CONFLUENCE_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_confluence_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_CONFLUENCE_BASE_URL, "https://acme.atlassian.net/wiki")
    monkeypatch.setenv(ENV_CONFLUENCE_EMAIL, "user@acme.com")
    monkeypatch.setenv(ENV_CONFLUENCE_API_TOKEN, "token")
    config = ConfluenceIntegrationConfig.from_env()
    assert config.base_url == "https://acme.atlassian.net/wiki"
    assert config.api_base_url == "https://acme.atlassian.net/wiki/rest/api"


def test_confluence_config_requires_credentials() -> None:
    with pytest.raises(IntegrationConfigurationError, match="base_url is required"):
        create_confluence_wiki_knowledge(base_url="", email="a", api_token="b", http_client=MagicMock())


def test_get_page_parses_storage_body() -> None:
    http = _mock_http_client(
        get_payload={
            "id": "42",
            "title": "Runbook",
            "space": {"key": "OPS"},
            "version": {"number": 3},
            "body": {"storage": {"value": "<p>Restart service</p>"}},
        }
    )
    wiki = create_confluence_wiki_knowledge(**_confluence_config().model_dump(), http_client=http)

    page = wiki.get_page("42")

    assert page.id == "42"
    assert page.title == "Runbook"
    assert page.space_key == "OPS"
    assert page.body == "Restart service"
    assert page.version == 3
    assert page.url.endswith("pageId=42")
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == "/content/42"
    assert_wiki_knowledge(wiki)


def test_search_pages_builds_cql_and_parses_results() -> None:
    http = _mock_http_client(
        get_payload={
            "totalSize": 1,
            "results": [
                {
                    "id": "99",
                    "title": "Guide",
                    "space": {"key": "DOC"},
                    "body": {"storage": {"value": "<p>Hello</p>"}},
                }
            ],
        }
    )
    wiki = create_confluence_wiki_knowledge(**_confluence_config().model_dump(), http_client=http)

    result = wiki.search_pages("hello", limit=5)

    assert result.total == 1
    assert len(result.pages) == 1
    assert result.pages[0].title == "Guide"
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == "/content/search"
    params = http.get.call_args.kwargs["params"]
    assert params["cql"] == 'type=page AND text ~ "hello"'
    assert params["limit"] == 5


def test_create_confluence_integration_bundle() -> None:
    http = _mock_http_client()
    bundle = create_confluence_integration(**_confluence_config().model_dump(), http_client=http)

    assert isinstance(bundle, ConfluenceIntegrationBundle)
    assert isinstance(bundle.wiki_knowledge, ConfluenceWikiKnowledgeIntegration)


def test_register_and_resolve_via_profile() -> None:
    register_confluence_integration()
    profile = IntegrationProfile(wiki_knowledge="confluence")
    http = _mock_http_client()

    wiki = resolve(
        IntegrationCategory.WIKI_KNOWLEDGE,
        profile=profile,
        config={**_confluence_config().model_dump(), "http_client": http},
    )

    assert_wiki_knowledge(wiki)
    assert isinstance(wiki, ConfluenceWikiKnowledgeIntegration)


def test_register_default_integrations_includes_confluence() -> None:
    register_default_integrations()
    profile = IntegrationProfile(wiki_knowledge="confluence")
    http = _mock_http_client()

    wiki = resolve(
        IntegrationCategory.WIKI_KNOWLEDGE,
        profile=profile,
        config={**_confluence_config().model_dump(), "http_client": http},
    )

    assert isinstance(wiki, ConfluenceWikiKnowledgeIntegration)


def test_opens_creates_httpx_client_when_not_injected() -> None:
    config = _confluence_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.wiki_knowledge.confluence.opens._create_http_client",
        return_value=mock_client,
    ) as create_mock:
        from intergrax.integrations.providers.wiki_knowledge.confluence.opens import open_confluence_rest_client

        client = open_confluence_rest_client(config)

    create_mock.assert_called_once_with(config)
    assert client.config is config
