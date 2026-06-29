# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Elasticsearch integration provider (Phase M.6 P2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_observability_backend
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.observability_backend.elasticsearch.adapter import _ElasticsearchObservabilityBackend
from intergrax.integrations.providers.observability_backend.elasticsearch.bundle import (
    ElasticsearchIntegrationBundle,
    create_elasticsearch_integration,
    create_elasticsearch_observability_backend,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    ENV_ELASTICSEARCH_INDEX,
    ENV_ELASTICSEARCH_URL,
    ElasticsearchIntegrationConfig,
)
from intergrax.integrations.providers.observability_backend.elasticsearch.register import register_elasticsearch_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ES_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "elasticsearch"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "ElasticsearchRestClient(",
    "integrations.providers.elasticsearch.client",
    "integrations.providers.elasticsearch.opens",
    "httpx.Client(",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _elasticsearch_config() -> ElasticsearchIntegrationConfig:
    return ElasticsearchIntegrationConfig(
        base_url="http://elasticsearch.local:9200",
        index="logs-*",
    )


def _mock_http_client(*, post_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = post_payload or {}
    response.raise_for_status.return_value = None
    client.post.return_value = response
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
    for path in _ES_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "httpx" in text:
            violations.append(path.name)
    assert violations == []


def test_elasticsearch_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _ES_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_elasticsearch_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ELASTICSEARCH_URL, "http://es:9200")
    monkeypatch.setenv(ENV_ELASTICSEARCH_INDEX, "app-logs-*")
    config = ElasticsearchIntegrationConfig.from_env()
    assert config.base_url == "http://es:9200"
    assert config.index == "app-logs-*"


def test_query_instant_parses_value_count() -> None:
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 42}}})
    backend = create_elasticsearch_observability_backend(
        **_elasticsearch_config().model_dump(),
        http_client=http,
    )

    result = backend.query_instant("level:error", eval_time=1_700_000_000.0)

    assert result.result_type == "vector"
    assert len(result.series) == 1
    assert result.series[0].points[0].value == 42.0
    assert result.series[0].points[0].timestamp == 1_700_000_000.0
    http.post.assert_called_once()
    assert http.post.call_args.args[0] == "/logs-*/_search"
    assert_observability_backend(backend)


def test_query_range_parses_date_histogram() -> None:
    http = _mock_http_client(
        post_payload={
            "aggregations": {
                "timeline": {
                    "buckets": [
                        {"key": 1_700_000_000_000, "doc_count": 5},
                        {"key": 1_700_000_015_000, "doc_count": 7},
                    ]
                }
            }
        }
    )
    backend = create_elasticsearch_observability_backend(
        **_elasticsearch_config().model_dump(),
        http_client=http,
    )

    result = backend.query_range(
        "service:api",
        start=1_700_000_000.0,
        end=1_700_000_030.0,
        step="15s",
    )

    assert result.result_type == "matrix"
    assert len(result.series[0].points) == 2
    assert result.series[0].points[0].value == 5.0
    body = http.post.call_args.kwargs["json"]
    assert body["aggs"]["timeline"]["date_histogram"]["fixed_interval"] == "15s"


def test_create_elasticsearch_integration_bundle() -> None:
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 0}}})
    bundle = create_elasticsearch_integration(**_elasticsearch_config().model_dump(), http_client=http)

    assert isinstance(bundle, ElasticsearchIntegrationBundle)
    assert isinstance(bundle.observability_backend, _ElasticsearchObservabilityBackend)


def test_register_and_resolve_via_profile() -> None:
    register_elasticsearch_integration()
    profile = IntegrationProfile(observability_backend="elasticsearch")
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 1}}})

    backend = resolve(
        IntegrationCategory.OBSERVABILITY_BACKEND,
        profile=profile,
        config={**_elasticsearch_config().model_dump(), "http_client": http},
    )

    assert_observability_backend(backend)
    assert isinstance(backend, _ElasticsearchObservabilityBackend)


def test_register_default_integrations_includes_elasticsearch() -> None:
    register_default_integrations()
    profile = IntegrationProfile(observability_backend="elasticsearch")
    http = _mock_http_client(post_payload={"aggregations": {"count": {"value": 1}}})

    backend = resolve(
        IntegrationCategory.OBSERVABILITY_BACKEND,
        profile=profile,
        config={**_elasticsearch_config().model_dump(), "http_client": http},
    )

    assert isinstance(backend, _ElasticsearchObservabilityBackend)


def test_opens_creates_httpx_client_when_not_injected() -> None:
    config = _elasticsearch_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.observability_backend.elasticsearch.opens._create_http_client",
        return_value=mock_client,
    ) as create_mock:
        from intergrax.integrations.providers.observability_backend.elasticsearch.opens import open_elasticsearch_rest_client

        client = open_elasticsearch_rest_client(config)

    create_mock.assert_called_once_with(config)
    assert client.config is config
