# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Prometheus integration provider (Phase M.6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_observability_backend
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.prometheus.integration import (
    PrometheusObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.prometheus.bundle import (
    PrometheusIntegrationBundle,
    create_prometheus_integration,
    create_prometheus_observability_backend,
)
from intergrax.integrations.providers.observability_backend.prometheus.config import (
    ENV_PROMETHEUS_BASE_URL,
    ENV_PROMETHEUS_BEARER_TOKEN,
    PrometheusIntegrationConfig,
)
from intergrax.integrations.providers.observability_backend.prometheus.register import register_prometheus_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROMETHEUS_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "prometheus"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "PrometheusRestClient(",
    "integrations.providers.prometheus.client",
    "integrations.providers.prometheus.opens",
    "httpx.Client(",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _prometheus_config() -> PrometheusIntegrationConfig:
    return PrometheusIntegrationConfig(base_url="http://prometheus.local:9090")


def _mock_http_client(*, get_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = get_payload or {"status": "success", "data": {"resultType": "vector", "result": []}}
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
    for path in _PROMETHEUS_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "httpx" in text:
            violations.append(path.name)
    assert violations == []


def test_prometheus_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _PROMETHEUS_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_prometheus_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PROMETHEUS_BASE_URL, "http://metrics:9090")
    monkeypatch.setenv(ENV_PROMETHEUS_BEARER_TOKEN, "token")
    config = PrometheusIntegrationConfig.from_env()
    assert config.base_url == "http://metrics:9090"
    assert config.bearer_token == "token"


def test_query_instant_parses_vector() -> None:
    http = _mock_http_client(
        get_payload={
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "prometheus"},
                        "value": [1435781431.781, "1"],
                    }
                ],
            },
        }
    )
    backend = create_prometheus_observability_backend(**_prometheus_config().model_dump(), http_client=http)

    result = backend.query_instant("up")

    assert result.result_type == "vector"
    assert len(result.series) == 1
    assert result.series[0].metric["__name__"] == "up"
    assert result.series[0].points[0].value == 1.0
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == "/api/v1/query"
    assert http.get.call_args.kwargs["params"]["query"] == "up"
    assert_observability_backend(backend)


def test_query_range_parses_matrix() -> None:
    http = _mock_http_client(
        get_payload={
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "up"},
                        "values": [[1435781430.0, "1"], [1435781445.0, "0"]],
                    }
                ],
            },
        }
    )
    backend = create_prometheus_observability_backend(**_prometheus_config().model_dump(), http_client=http)

    result = backend.query_range("up", start=1435781430.0, end=1435781445.0, step="15s")

    assert result.result_type == "matrix"
    assert len(result.series[0].points) == 2
    assert result.series[0].points[1].value == 0.0
    assert http.get.call_args.args[0] == "/api/v1/query_range"


def test_query_raises_on_prometheus_error() -> None:
    http = _mock_http_client(
        get_payload={
            "status": "error",
            "errorType": "bad_data",
            "error": "invalid query",
        }
    )
    backend = create_prometheus_observability_backend(**_prometheus_config().model_dump(), http_client=http)

    with pytest.raises(IntegrationConfigurationError, match="bad_data"):
        backend.query_instant("bad(")


def test_create_prometheus_integration_bundle() -> None:
    http = _mock_http_client()
    bundle = create_prometheus_integration(**_prometheus_config().model_dump(), http_client=http)

    assert isinstance(bundle, PrometheusIntegrationBundle)
    assert isinstance(bundle.observability_backend, PrometheusObservabilityIntegration)


def test_register_and_resolve_via_profile() -> None:
    register_prometheus_integration()
    profile = IntegrationProfile(observability_backend="prometheus")
    http = _mock_http_client()

    backend = resolve(
        IntegrationCategory.OBSERVABILITY_BACKEND,
        profile=profile,
        config={**_prometheus_config().model_dump(), "http_client": http},
    )

    assert_observability_backend(backend)
    assert isinstance(backend, PrometheusObservabilityIntegration)


def test_register_default_integrations_includes_prometheus() -> None:
    register_default_integrations()
    profile = IntegrationProfile(observability_backend="prometheus")
    http = _mock_http_client()

    backend = resolve(
        IntegrationCategory.OBSERVABILITY_BACKEND,
        profile=profile,
        config={**_prometheus_config().model_dump(), "http_client": http},
    )

    assert isinstance(backend, PrometheusObservabilityIntegration)


def test_opens_creates_httpx_client_when_not_injected() -> None:
    config = _prometheus_config()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.observability_backend.prometheus.opens._create_http_client",
        return_value=mock_client,
    ) as create_mock:
        from intergrax.integrations.providers.observability_backend.prometheus.opens import open_prometheus_rest_client

        client = open_prometheus_rest_client(config)

    create_mock.assert_called_once_with(config)
    assert client.config is config
