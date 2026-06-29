# © Artur Czarnecki. All rights reserved.

"""Regression guards — observability integration owns catalog behavior (no legacy delegates)."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.observability_backend import MetricQueryResult, TraceQueryResult
from intergrax.integrations.providers.layout import SLUG_CATEGORY

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_OBS_ROOT = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "observability_backend"

INLINE_WAVE_SLUGS: tuple[str, ...] = (
    "braintrust",
    "langsmith",
    "prometheus",
    "elasticsearch",
    "opensearch",
    "langfuse",
    "datadog",
    "clickhouse",
    "sentry",
    "helicone",
    "posthog",
    "signoz",
    "honeycomb",
    "arize",
    "phoenix",
    "wandb",
)

_FORBIDDEN_INTEGRATION_SNIPPETS: tuple[str, ...] = (
    "__pydantic_private__",
    "def _require_runtime",
    "def __getattr__(self, name",
    "_backend: Any",
)

_CLASS_NAME_OVERRIDES: dict[str, str] = {
    "newrelic": "NewRelic",
    "opentelemetry_collector": "OpenTelemetryCollector",
}


def _slug_to_class(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def _integration_class(slug: str) -> type:
    mod = importlib.import_module(f"intergrax.integrations.providers.observability_backend.{slug}.integration")
    return getattr(mod, f"{_slug_to_class(slug)}ObservabilityIntegration")


def _legacy_factory(slug: str) -> Any:
    bundle = importlib.import_module(f"intergrax.integrations.providers.observability_backend.{slug}.bundle")
    return getattr(bundle, f"create_{slug}_observability_backend")


class _FakeCatalogClient:
    def query_instant(self, promql: str, *, eval_time: float | None = None) -> MetricQueryResult:
        del promql, eval_time
        return MetricQueryResult(result_type="vector")

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        del promql, start, end, step
        return MetricQueryResult(result_type="matrix")

    def query_traces(self, *, limit: int = 20, name: str | None = None) -> TraceQueryResult:
        del limit, name
        return TraceQueryResult()


def test_observability_integration_modules_forbid_legacy_delegation_patterns() -> None:
    violations: list[str] = []
    for path in sorted(_OBS_ROOT.glob("*/integration.py")):
        source = path.read_text(encoding="utf-8")
        for snippet in _FORBIDDEN_INTEGRATION_SNIPPETS:
            if snippet in source:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {snippet!r}")
    assert violations == []


@pytest.mark.parametrize("slug", INLINE_WAVE_SLUGS)
def test_integration_has_no_runtime_delegate_surface(slug: str) -> None:
    integration_cls = _integration_class(slug)
    assert not hasattr(integration_cls, "_require_runtime")
    assert "_require_runtime" not in inspect.getsource(integration_cls)
    assert "def __getattr__" not in inspect.getsource(integration_cls)
    instance = integration_cls.from_client(_FakeCatalogClient())  # type: ignore[arg-type]
    assert getattr(instance, "_backend", "missing") in (None, "missing")


@pytest.mark.parametrize("slug", INLINE_WAVE_SLUGS)
def test_integration_query_methods_use_client_not_private_backend(slug: str) -> None:
    integration = _integration_class(slug).from_client(_FakeCatalogClient())  # type: ignore[arg-type]
    assert integration.query_instant("up").result_type == "vector"
    assert integration.query_range("up", start=0.0, end=1.0).result_type == "matrix"
    assert integration.query_traces(limit=1).traces == []


@pytest.mark.parametrize("slug", INLINE_WAVE_SLUGS)
def test_legacy_factory_returns_integration_without_private_adapter_class(slug: str) -> None:
    integration_cls = _integration_class(slug)
    factory = _legacy_factory(slug)
    if slug in {"braintrust", "langsmith", "prometheus", "elasticsearch", "opensearch"}:
        backend = factory(client=_FakeCatalogClient())  # type: ignore[call-arg]
    else:
        backend = factory(client=_FakeCatalogClient())
    assert isinstance(backend, integration_cls)
    assert type(backend).__name__ == integration_cls.__name__
    assert not type(backend).__name__.startswith("_")


def test_observability_backend_layout_slugs_have_integration_modules() -> None:
    slugs = {slug for slug, category in SLUG_CATEGORY.items() if category == "observability_backend"}
    missing = [slug for slug in sorted(slugs) if not (_OBS_ROOT / slug / "integration.py").is_file()]
    assert missing == []


def test_braintrust_log_eval_on_integration() -> None:
    from intergrax.integrations.providers.observability_backend.braintrust.integration import (
        BraintrustObservabilityIntegration,
    )

    client = MagicMock()
    client.log_eval.return_value = "log-42"
    integration = BraintrustObservabilityIntegration.from_client(client)
    assert integration.log_eval(name="accuracy", score=0.9) == "log-42"


def test_prometheus_health_on_integration() -> None:
    from intergrax.integrations.providers.observability_backend.prometheus.integration import (
        PrometheusObservabilityIntegration,
    )

    client = MagicMock()
    client.health.return_value = True
    integration = PrometheusObservabilityIntegration.from_client(client)
    assert integration.health().healthy is True
