# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.runtime.integrations.contracts import PlatformIntegrationKind
from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
)
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
    sanitize_application_observability_attributes,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    FORBIDDEN_EXPORT_CONTENT_FIELDS,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
)

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[5]

WAVE1_SLUGS = ("arize", "phoenix", "langsmith", "helicone", "braintrust", "wandb")
WAVE2_SLUGS = ("datadog", "sentry", "signoz", "honeycomb", "newrelic", "splunk", "posthog")
WAVE3_SLUGS = (
    "prometheus",
    "elasticsearch",
    "opensearch",
    "otel",
    "opentelemetry_collector",
    "grafana",
    "loki",
    "tempo",
    "influxdb",
    "clickhouse",
    "mlflow",
)
ALL_MIGRATED_SLUGS = WAVE1_SLUGS + WAVE2_SLUGS + WAVE3_SLUGS

LLM_SLUGS = frozenset(WAVE1_SLUGS)

# Slugs whose public integration class names are not plain per-segment .capitalize().
_CLASS_NAME_OVERRIDES: dict[str, str] = {
    "newrelic": "NewRelic",
    "opentelemetry_collector": "OpenTelemetryCollector",
}

_OBSERVABILITY_BACKEND_SLUGS = frozenset(
    slug for slug, category in SLUG_CATEGORY.items() if category == "observability_backend"
)

_FORBIDDEN_IMPORT_PREFIXES: dict[str, tuple[str, ...]] = {
    "opentelemetry_collector": ("opentelemetry",),
    "newrelic": ("newrelic",),
    "elasticsearch": ("elasticsearch",),
    "opensearch": ("opensearch",),
    "clickhouse": ("clickhouse",),
    "influxdb": ("influxdb",),
    "mlflow": ("mlflow",),
    "wandb": ("wandb",),
    "sentry": ("sentry_sdk", "sentry"),
    "datadog": ("datadog",),
    "splunk": ("splunk",),
    "posthog": ("posthog",),
    "grafana": ("grafana",),
    "prometheus": ("prometheus",),
    "otel": ("opentelemetry",),
}


def _slug_to_class(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def _provider_pkg(slug: str) -> str:
    return f"intergrax.integrations.providers.observability_backend.{slug}"


def _integration_module(slug: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug)}.integration")


def _bundle_module(slug: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug)}.bundle")


def _register_module(slug: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug)}.register")


class ExampleWorkspaceObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "local_workspace"
    operation: str | None = "index_job"
    file_count: int | None = None


class _FakeTransport:
    def __init__(self) -> None:
        self.payloads: list[ObservabilityVendorPayload] = []
        self.send_count = 0

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        self.send_count += 1
        self.payloads.append(payload)


def _sanitized_envelope_with_attributes() -> ObservabilityExportEnvelope:
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=3)
    sanitize_result = sanitize_application_observability_attributes(attributes)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        capability="search",
        event_type="tool.completed",
        status=ExportStatus.SUCCEEDED,
        latency_ms=42,
        counts={"hit_count": 2},
        tool_id="grep",
        application_attributes=attributes,
        sanitized_application_attributes=sanitize_result.sanitized,
    )
    policy_result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )
    assert policy_result.exported and policy_result.envelope is not None
    return policy_result.envelope


def _integration_class(slug: str) -> type:
    mod = _integration_module(slug)
    return getattr(mod, f"{_slug_to_class(slug)}ObservabilityIntegration")


def _create_integration_factory(slug: str) -> Any:
    return getattr(_bundle_module(slug), f"create_{slug}_observability_integration")


def _create_backend_factory(slug: str) -> Any:
    return getattr(_bundle_module(slug), f"create_{slug}_observability_backend")


def _register_fn(slug: str) -> Any:
    return getattr(_register_module(slug), f"register_{slug}_integration")


def _provider_id_const(slug: str) -> str:
    mod = _integration_module(slug)
    const_name = slug.upper().replace("-", "_") + "_OBSERVABILITY_PROVIDER_ID"
    return getattr(mod, const_name)


def _forbidden_prefixes(slug: str) -> tuple[str, ...]:
    return _FORBIDDEN_IMPORT_PREFIXES.get(slug, (slug,))


def test_layout_observability_backend_slugs_match_migrated_batch() -> None:
    """Guard: every layout observability_backend slug except Langfuse pilot is in ALL_MIGRATED_SLUGS."""
    expected_batch = _OBSERVABILITY_BACKEND_SLUGS - {"langfuse"}
    assert frozenset(ALL_MIGRATED_SLUGS) == expected_batch


def test_every_layout_observability_backend_slug_has_integration_module() -> None:
    for slug in sorted(_OBSERVABILITY_BACKEND_SLUGS):
        integration_path = (
            _PROJECT_ROOT
            / "intergrax"
            / "integrations"
            / "providers"
            / "observability_backend"
            / slug
            / "integration.py"
        )
        assert integration_path.is_file(), f"missing integration.py for observability_backend/{slug}"


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_integration_derives_from_observability_vendor_contract(slug: str) -> None:
    integration_cls = _integration_class(slug)
    integration = integration_cls.from_transport(_FakeTransport())

    assert isinstance(integration, ObservabilityVendorIntegrationContract)


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_provider_id_matches_slug(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())

    assert integration.provider_id == slug
    assert _provider_id_const(slug) == slug


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_integration_kind_is_observability_vendor(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())

    assert integration.integration_kind == PlatformIntegrationKind.OBSERVABILITY_VENDOR.value


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_config_disabled_by_default(slug: str) -> None:
    integration = _create_integration_factory(slug)(enabled=False, transport=None)

    assert integration.config.enabled is False


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_supported_signals_non_empty(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())

    assert integration.supported_signals


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_create_integration_disabled_without_transport(slug: str) -> None:
    integration = _create_integration_factory(slug)(enabled=False, transport=None)

    assert integration.transport is None
    assert integration.config.enabled is False


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_create_integration_enabled_without_transport_raises(slug: str) -> None:
    with pytest.raises(IntegrationConfigurationError, match="transport"):
        _create_integration_factory(slug)(enabled=True, transport=None)


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_create_integration_enabled_with_fake_transport(slug: str) -> None:
    transport = _FakeTransport()
    integration = _create_integration_factory(slug)(enabled=True, transport=transport)

    assert integration.transport is transport
    assert integration.config.enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
async def test_export_delivers_payload_to_fake_transport(slug: str) -> None:
    transport = _FakeTransport()
    integration = _create_integration_factory(slug)(enabled=True, transport=transport)
    envelope = _sanitized_envelope_with_attributes()

    await integration.export(envelope)

    assert transport.send_count == 1
    assert len(transport.payloads) == 1
    assert transport.payloads[0].provider_id == slug


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_map_envelope_rejects_raw_application_attributes(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=1)
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.RUNTIME_EVENT,
        application_attributes=attributes,
    )

    with pytest.raises(ValueError, match="raw application_attributes"):
        integration.map_envelope(envelope)


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_payload_excludes_forbidden_raw_content(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())
    envelope = _sanitized_envelope_with_attributes()
    payload = integration.map_envelope(envelope).payload
    serialized = json.dumps(payload.model_dump(mode="json")).lower()

    for field_name in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{field_name}"' not in serialized

    assert "application_attributes" not in payload.model_dump()


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_integration_module_has_no_vendor_sdk_imports(slug: str) -> None:
    path = (
        _PROJECT_ROOT
        / "intergrax"
        / "integrations"
        / "providers"
        / "observability_backend"
        / slug
        / "integration.py"
    )
    source = path.read_text(encoding="utf-8").lower()

    for token in _forbidden_prefixes(slug):
        assert f"import {token}" not in source
        assert f"from {token}" not in source


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_legacy_backend_factory_importable(slug: str) -> None:
    factory = _create_backend_factory(slug)
    assert callable(factory)


@pytest.mark.parametrize("slug", ALL_MIGRATED_SLUGS)
def test_register_remains_legacy_compatible(slug: str) -> None:
    register_fn = _register_fn(slug)
    source = inspect.getsource(register_fn)
    assert f"create_{slug}_observability_backend" in source
    assert f"create_{slug}_observability_integration" not in source


@pytest.mark.parametrize("slug", WAVE1_SLUGS)
def test_llm_provider_signals(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())
    assert ObservabilityVendorSignal.LLM_EVENTS in integration.supported_signals


@pytest.mark.parametrize("slug", WAVE2_SLUGS + WAVE3_SLUGS)
def test_telemetry_provider_signals(slug: str) -> None:
    integration = _integration_class(slug).from_transport(_FakeTransport())
    assert ObservabilityVendorSignal.LOGS in integration.supported_signals
    assert ObservabilityVendorSignal.METRICS in integration.supported_signals


def test_no_real_network_in_parametrized_exports() -> None:
    for slug in ALL_MIGRATED_SLUGS:
        transport = _FakeTransport()
        integration = _create_integration_factory(slug)(enabled=True, transport=transport)
        envelope = _sanitized_envelope_with_attributes()
        asyncio.run(integration.export(envelope))
        assert transport.send_count == 1
