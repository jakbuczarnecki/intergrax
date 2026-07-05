# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Sentry observability provider transport (OBS-SENTRY-1)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.sentry.bundle import (
    create_sentry_observability_integration,
    create_sentry_observability_transport,
)
from intergrax.integrations.providers.observability_backend.sentry.client import (
    SentrySdkCaptureClient,
    open_sentry_sdk_capture_client,
)
from intergrax.integrations.providers.observability_backend.sentry.config import (
    ENV_SENTRY_DSN,
    ENV_SENTRY_ENVIRONMENT,
    SentryIntegrationConfig,
)
from intergrax.integrations.providers.observability_backend.sentry.integration import (
    SentryObservabilityIntegration,
)
from intergrax.integrations.providers.observability_backend.sentry.transport import (
    SentrySdkObservabilityTransport,
    map_vendor_payload_to_sentry_event,
)
from intergrax.runtime.integrations.observability import (
    OBSERVABILITY_VENDOR_PAYLOAD_SCHEMA,
    ObservabilityVendorPayload,
)
from intergrax.runtime.observability.export_attributes import (
    ApplicationObservabilityAttributes,
    observability_attribute_key,
    sanitize_application_observability_attributes,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    apply_observability_export_policy,
)

pytestmark = pytest.mark.unit


class ExampleWorkspaceObservabilityAttributes(ApplicationObservabilityAttributes):
    namespace: str = "local_workspace"
    operation: str | None = "index_job"
    file_count: int | None = None


class FakeSentryCaptureClient:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.flush_calls: list[float | None] = []

    def capture_event(self, event: dict[str, object]) -> str | None:
        self.events.append(dict(event))
        return "fake-event-id"

    def flush(self, timeout: float | None = None) -> None:
        self.flush_calls.append(timeout)


def _problem_payload(**overrides: object) -> ObservabilityVendorPayload:
    base: dict[str, object] = {
        "provider_id": "sentry",
        "integration_id": "sentry:observability_vendor",
        "integration_kind": "observability_vendor",
        "record_type": "problem_signal",
        "recorded_at": datetime(2026, 7, 5, 8, 0, 0, tzinfo=timezone.utc),
        "problem_kind": "lkw.retrieve_failed",
        "problem_severity": "error",
        "problem_error_code": "LKW_RETRIEVE_FAILED",
        "run_id": "run-1",
        "task_id": "task-1",
        "agent_id": "agent-1",
        "correlation_id": "corr-1",
        "event_id": "event-1",
    }
    base.update(overrides)
    return ObservabilityVendorPayload.model_validate(base)


def _problem_envelope(**overrides: object) -> ObservabilityExportEnvelope:
    envelope = ObservabilityExportEnvelope(
        record_kind=ExportRecordKind.PROBLEM_SIGNAL,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        correlation_id="corr-1",
        event_id="event-1",
        problem_kind="lkw.retrieve_failed",
        problem_severity="error",
        problem_error_code="LKW_RETRIEVE_FAILED",
        status=ExportStatus.FAILED,
        **{k: v for k, v in overrides.items() if k not in {"record_kind"}},
    )
    policy_result = apply_observability_export_policy(
        envelope,
        ObservabilityExportPolicy(enabled=True),
    )
    assert policy_result.exported and policy_result.envelope is not None
    return policy_result.envelope


def test_problem_payload_maps_to_sentry_issue_event() -> None:
    payload = _problem_payload()
    event = map_vendor_payload_to_sentry_event(payload)

    assert "lkw.retrieve_failed" in str(event["message"])
    assert event["level"] == "error"
    assert event["fingerprint"] == [
        "intergrax",
        "problem_signal",
        "lkw.retrieve_failed",
        "LKW_RETRIEVE_FAILED",
    ]
    tags = event["tags"]
    assert isinstance(tags, dict)
    assert tags["intergrax.problem_kind"] == "lkw.retrieve_failed"
    assert tags["intergrax.problem_severity"] == "error"
    assert tags["intergrax.problem_error_code"] == "LKW_RETRIEVE_FAILED"
    assert tags["intergrax.run_id"] == "run-1"
    assert tags["intergrax.correlation_id"] == "corr-1"

    serialized = json.dumps(event).lower()
    forbidden = (
        "application_attributes",
        "raw prompt text",
        "secret-api-key",
        "tool_arguments",
        "raw_chunks",
        "headers",
        "authorization",
        "token",
        "exception",
        "stacktrace",
        "traceback",
        "user",
    )
    for sample in forbidden:
        assert sample not in serialized


@pytest.mark.parametrize(
    ("severity", "expected_level"),
    [
        ("critical", "fatal"),
        ("fatal", "fatal"),
        ("error", "error"),
        ("warning", "warning"),
        ("warn", "warning"),
        ("info", "info"),
        ("debug", "debug"),
        ("unknown", "error"),
    ],
)
def test_severity_mapping(severity: str, expected_level: str) -> None:
    payload = _problem_payload(problem_severity=severity)
    event = map_vendor_payload_to_sentry_event(payload)
    assert event["level"] == expected_level


def test_non_problem_payload_defaults_to_info_level() -> None:
    payload = _problem_payload(record_type="runtime_event", problem_severity="", problem_kind="")
    event = map_vendor_payload_to_sentry_event(payload)
    assert event["level"] == "info"


def test_sanitized_attributes_are_mapped_safely() -> None:
    attributes = ExampleWorkspaceObservabilityAttributes(file_count=3)
    sanitize_result = sanitize_application_observability_attributes(attributes)
    payload = _problem_payload(sanitized_application_attributes=sanitize_result.sanitized)
    event = map_vendor_payload_to_sentry_event(payload)

    extra = event.get("extra")
    assert isinstance(extra, dict)
    assert extra["application_namespace"] == "local_workspace"
    assert extra[observability_attribute_key("local_workspace", "file_count")] == 3
    assert "application_attributes" not in json.dumps(event)


def test_vendor_payload_has_no_raw_application_attributes_field() -> None:
  payload = _problem_payload()
  assert "application_attributes" not in ObservabilityVendorPayload.model_fields
  assert payload.schema_id == OBSERVABILITY_VENDOR_PAYLOAD_SCHEMA


@pytest.mark.asyncio
async def test_transport_sends_mapped_event_to_fake_client() -> None:
    fake_client = FakeSentryCaptureClient()
    transport = SentrySdkObservabilityTransport(fake_client)
    payload = _problem_payload()

    await transport.send_observability_payload(payload)

    assert len(fake_client.events) == 1
    event = fake_client.events[0]
    assert "lkw.retrieve_failed" in str(event["message"])
    assert event["level"] == "error"
    assert event["tags"]["intergrax.problem_kind"] == "lkw.retrieve_failed"


def test_transport_factory_with_injected_client_does_not_import_sentry_sdk() -> None:
    before_modules = set(sys.modules)
    fake_client = FakeSentryCaptureClient()
    transport = create_sentry_observability_transport(client=fake_client)
    new_modules = set(sys.modules) - before_modules

    assert isinstance(transport, SentrySdkObservabilityTransport)
    assert not any(
        module_name == "sentry_sdk" or module_name.startswith("sentry_sdk.")
        for module_name in new_modules
    )


def test_real_sdk_factory_imports_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sdk = MagicMock()
    fake_sdk.init.return_value = None
    fake_sdk.capture_event.return_value = "evt-1"
    fake_sdk.flush.return_value = None
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake_sdk)

    config = SentryIntegrationConfig(dsn="https://example@sentry.io/1", environment="test")
    client = open_sentry_sdk_capture_client(config)
    event_id = client.capture_event({"message": "test", "level": "info"})

    fake_sdk.init.assert_called_once()
    init_kwargs = fake_sdk.init.call_args.kwargs
    assert init_kwargs["dsn"] == "https://example@sentry.io/1"
    assert init_kwargs["environment"] == "test"
    assert init_kwargs["send_default_pii"] is False
    assert init_kwargs["attach_stacktrace"] is False
    fake_sdk.capture_event.assert_called_once()
    assert event_id == "evt-1"


def test_real_sdk_factory_requires_dsn() -> None:
    with pytest.raises(IntegrationConfigurationError, match="DSN"):
        open_sentry_sdk_capture_client(SentryIntegrationConfig(dsn=""))


def test_create_transport_without_dsn_raises() -> None:
    with pytest.raises(IntegrationConfigurationError, match="DSN"):
        create_sentry_observability_transport(dsn="")


@pytest.mark.asyncio
async def test_enabled_integration_exports_through_shared_contract() -> None:
    fake_client = FakeSentryCaptureClient()
    transport = SentrySdkObservabilityTransport(fake_client)
    integration = create_sentry_observability_integration(transport=transport, enabled=True)
    envelope = _problem_envelope()

    await integration.export(envelope)

    assert len(fake_client.events) == 1
    event = fake_client.events[0]
    assert "lkw.retrieve_failed" in str(event["message"])
    assert event["tags"]["intergrax.provider_id"] == "sentry"


@pytest.mark.asyncio
async def test_disabled_integration_does_not_export() -> None:
    fake_client = FakeSentryCaptureClient()
    transport = SentrySdkObservabilityTransport(fake_client)
    integration = create_sentry_observability_integration(transport=transport, enabled=False)
    envelope = _problem_envelope()

    await integration.export(envelope)

    assert fake_client.events == []


def test_create_integration_enabled_without_transport_raises() -> None:
    with pytest.raises(IntegrationConfigurationError, match="transport"):
        create_sentry_observability_integration(enabled=True, transport=None)


def test_sentry_event_serialization_excludes_forbidden_samples() -> None:
    payload = _problem_payload()
    event = map_vendor_payload_to_sentry_event(payload)
    serialized = json.dumps(event).lower()
    forbidden_samples = (
        "raw prompt text",
        "secret-api-key",
        "/home/user/secret/document.pdf",
        "c:\\users\\secret\\document.pdf",
        "tool_arguments",
        "raw_chunks",
        "headers",
        "authorization",
        "token",
    )
    for sample in forbidden_samples:
        assert sample not in serialized


def test_sentry_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SENTRY_DSN, "https://example@sentry.io/1")
    monkeypatch.setenv(ENV_SENTRY_ENVIRONMENT, "staging")
    config = SentryIntegrationConfig.from_env()
    assert config.dsn == "https://example@sentry.io/1"
    assert config.environment == "staging"
    assert config.send_default_pii is False
    assert config.attach_stacktrace is False


def test_sentry_config_rejects_negative_shutdown_timeout() -> None:
    with pytest.raises(ValueError, match="shutdown_timeout_seconds"):
        SentryIntegrationConfig(dsn="https://example@sentry.io/1", shutdown_timeout_seconds=-1)


def test_integration_factory_with_transport() -> None:
    fake_client = FakeSentryCaptureClient()
    transport = SentrySdkObservabilityTransport(fake_client)
    integration = create_sentry_observability_integration(transport=transport, enabled=True)
    assert isinstance(integration, SentryObservabilityIntegration)
    assert integration.transport is transport
