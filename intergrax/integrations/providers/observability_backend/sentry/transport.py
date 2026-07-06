# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sentry observability export transport (OBS-SENTRY-1)."""

from __future__ import annotations

import asyncio
from typing import Any

from intergrax.integrations.providers.observability_backend.sentry.client import SentryCaptureClient
from intergrax.runtime.integrations.observability import ObservabilityVendorPayload
from intergrax.runtime.observability.export_attributes import ObservabilityAttributeValue

_SEVERITY_TO_LEVEL: dict[str, str] = {
    "critical": "fatal",
    "fatal": "fatal",
    "error": "error",
    "warning": "warning",
    "warn": "warning",
    "info": "info",
    "debug": "debug",
}

SentryEventPayload = dict[str, object]


def _set_tag(tags: dict[str, str], key: str, value: str) -> None:
    if value:
        tags[key] = value


def _attribute_value_to_extra(value: ObservabilityAttributeValue) -> Any:
    if isinstance(value, list):
        return list(value)
    return value


def _safe_extra_from_sanitized_attributes(
    payload: ObservabilityVendorPayload,
) -> dict[str, object]:
    sanitized = payload.sanitized_application_attributes
    if sanitized is None:
        return {}
    extra: dict[str, object] = {}
    if sanitized.namespace:
        extra["application_namespace"] = sanitized.namespace
    for key, value in sorted(sanitized.attributes.items()):
        extra[key] = _attribute_value_to_extra(value)
    return extra


def _sentry_message(payload: ObservabilityVendorPayload) -> str:
    if payload.record_type == "problem_signal":
        if payload.problem_kind:
            return f"Intergrax problem: {payload.problem_kind}"
        if payload.problem_error_code:
            return f"Intergrax problem: {payload.problem_error_code}"
    if payload.problem_kind:
        return f"Intergrax problem: {payload.problem_kind}"
    if payload.problem_error_code:
        return f"Intergrax problem: {payload.problem_error_code}"
    return f"Intergrax observability: {payload.record_type}"


def _sentry_level(payload: ObservabilityVendorPayload) -> str:
    severity = payload.problem_severity.casefold()
    if severity:
        mapped = _SEVERITY_TO_LEVEL.get(severity)
        if mapped is not None:
            return mapped
        if payload.record_type == "problem_signal":
            return "error"
    if payload.record_type == "problem_signal":
        return "error"
    return "info"


def map_vendor_payload_to_sentry_event(payload: ObservabilityVendorPayload) -> SentryEventPayload:
    """Map a policy-safe vendor payload to a Sentry issue-shaped event."""
    tags: dict[str, str] = {}
    _set_tag(tags, "intergrax.record_type", payload.record_type)
    _set_tag(tags, "intergrax.provider_id", payload.provider_id)
    _set_tag(tags, "intergrax.integration_id", payload.integration_id)
    _set_tag(tags, "intergrax.integration_kind", payload.integration_kind)
    _set_tag(tags, "intergrax.problem_kind", payload.problem_kind)
    _set_tag(tags, "intergrax.problem_severity", payload.problem_severity)
    _set_tag(tags, "intergrax.problem_error_code", payload.problem_error_code)
    _set_tag(tags, "intergrax.run_id", payload.run_id)
    _set_tag(tags, "intergrax.task_id", payload.task_id)
    _set_tag(tags, "intergrax.agent_id", payload.agent_id)
    _set_tag(tags, "intergrax.capability", payload.capability)
    _set_tag(tags, "intergrax.tool_id", payload.tool_id)
    _set_tag(tags, "intergrax.status", payload.status)
    _set_tag(tags, "intergrax.correlation_id", payload.correlation_id)
    _set_tag(tags, "intergrax.event_id", payload.event_id)
    _set_tag(tags, "intergrax.tenant_id", payload.tenant_id)
    _set_tag(tags, "intergrax.workspace_id", payload.workspace_id)
    _set_tag(tags, "intergrax.source_schema_id", payload.source_schema_id)

    intergrax_context: dict[str, object] = {
        "record_type": payload.record_type,
        "provider_id": payload.provider_id,
        "integration_id": payload.integration_id,
        "integration_kind": payload.integration_kind,
        "recorded_at": payload.recorded_at.isoformat(),
    }
    if payload.run_id:
        intergrax_context["run_id"] = payload.run_id
    if payload.task_id:
        intergrax_context["task_id"] = payload.task_id
    if payload.agent_id:
        intergrax_context["agent_id"] = payload.agent_id
    if payload.correlation_id:
        intergrax_context["correlation_id"] = payload.correlation_id
    if payload.event_id:
        intergrax_context["event_id"] = payload.event_id

    contexts: dict[str, dict[str, object]] = {"intergrax": intergrax_context}

    if payload.record_type == "problem_signal" or any(
        (payload.problem_kind, payload.problem_severity, payload.problem_error_code)
    ):
        problem_context: dict[str, object] = {}
        if payload.problem_kind:
            problem_context["problem_kind"] = payload.problem_kind
        if payload.problem_severity:
            problem_context["problem_severity"] = payload.problem_severity
        if payload.problem_error_code:
            problem_context["problem_error_code"] = payload.problem_error_code
        if problem_context:
            contexts["intergrax_problem"] = problem_context

    if any((payload.artifact_ref, payload.sha256, payload.safe_relative_path)):
        artifact_context: dict[str, object] = {}
        if payload.artifact_ref:
            artifact_context["artifact_ref"] = payload.artifact_ref
        if payload.sha256:
            artifact_context["sha256"] = payload.sha256
        if payload.safe_relative_path:
            artifact_context["safe_relative_path"] = payload.safe_relative_path
        contexts["intergrax_artifact"] = artifact_context

    extra: dict[str, object] = {}
    if payload.counts:
        extra["counts"] = dict(payload.counts)
    if payload.latency_ms is not None:
        extra["latency_ms"] = payload.latency_ms
    extra.update(_safe_extra_from_sanitized_attributes(payload))

    event: SentryEventPayload = {
        "message": _sentry_message(payload),
        "level": _sentry_level(payload),
        "fingerprint": [
            "intergrax",
            payload.record_type,
            payload.problem_kind or payload.event_type or "",
            payload.problem_error_code or "",
        ],
        "tags": tags,
        "contexts": contexts,
    }
    if extra:
        event["extra"] = extra
    return event


class SentrySdkObservabilityTransport:
    """Deliver policy-safe observability payloads via provider-owned Sentry capture client."""

    def __init__(
        self,
        client: SentryCaptureClient,
        *,
        flush_after_capture: bool = False,
        flush_timeout: float | None = None,
    ) -> None:
        self._client = client
        self._flush_after_capture = flush_after_capture
        self._flush_timeout = flush_timeout

    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:
        event = map_vendor_payload_to_sentry_event(payload)
        await asyncio.to_thread(self._client.capture_event, event)
        if self._flush_after_capture:
            await asyncio.to_thread(self._client.flush, self._flush_timeout)
