# © Artur Czarnecki. All rights reserved.

"""Capability health output redaction (P1.5 / P1.4A)."""

from __future__ import annotations

import re

from intergrax.applications._shared.profile_resolution.redaction import encode_provenance_value
from intergrax.applications.contracts.capability_health.fact import (
    CapabilityHealthFact,
    CapabilityHealthReason,
)
from intergrax.applications.contracts.capability_health.result import (
    CapabilityHealthProviderFailure,
    EffectiveCapabilityHealth,
)
from intergrax.applications.contracts.capability_health.safe_views import (
    SafeCapabilityHealthFactView,
    SafeCapabilityHealthProviderFailureView,
    SafeCapabilityHealthReasonView,
    SafeEffectiveCapabilityHealthView,
)

_EMBEDDED_SECRET_MARKERS: frozenset[str] = frozenset(
    {
        "secret",
        "token",
        "api_key",
        "credential",
        "password",
    },
)
_KEY_VALUE_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|credential|password)\s*[:=]\s*\S+",
)


def _message_likely_contains_secret(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _EMBEDDED_SECRET_MARKERS):
        return True
    return _KEY_VALUE_SECRET_PATTERN.search(text) is not None


def sanitize_health_text(path: str, text: str) -> str:
    """Redact free-form health text that may embed secret material."""
    if not text:
        return text
    if not _message_likely_contains_secret(text):
        return text
    encoded = encode_provenance_value(f"{path}.secret", text)
    return encoded if encoded is not None else text


def sanitize_health_provider_failure_reason(exc: BaseException) -> str:
    """Return stable, sanitized provider failure reason without raw secret leakage."""
    message = str(exc).strip()
    exc_type = type(exc).__name__
    if not message:
        return exc_type
    sanitized = sanitize_health_text("provider_failure.reason", message)
    return f"{exc_type}: {sanitized}"


def _safe_reason(reason: CapabilityHealthReason) -> SafeCapabilityHealthReasonView:
    path = f"health.reason.{reason.reason_code}.{reason.subject_ref}"
    return SafeCapabilityHealthReasonView(
        reason_code=reason.reason_code,
        source=reason.source,
        subject_ref=reason.subject_ref,
        detail=(
            sanitize_health_text(path, reason.detail)
            if reason.detail is not None
            else None
        ),
    )


def _safe_fact(fact: CapabilityHealthFact) -> SafeCapabilityHealthFactView:
    return SafeCapabilityHealthFactView(
        capability=fact.capability,
        source=fact.source,
        condition_kind=fact.condition_kind.value,
        condition_ref=fact.condition_ref,
        scope_application_id=fact.scope_application_id,
        scope_tenant_id=fact.scope_tenant_id,
        status=fact.status.value,
        blocking=fact.blocking,
        reason=_safe_reason(fact.reason),
        provider_id=fact.provider_id,
    )


def _safe_provider_failure(
    failure: CapabilityHealthProviderFailure,
) -> SafeCapabilityHealthProviderFailureView:
    return SafeCapabilityHealthProviderFailureView(
        provider_id=failure.provider_id,
        reason=sanitize_health_text(
            f"health.provider_failure.{failure.provider_id}",
            failure.reason,
        ),
    )


def safe_effective_capability_health_view(
    health: EffectiveCapabilityHealth,
) -> SafeEffectiveCapabilityHealthView:
    """Build serialized health view safe for direct model_dump."""
    return SafeEffectiveCapabilityHealthView(
        schema_version=health.schema_version,
        capability=health.capability,
        status=health.status,
        reasons=tuple(_safe_reason(item) for item in health.reasons),
        facts=tuple(_safe_fact(item) for item in health.facts),
        provenance=health.provenance,
        provider_failures=tuple(
            _safe_provider_failure(item) for item in health.provider_failures
        ),
        effective_profile_revision_id=health.effective_profile_revision_id,
        effective_profile_fingerprint=health.effective_profile_fingerprint,
    )
