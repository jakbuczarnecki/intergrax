# © Artur Czarnecki. All rights reserved.

"""Inspection output redaction (P1.4 / P1.4A)."""

from __future__ import annotations

import json
import re
from typing import Any

from intergrax.applications._shared.profile_resolution.redaction import encode_provenance_value
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution.decision import ProfileResolutionWarning
from intergrax.applications.contracts.profile_resolution.diff import (
    EffectiveProfileDiff,
    ProfileDiffEntry,
)
from intergrax.applications.contracts.profile_resolution.layer import ProfileLayer
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision import EffectiveProfileRevision
from intergrax.applications.contracts.runtime_inspection.provider import InspectionExtensionEvidence
from intergrax.applications.contracts.runtime_inspection.safe_views import (
    RedactedProfileSnapshot,
    SafeEffectiveProfileDiffView,
    SafeEffectiveProfileRevisionView,
    SafeProfileDiffEntryView,
    SafeProfileLayerView,
    SafeProfileResolutionView,
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


def _redact_mapping(path: str, value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _redact_mapping(f"{path}.{key}" if path else key, item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_redact_mapping(f"{path}[{index}]", item) for index, item in enumerate(value)]
    encoded = encode_provenance_value(path, value)
    if encoded is None:
        return None
    return encoded


def redacted_profile_snapshot(profile: ApplicationEnvironmentProfile) -> dict[str, Any]:
    """Serialize profile for inspection without leaking secret-like values."""
    payload = profile.model_dump(mode="json")
    redacted = _redact_mapping("", payload)
    assert isinstance(redacted, dict)
    return redacted


def profile_contains_no_raw_secrets(
    payload: dict[str, Any] | str,
    *,
    raw_secret: str,
) -> bool:
    """Return True when raw secret value is absent from serialized inspection output."""
    serialized = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else payload
    return raw_secret not in serialized


def _message_likely_contains_secret(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _EMBEDDED_SECRET_MARKERS):
        return True
    return _KEY_VALUE_SECRET_PATTERN.search(text) is not None


def sanitize_inspection_text(path: str, text: str) -> str:
    """Redact free-form inspection text that may embed secret material."""
    if not text:
        return text
    if not _message_likely_contains_secret(text):
        return text
    encoded = encode_provenance_value(f"{path}.secret", text)
    return encoded if encoded is not None else text


def sanitize_provider_failure_reason(exc: BaseException) -> str:
    """Return stable, sanitized provider failure reason without raw secret leakage."""
    message = str(exc).strip()
    exc_type = type(exc).__name__
    if not message:
        return exc_type
    sanitized = sanitize_inspection_text("provider_failure.reason", message)
    return f"{exc_type}: {sanitized}"


def sanitize_extension_payload(payload: dict[str, str]) -> dict[str, str]:
    """Defensively redact extension evidence payload values."""
    sanitized: dict[str, str] = {}
    for key, value in sorted(payload.items()):
        path = f"extension_evidence.payload.{key}"
        encoded = encode_provenance_value(path, value)
        candidate = encoded if encoded is not None else value
        if candidate == value and _message_likely_contains_secret(value):
            encoded_secret = encode_provenance_value(f"{path}.secret", value)
            candidate = encoded_secret if encoded_secret is not None else candidate
        sanitized[key] = candidate
    return sanitized


def sanitize_extension_evidence(
    evidence: InspectionExtensionEvidence,
) -> InspectionExtensionEvidence:
    """Return extension evidence safe for direct serialization."""
    return evidence.model_copy(
        update={"payload": sanitize_extension_payload(evidence.payload)},
    )


def _safe_warning(warning: ProfileResolutionWarning) -> ProfileResolutionWarning:
    return warning.model_copy(
        update={"message": sanitize_inspection_text(warning.path, warning.message)},
    )


def safe_profile_layer_view(layer: object) -> SafeProfileLayerView:
    """Build serialized layer view with redacted delta payload."""
    layer_enum = getattr(layer, "layer")
    revision = getattr(layer, "revision", None)
    delta = getattr(layer, "delta", None)
    redacted_delta: dict[str, Any] | None = None
    if delta is not None:
        dumped = delta.model_dump(mode="json")
        redacted = _redact_mapping("layer.delta", dumped)
        assert isinstance(redacted, dict)
        redacted_delta = redacted
    return SafeProfileLayerView(
        layer=ProfileLayer(layer_enum),
        revision=revision,
        redacted_delta=redacted_delta,
    )


def safe_profile_resolution_view(resolution: ProfileResolution) -> SafeProfileResolutionView:
    """Build serialized profile resolution view without raw effective profile."""
    return SafeProfileResolutionView(
        fingerprint=resolution.fingerprint,
        effective_profile=RedactedProfileSnapshot(
            values=redacted_profile_snapshot(resolution.effective_profile),
        ),
        layers=tuple(safe_profile_layer_view(layer) for layer in resolution.layers),
        decisions=resolution.decisions,
        warnings=tuple(_safe_warning(item) for item in resolution.warnings),
        dependency_failures=tuple(
            item.model_copy(
                update={"reason": sanitize_inspection_text(item.capability, item.reason)},
            )
            for item in resolution.dependency_failures
        ),
        degraded_capabilities=tuple(
            item.model_copy(
                update={"reason": sanitize_inspection_text(item.capability, item.reason)},
            )
            for item in resolution.degraded_capabilities
        ),
    )


def safe_effective_profile_revision_view(
    revision: EffectiveProfileRevision,
) -> SafeEffectiveProfileRevisionView:
    """Build serialized revision view without raw canonical objects."""
    return SafeEffectiveProfileRevisionView(
        revision_id=revision.revision_id,
        fingerprint=revision.fingerprint,
        scope=revision.scope,
        predecessor_revision_id=revision.predecessor_revision_id,
        effective_profile=RedactedProfileSnapshot(
            values=redacted_profile_snapshot(revision.effective_profile),
        ),
        resolution=safe_profile_resolution_view(revision.resolution),
    )


def _safe_diff_value(path: str, value: str | None) -> str | None:
    if value is None:
        return None
    encoded = encode_provenance_value(path, value)
    if encoded != value:
        return encoded
    if _message_likely_contains_secret(value):
        redacted = encode_provenance_value(f"{path}.secret", value)
        return redacted if redacted is not None else encoded
    return encoded


def _safe_diff_entry(entry: ProfileDiffEntry) -> SafeProfileDiffEntryView:
    return SafeProfileDiffEntryView(
        path=entry.path,
        before=_safe_diff_value(entry.path, entry.before),
        after=_safe_diff_value(entry.path, entry.after),
        change_kind=entry.change_kind,
    )


def safe_effective_profile_diff_view(diff: EffectiveProfileDiff) -> SafeEffectiveProfileDiffView:
    """Build serialized semantic diff with redacted before/after values."""
    return SafeEffectiveProfileDiffView(
        schema_version=diff.schema_version,
        from_revision_id=diff.from_revision_id,
        to_revision_id=diff.to_revision_id,
        from_fingerprint=diff.from_fingerprint,
        to_fingerprint=diff.to_fingerprint,
        entries=tuple(_safe_diff_entry(entry) for entry in diff.entries),
    )
