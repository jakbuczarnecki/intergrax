# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed observability export/redaction policy and failure isolation (OBS-EXPORT-2)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.observability.export_attributes import (
    sanitize_application_observability_attributes,
    sanitized_application_attributes_are_content_safe,
)
from intergrax.runtime.observability.export_boundary import (
    NoOpObservabilityExporter,
    ObservabilityExportEnvelope,
    ObservabilityExporter,
    envelope_is_content_safe,
)

logger = logging.getLogger(__name__)


class ObservabilityExportMode(StrEnum):
    DISABLED = "disabled"
    METADATA_ONLY = "metadata_only"


class ObservabilityFieldAction(StrEnum):
    ALLOW = "allow"
    DROP = "drop"
    HASH = "hash"


@dataclass(frozen=True, slots=True)
class ObservabilityExportPolicy:
    """Local-first export posture — disabled and metadata-only by default."""

    enabled: bool = False
    export_content: bool = False
    strict_redaction: bool = True
    hash_sensitive_paths: bool = True
    allow_tenant_ids: bool = True
    allow_workspace_ids: bool = True

    @property
    def mode(self) -> ObservabilityExportMode:
        if not self.enabled:
            return ObservabilityExportMode.DISABLED
        return ObservabilityExportMode.METADATA_ONLY


@dataclass(frozen=True, slots=True)
class ExportPolicyResult:
    exported: bool
    envelope: ObservabilityExportEnvelope | None = None
    decision: ObservabilityExportMode = ObservabilityExportMode.DISABLED
    reason: str = ""


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _looks_like_unsafe_path(value: str) -> bool:
    if not value:
        return False
    normalized = value.strip()
    if normalized.startswith(("/", "\\")):
        return True
    if len(normalized) > 1 and normalized[1] == ":" and normalized[0].isalpha():
        return True
    parts = normalized.replace("\\", "/").split("/")
    return ".." in parts


def _path_field_action(
    value: str,
    *,
    policy: ObservabilityExportPolicy,
) -> str:
    if not value:
        return ""
    if not policy.strict_redaction:
        return value
    if not _looks_like_unsafe_path(value):
        return value
    if policy.hash_sensitive_paths:
        return _hash_value(value)
    return ""


def apply_observability_export_policy(
    envelope: ObservabilityExportEnvelope,
    policy: ObservabilityExportPolicy | None = None,
) -> ExportPolicyResult:
    """Apply export/redaction policy; return None envelope when export is disabled."""
    active = policy or ObservabilityExportPolicy()
    if not active.enabled:
        return ExportPolicyResult(
            exported=False,
            envelope=None,
            decision=ObservabilityExportMode.DISABLED,
            reason="export_disabled",
        )

    if active.export_content:
        return ExportPolicyResult(
            exported=False,
            envelope=None,
            decision=ObservabilityExportMode.DISABLED,
            reason="export_content_not_supported",
        )

    tenant_id = envelope.tenant_id if active.allow_tenant_ids else ""
    workspace_id = envelope.workspace_id if active.allow_workspace_ids else ""
    safe_relative_path = _path_field_action(envelope.safe_relative_path, policy=active)
    artifact_ref = _path_field_action(envelope.artifact_ref, policy=active)

    sanitized_application_attributes = None
    if envelope.application_attributes is not None:
        attribute_result = sanitize_application_observability_attributes(
            envelope.application_attributes,
            strict_redaction=active.strict_redaction,
            hash_sensitive_paths=active.hash_sensitive_paths,
        )
        sanitized_application_attributes = attribute_result.sanitized

    sanitized = envelope.model_copy(
        update={
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "safe_relative_path": safe_relative_path,
            "artifact_ref": artifact_ref,
            "application_attributes": None,
            "sanitized_application_attributes": sanitized_application_attributes,
        }
    )

    if active.strict_redaction and not envelope_is_content_safe(sanitized):
        return ExportPolicyResult(
            exported=False,
            envelope=None,
            decision=ObservabilityExportMode.METADATA_ONLY,
            reason="forbidden_content_fields",
        )

    if active.strict_redaction and not sanitized_application_attributes_are_content_safe(
        sanitized.sanitized_application_attributes
    ):
        return ExportPolicyResult(
            exported=False,
            envelope=None,
            decision=ObservabilityExportMode.METADATA_ONLY,
            reason="forbidden_application_attribute_fields",
        )

    return ExportPolicyResult(
        exported=True,
        envelope=sanitized,
        decision=ObservabilityExportMode.METADATA_ONLY,
        reason="metadata_only",
    )


async def try_export_observability_envelope(
    envelope: ObservabilityExportEnvelope,
    *,
    exporter: ObservabilityExporter | None = None,
    policy: ObservabilityExportPolicy | None = None,
) -> ExportPolicyResult:
    """Apply policy then export; exporter failures never propagate to callers."""
    active_exporter = exporter or NoOpObservabilityExporter()
    policy_result = apply_observability_export_policy(envelope, policy)
    if not policy_result.exported or policy_result.envelope is None:
        return policy_result

    try:
        await active_exporter.export(policy_result.envelope)
    except Exception:
        logger.warning(
            "observability export failed run_id=%s event_id=%s record_kind=%s",
            policy_result.envelope.run_id,
            policy_result.envelope.event_id,
            policy_result.envelope.record_kind.value,
            exc_info=True,
        )
        return ExportPolicyResult(
            exported=False,
            envelope=policy_result.envelope,
            decision=policy_result.decision,
            reason="exporter_failed",
        )

    return policy_result
