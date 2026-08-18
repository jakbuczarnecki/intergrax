# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared immutable DTOs for domain plugin bootstrap evidence (ENTERPRISE-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.core.plugins.discovery import EntryPointLoadResult, EntryPointSpec


class PluginAdmissionReasonCode(StrEnum):
    """Operator-visible admission rejection classification."""

    INVALID_TARGET_TYPE = "invalid_target_type"
    PLUGIN_ID_COLLISION = "plugin_id_collision"
    SHIPPED_ID_COLLISION = "shipped_id_collision"
    ALREADY_REGISTERED = "already_registered"
    PLUGIN_ID_SKIPPED = "plugin_id_skipped"
    NOT_IN_ALLOWLIST = "not_in_allowlist"
    PRODUCTION_ADMISSION_DENIED = "production_admission_denied"
    INVALID_POLICY_CONTRIBUTION_SOURCE = "invalid_policy_contribution_source"
    POLICY_HANDLER_BINDING_MISSING = "policy_handler_binding_missing"
    POLICY_HANDLER_PROVENANCE_MISMATCH = "policy_handler_provenance_mismatch"
    UNRESOLVED_PACKAGE_IDENTITY = "unresolved_package_identity"


@dataclass(frozen=True, slots=True)
class PluginAdmissionRejection:
    """Admission/validation rejection — distinct from entry-point load failure."""

    spec: EntryPointSpec
    reason_code: PluginAdmissionReasonCode
    reason: str
    plugin_id: str | None = None
    fail_closed: bool = True


@dataclass(frozen=True, slots=True)
class DomainPluginLoadReport:
    """Per-bootstrap domain evidence. Not a unified operator inventory (F006)."""

    group: str
    accepted: tuple[EntryPointSpec, ...]
    rejected: tuple[PluginAdmissionRejection, ...]
    failed: tuple[EntryPointLoadResult, ...]
    registered_count: int

    @property
    def critical_bootstrap_acceptable(self) -> bool:
        """True when no load failures and no fail-closed admission rejections."""
        if self.failed:
            return False
        return not any(item.fail_closed for item in self.rejected)

    def to_audit_dict(self) -> dict[str, object]:
        """JSON-friendly metadata without live plugin targets."""
        return {
            "group": self.group,
            "registered_count": self.registered_count,
            "critical_bootstrap_acceptable": self.critical_bootstrap_acceptable,
            "accepted": tuple(_spec_audit(spec) for spec in self.accepted),
            "rejected": tuple(_rejection_audit(item) for item in self.rejected),
            "failed": tuple(_failed_audit(item) for item in self.failed),
        }

    @classmethod
    def empty(cls, group: str) -> DomainPluginLoadReport:
        return cls(
            group=group,
            accepted=(),
            rejected=(),
            failed=(),
            registered_count=0,
        )


def _spec_audit(spec: EntryPointSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "group": spec.group,
        "value": spec.value,
        "distribution": spec.distribution,
    }


def _rejection_audit(item: PluginAdmissionRejection) -> dict[str, object]:
    payload = _spec_audit(item.spec)
    payload["reason_code"] = item.reason_code.value
    payload["reason"] = item.reason
    payload["plugin_id"] = item.plugin_id
    payload["fail_closed"] = item.fail_closed
    return payload


def _failed_audit(item: EntryPointLoadResult) -> dict[str, object]:
    payload = _spec_audit(item.spec)
    error = item.error
    payload["error_type"] = type(error).__name__ if error is not None else None
    payload["error"] = str(error) if error is not None else None
    return payload
