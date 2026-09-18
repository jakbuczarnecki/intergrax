# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host/admission-bound root governance identity (ADR-GOVERNED-EXECUTION-003).

Construct only after authentication or durable binding resolution at the Host/Admission
boundary. Execution and Governance consume this type; they must not derive it from
Task payloads, metadata, or runtime request fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_non_empty(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must be non-empty")
    return stripped


@dataclass(frozen=True, slots=True)
class AdmittedRootGovernanceIdentity:
    """Atomic admitted tenant/workspace/principal triple for governed root execution."""

    tenant_id: str
    workspace_id: str
    principal_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant_id", _require_non_empty(self.tenant_id, label="tenant_id")
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, label="workspace_id"),
        )
        object.__setattr__(
            self,
            "principal_id",
            _require_non_empty(self.principal_id, label="principal_id"),
        )


__all__ = ["AdmittedRootGovernanceIdentity"]
