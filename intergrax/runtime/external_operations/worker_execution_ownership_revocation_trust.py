# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-bound MAC trust for worker execution ownership revocation (L3-R9)."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationAuthorizationRejectedError,
    WorkerExecutionOwnershipRevocationAuthorizationV1,
    WorkerExecutionOwnershipRevocationTrustNotConfiguredError,
)

_QUALIFICATION_MAC_KEY = b"tool-life-x-qualification-revocation-mac-v1"
_ENV_MAC_KEY = "INTERGRAX_WORKER_EXECUTION_REVOCATION_MAC_KEY"

_TRUST: WorkerExecutionOwnershipRevocationTrustMaterial | None = None


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationTrustMaterial:
    """Unforgeable MAC key bound at composition — not derivable from public authorization fields."""

    mac_key: bytes

    def __post_init__(self) -> None:
        if not self.mac_key:
            raise ValueError("revocation MAC key must be non-empty")


def configure_worker_execution_ownership_revocation_trust(
    material: WorkerExecutionOwnershipRevocationTrustMaterial,
) -> None:
    global _TRUST
    _TRUST = material


def require_worker_execution_ownership_revocation_trust() -> (
    WorkerExecutionOwnershipRevocationTrustMaterial
):
    if _TRUST is None:
        raise WorkerExecutionOwnershipRevocationTrustNotConfiguredError(
            "worker execution ownership revocation trust is not configured",
        )
    return _TRUST


def resolve_worker_execution_revocation_trust_from_environment() -> (
    WorkerExecutionOwnershipRevocationTrustMaterial
):
    raw = os.environ.get(_ENV_MAC_KEY, "").strip()
    if not raw:
        raise WorkerExecutionOwnershipRevocationTrustNotConfiguredError(
            f"{_ENV_MAC_KEY} is required for durable revocation trust",
        )
    return WorkerExecutionOwnershipRevocationTrustMaterial(mac_key=raw.encode("utf-8"))


def bind_worker_execution_ownership_revocation_trust_for_qualification() -> None:
    configure_worker_execution_ownership_revocation_trust(
        WorkerExecutionOwnershipRevocationTrustMaterial(mac_key=_QUALIFICATION_MAC_KEY),
    )


def reset_worker_execution_ownership_revocation_trust_for_tests() -> None:
    global _TRUST
    _TRUST = None


def compute_worker_execution_ownership_revocation_authorization_mac(
    authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
    *,
    trust: WorkerExecutionOwnershipRevocationTrustMaterial,
) -> str:
    payload = f"{authorization.authorization_id}:{authorization.authorization_digest}".encode(
        "utf-8",
    )
    return hmac.new(trust.mac_key, payload, hashlib.sha256).hexdigest()


def verify_worker_execution_ownership_revocation_authorization_mac(
    authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
    *,
    trust: WorkerExecutionOwnershipRevocationTrustMaterial,
) -> None:
    expected = compute_worker_execution_ownership_revocation_authorization_mac(
        authorization,
        trust=trust,
    )
    if not hmac.compare_digest(authorization.authorization_mac, expected):
        raise WorkerExecutionOwnershipRevocationAuthorizationRejectedError(
            "revocation authorization MAC mismatch",
        )


__all__ = [
    "WorkerExecutionOwnershipRevocationTrustMaterial",
    "bind_worker_execution_ownership_revocation_trust_for_qualification",
    "compute_worker_execution_ownership_revocation_authorization_mac",
    "configure_worker_execution_ownership_revocation_trust",
    "require_worker_execution_ownership_revocation_trust",
    "reset_worker_execution_ownership_revocation_trust_for_tests",
    "resolve_worker_execution_revocation_trust_from_environment",
    "verify_worker_execution_ownership_revocation_authorization_mac",
]
