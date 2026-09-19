# © Artur Czarnecki. All rights reserved.

"""Tier-3 Multiplayer consumer — public contracts only (MP-7B boundary fixture).

Mirrors the dependency shape LKW domain/application code would use:
inject ``MeaningfulSideEffectAuthorizationPort``, never private CW modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.collaborative_work import (
    CollaborativePrincipal,
    CollaborativeWorkEnforcementRequest,
    PrincipalKind,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)


class Tier3MultiplayerAuthorizationDeniedError(PermissionError):
    """Raised when the public authorization port returns a non-permitted decision."""


class Tier3MultiplayerAuthorizationFailedError(RuntimeError):
    """Raised when the public authorization port raises (fail closed)."""


@dataclass(frozen=True, slots=True)
class Tier3MultiplayerConsumer:
    """Reference Tier-3 consumer of Multiplayer authorization via public Protocol only."""

    authorization: MeaningfulSideEffectAuthorizationPort

    def request_authorized_side_effect(
        self,
        request: CollaborativeWorkEnforcementRequest,
    ) -> MeaningfulSideEffectAuthorizationResult:
        """Ask the platform authorization port; never infer authority from IDs locally."""
        try:
            result = self.authorization.authorize(request)
        except Exception as exc:  # noqa: BLE001 — fail closed on any port failure
            raise Tier3MultiplayerAuthorizationFailedError(
                "meaningful side-effect authorization port failed",
            ) from exc
        if not result.permitted:
            raise Tier3MultiplayerAuthorizationDeniedError(
                "meaningful side-effect authorization denied",
            )
        return result


def map_lkw_actor_fixture_to_principal(
    *,
    lkw_actor_id: str,
    tenant_id: str,
    principal_kind: PrincipalKind = PrincipalKind.HUMAN,
) -> CollaborativePrincipal:
    """Test-only LKW actor → public CollaborativePrincipal mapping (not production code)."""
    return CollaborativePrincipal(
        principal_id=lkw_actor_id.strip(),
        principal_kind=principal_kind,
        tenant_id=tenant_id.strip(),
    )


def build_enforcement_request_from_lkw_scope_fixture(
    *,
    tenant_id: str,
    workspace_id: str,
    acting_principal_id: str,
    operation_id: str,
    resource_scope: str | None = None,
) -> CollaborativeWorkEnforcementRequest:
    """Test-only product scope → public enforcement request (not ManagedWorkspace mutation)."""
    return CollaborativeWorkEnforcementRequest(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        operation_id=operation_id,
        acting_principal_id=acting_principal_id,
        resource_scope=resource_scope,
    )


__all__ = [
    "Tier3MultiplayerAuthorizationDeniedError",
    "Tier3MultiplayerAuthorizationFailedError",
    "Tier3MultiplayerConsumer",
    "build_enforcement_request_from_lkw_scope_fixture",
    "map_lkw_actor_fixture_to_principal",
]
