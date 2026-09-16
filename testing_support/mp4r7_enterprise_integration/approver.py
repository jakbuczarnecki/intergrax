# © Artur Czarnecki. All rights reserved.

"""Explicit test-provider human approver evidence for MP-4R7 (not local-development fallback)."""

from __future__ import annotations

from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.human_approver import HumanApproverAuthMode, HumanApproverEvidence

_QUALIFICATION_APPROVER_FACTORY_CALLS = 0


def qualification_approver_factory_call_count() -> int:
    return _QUALIFICATION_APPROVER_FACTORY_CALLS


def reset_qualification_approver_factory_call_count() -> None:
    global _QUALIFICATION_APPROVER_FACTORY_CALLS
    _QUALIFICATION_APPROVER_FACTORY_CALLS = 0


def qualification_identity_provider_approver_evidence(
    *,
    tenant_id: str,
    actor_id: str = "mp4r7-qualification-operator",
) -> HumanApproverEvidence:
    """Configured qualification principal — identity-provider auth mode, fail-closed if misused in prod."""
    global _QUALIFICATION_APPROVER_FACTORY_CALLS
    _QUALIFICATION_APPROVER_FACTORY_CALLS += 1
    return HumanApproverEvidence(
        tenant_id=tenant_id,
        user_id=actor_id,
        principal_type=PrincipalType.USER,
        auth_subject=f"idp:subject:{actor_id}",
        auth_mode=HumanApproverAuthMode.IDENTITY_PROVIDER,
    )


__all__ = [
    "qualification_approver_factory_call_count",
    "qualification_identity_provider_approver_evidence",
    "reset_qualification_approver_factory_call_count",
]
