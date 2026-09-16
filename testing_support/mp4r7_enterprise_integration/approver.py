# © Artur Czarnecki. All rights reserved.

"""Explicit test-provider human approver evidence for MP-4R7 (not local-development fallback)."""

from __future__ import annotations

from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.human_approver import HumanApproverAuthMode, HumanApproverEvidence


def qualification_identity_provider_approver_evidence(
    *,
    tenant_id: str,
    actor_id: str = "mp4r7-qualification-operator",
) -> HumanApproverEvidence:
    """Configured qualification principal — identity-provider auth mode, fail-closed if misused in prod."""
    return HumanApproverEvidence(
        tenant_id=tenant_id,
        user_id=actor_id,
        principal_type=PrincipalType.USER,
        auth_subject=f"idp:subject:{actor_id}",
        auth_mode=HumanApproverAuthMode.IDENTITY_PROVIDER,
    )


__all__ = ["qualification_identity_provider_approver_evidence"]
