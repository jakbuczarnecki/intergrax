# © Artur Czarnecki. All rights reserved.

"""Canonical human approver evidence (IDT-FIX-C identity spine)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.agent_run_enums import PrincipalType


class HumanApproverAuthMode(str, Enum):
    """Canonical auth provenance for human approver evidence."""

    IDENTITY_PROVIDER = "identity_provider"
    API_KEY = "api_key"
    LOCAL_DEVELOPMENT = "local_development"


class HumanApproverEvidence(BaseModel):
    """Provider-neutral authenticated approver principal for HITL decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    user_id: str
    principal_type: PrincipalType
    auth_subject: str
    auth_mode: HumanApproverAuthMode


def local_development_approver_evidence(
    *,
    tenant_id: str,
    actor_id: str = "local_development_operator",
) -> HumanApproverEvidence:
    """Explicit unauthenticated local-dev approver — not verified identity-provider proof."""
    return HumanApproverEvidence(
        tenant_id=tenant_id,
        user_id=actor_id,
        principal_type=PrincipalType.USER,
        auth_subject=actor_id,
        auth_mode=HumanApproverAuthMode.LOCAL_DEVELOPMENT,
    )


def human_approval_event_payload(
    *,
    task_id: str,
    pause_id: str,
    human_request_id: str,
    verdict: str,
    approver: HumanApproverEvidence,
    response_text: str | None = None,
) -> dict[str, Any]:
    """Safe forensic payload for HUMAN_APPROVAL_RECEIVED — no credential secrets."""
    return {
        "task_id": task_id,
        "pause_id": pause_id,
        "human_request_id": human_request_id,
        "verdict": verdict,
        "decision": verdict,
        "approver_user_id": approver.user_id,
        "principal_type": approver.principal_type.value,
        "auth_subject": approver.auth_subject,
        "auth_mode": approver.auth_mode.value,
        "response": response_text or verdict,
    }
