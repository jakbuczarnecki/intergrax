# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Cross-turn legal workspace snapshot (counts/status only) stored in ChatSession.metadata.

Kept separate from :mod:`~intergrax.agents_packages.legal_agent.memory.legal_memory_policy` so agent state can
depend on the snapshot model without importing the full memory policy module.

To drop persisted hints when a session ends, hosts call :func:`clear_persisted_legal_workspace_snapshot`
(alongside their existing session-close flow).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_manager import SessionManager

LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY = "intergrax.legal_workspace_snapshot_v1"


class LegalWorkspaceSessionSnapshotV1(BaseModel):
    """
    Redacted cross-turn hints persisted in :attr:`~intergrax.runtime.nexus.session.chat_session.ChatSession.metadata`.

    Counts and status only — no clause text, recommendations, or PII payloads. Written after a successful
    legal finalize wave; read at the start of the next LegalDynamicPipeline run for routing / replan JSON.
    """

    schema_version: Literal[1] = Field(default=1)
    clause_count: int = Field(ge=0, default=0)
    sensitive_flag_count: int = Field(ge=0, default=0)
    legal_check_count: int = Field(ge=0, default=0)
    compliance_result_count: int = Field(ge=0, default=0)
    policy_violation_count: int = Field(ge=0, default=0)
    recommendation_count: int = Field(ge=0, default=0)
    uncertainty_count: int = Field(ge=0, default=0)
    has_decision: bool = False
    decision_status: Optional[str] = None
    decision_confidence: Optional[float] = None
    blocking_issues_count: int = Field(ge=0, default=0)
    decision_enforcement_modified: bool = False
    final_opinion_present: bool = False


def try_load_legal_workspace_session_snapshot(metadata: object | None) -> LegalWorkspaceSessionSnapshotV1 | None:
    """Parse snapshot from session metadata; return None if missing or invalid."""
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        return None
    raw_obj = metadata.get(LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY)
    if raw_obj is None:
        return None
    if not isinstance(raw_obj, Mapping):
        return None
    try:
        return LegalWorkspaceSessionSnapshotV1.model_validate(dict(raw_obj))
    except ValidationError:
        return None


async def clear_persisted_legal_workspace_snapshot(
    *,
    session: ChatSession,
    session_manager: SessionManager,
) -> bool:
    """
    Remove :data:`LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY` from ``session.metadata`` and save.

    Idempotent. Hosts should call this when closing or archiving a chat if policy forbids retaining
    cross-turn legal routing hints (e.g. GDPR minimisation). Does not close the session itself.
    """
    md = dict(session.metadata or {})
    if LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY not in md:
        return False
    del md[LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY]
    session.metadata = md
    await session_manager.save_session(session)
    return True
